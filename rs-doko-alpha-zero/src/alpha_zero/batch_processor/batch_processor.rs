use rand_distr::num_traits::Zero;
use std::fmt::Debug;
use std::future::Future;
use std::rc::Rc;
use std::sync::atomic::{fence, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use async_channel::{Recv, RecvError};
use futures::future::Join;
use tokio::sync::{mpsc, oneshot, Mutex, MutexGuard, Notify, RwLock};
use tokio::time::error::Elapsed;
use tokio::time::{sleep, timeout, Timeout};
use tokio::{select, task, time};
use tokio::sync::oneshot::Sender;
use tokio::task::JoinSet;
use crate::alpha_zero::net::network::Network;
use crate::alpha_zero::train::glob::GlobalStats;

// Der Typ des Eingabe-Objekts des BatchProcessor muss diese Traits implementieren.
pub trait BPInputTypeTraits: Send {}

// Der Typ des Ausgabe-Objekts des BatchProcessor muss diese Traits implementieren.
pub trait BPOutputTypeTraits: Clone + Debug + Send {}

// Der BatchProcessor wird vom Aufrufer implementiert und gibt an,
// wie die Verarbeitung der Daten erfolgen soll. Im Falle des Batch-Aufrufs
// des KNN wird das Netzwerk aufgerufen und die Ergebnisse zurückgegeben.
pub trait BPProcessor<InputType, OutputType>: Send + Clone {
    fn process_batch(&mut self, input: &Vec<InputType>);

    fn get_batch_result_by_index(&self, index: usize) -> OutputType;
}

// Der BatchProcessorReceiver sammelt die Daten und übergibt diese
// dann an die Verarbeitungsfunktion. Wenn die Batch-Größe erreicht
// ist oder ein Timeout auftritt, wird die Verarbeitung angestoßen.
pub(crate) struct BatchProcessorReceiver<
    InputType: BPInputTypeTraits,
    OutputType: BPOutputTypeTraits,
    ProcessorType: BPProcessor<InputType, OutputType>,
> {
    pub mpmc_receiver: async_channel::Receiver<(InputType, oneshot::Sender<OutputType>)>,
    pub processor: ProcessorType,

    // Hier werden die Daten gesammelt, bis die Batch-Größe erreicht ist.
    pub buffer: Vec<InputType>,
    // Hier werden die Sender gesammelt, um die Ergebnisse zurückzugeben.
    pub sender: Vec<oneshot::Sender<OutputType>>,

    // Optionen
    pub batch_size: usize,
    pub batch_timeout: Duration,

    pub glob: Arc<GlobalStats>,
}

impl<
    InputType: BPInputTypeTraits,
    OutputType: BPOutputTypeTraits,
    ProcessorType: BPProcessor<InputType, OutputType>,
> BatchProcessorReceiver<InputType, OutputType, ProcessorType>
{
    pub fn new(
        mpmc_receiver: async_channel::Receiver<(InputType, oneshot::Sender<OutputType>)>,
        processor: ProcessorType,
        batch_size: usize,
        batch_timeout: Duration,
        glob: Arc<GlobalStats>,
    ) -> Self {
        Self {
            mpmc_receiver,

            processor,

            buffer: Vec::with_capacity(batch_size),
            sender: Vec::with_capacity(batch_size),

            batch_size,
            batch_timeout,

            glob,
        }
    }

    async fn calc(&mut self) {
        let time_at_start = std::time::Instant::now();

        // Wenn voll, dann verarbeiten...
        self.processor.process_batch(&self.buffer);

        let buffer_length = self.buffer.len();
        for i in 0..buffer_length {
            let i = buffer_length - i - 1;

            let sender = self.sender.remove(i);

            sender
                .send(self.processor.get_batch_result_by_index(i))
                .unwrap();
        }

        self.buffer.clear();

        let elapsed = time_at_start.elapsed();

        self.glob.number_of_batch_processor_hits.fetch_add(1, Ordering::Relaxed);
        self.glob.time_of_batch_processor.fetch_add(elapsed.as_millis() as usize, Ordering::Relaxed);
        self.glob.number_of_batch_processed_entries.fetch_add(buffer_length, Ordering::Relaxed);
        // print!(",");
        // println!("Batch processing took: {:?} for {buffer_length} items.", elapsed);
    }

    pub async fn handler(&mut self) {
        loop {
            let receiver_handle = self
                .mpmc_receiver
                .recv();

            let res = timeout(self.batch_timeout, receiver_handle).await;

            match res {
                Ok(received_data) => {
                    match received_data {
                        Ok(received_data) => {
                            let (data, sender) = received_data;

                            // Element hinzufügen
                            self.buffer.push(data);
                            self.sender.push(sender);

                            if self.buffer.len() == self.batch_size {
                                self.calc().await;
                            }
                        }
                        Err(error) => {
                            // Channel ist geschlossen, wir beenden die Verarbeitung.
                            println!("BatchProcessorReceiver: Channel closed: {:?}", error);
                            break;
                        }
                    }
                }
                Err(_) => {
                    // Timeout, wir stoßen eine Verarbeitung an,
                    // wenn wir wenigstens ein Element haben.
                    if self.buffer.len() == 0 {
                        continue;
                    }

                    self.calc().await;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct BatchProcessorSender<InputType: BPInputTypeTraits, OutputType: BPOutputTypeTraits> {
    pub mpsc_send: async_channel::Sender<(InputType, oneshot::Sender<OutputType>)>,
}


impl<InputType: BPInputTypeTraits, OutputType: BPOutputTypeTraits>
    BatchProcessorSender<InputType, OutputType>
{
    pub fn new(mpsc_send: async_channel::Sender<(InputType, oneshot::Sender<OutputType>)>) -> Self {
        Self { mpsc_send }
    }

    pub async fn process(&self, input: InputType) -> OutputType {
        let (sender, receiver) = oneshot::channel();

        self.mpsc_send
            .send((input, sender))
            .await
            .expect("Failed to send data to mpsc channel");

        let output = match receiver
            .await {
            Ok(ot) => ot,
            Err(recvError) => {
                panic!("Failed to receive data from mpsc channel: {:?}", recvError);
            }
        };


        return output;
    }
}

pub fn create_batch_processor<
    T: BPProcessor<InputType, OutputType> + 'static,
    InputType: BPInputTypeTraits + 'static,
    OutputType: BPOutputTypeTraits + 'static,
>(
    buffer_size: usize,
    batch_timeout: Duration,
    processors: Vec<T>,

    num_receivers: usize,
    glob: Arc<GlobalStats>,
) -> (
    BatchProcessorSender<InputType, OutputType>,
    JoinSet<()>
) {
    let (mpmc_sender, mpmc_receiver) = async_channel::unbounded();

    let mut join_set = JoinSet::new();

    for i in 0..num_receivers {
        let mut batch_processor_receiver = BatchProcessorReceiver::new(
            mpmc_receiver.clone(),
            processors[i].clone(),
            buffer_size,
            batch_timeout,
            glob.clone()
        );

        join_set.spawn(
            async move {
                batch_processor_receiver.handler().await;
            }
        );
    }

    let batch_processor = BatchProcessorSender::new(mpmc_sender);

    (batch_processor, join_set)
}
