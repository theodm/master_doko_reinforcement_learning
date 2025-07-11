use std::fmt::Debug;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

trait InputTypeTraits: Send {
}
impl InputTypeTraits for f32 {
}

trait OutputTypeTraits: Clone + Debug +  Send {
}
impl OutputTypeTraits for (f32, f32) {
}

pub(crate) struct BatchProcessorReceiver<
    InputType: InputTypeTraits,
    OutputType: OutputTypeTraits,

    ProcessorType: Processor<InputType, OutputType>
> {
    pub mpsc_receiver: mpsc::Receiver<(InputType, oneshot::Sender<OutputType>)>,
    pub processor: ProcessorType,

    pub buffer: Vec<InputType>,
    pub sender: Vec<oneshot::Sender<OutputType>>,

    pub batch_size: usize,
}

impl<
    InputType: InputTypeTraits,
    OutputType: OutputTypeTraits,
    ProcessorType: Processor<InputType, OutputType>
> BatchProcessorReceiver<
    InputType,
    OutputType,
    ProcessorType
> {

    pub fn new(
        mpsc_receiver: mpsc::Receiver<(InputType, oneshot::Sender<OutputType>)>,
        processor: ProcessorType,
        batch_size: usize,
    ) -> Self {
        Self {
            mpsc_receiver,

            processor,

            // Wir erlauben es, den Buffer zu überfüllen, um neben
            // der Verarbeitung weitere Tasks ausführen zu können.
            buffer: Vec::with_capacity(batch_size * 4),
            sender: Vec::with_capacity(batch_size * 4),

            batch_size,
        }
    }

    async fn calc(
        &mut self,
    ) {
        // Wenn voll, dann verarbeiten...
        let results = self
            .processor
            .process_batch(&self.buffer);

        let buffer_length = self.buffer.len();
        for i in 0..buffer_length {
            let i = buffer_length - i - 1;

            let sender = self
                .sender
                .remove(i);

            sender.send(
                results[i].clone()
            ).unwrap();
        }

        self.buffer.clear();
    }

    pub(crate) async fn handler(
        &mut self,
    ) {
        loop {
            let receiver_handle = self
                .mpsc_receiver
                .recv();

            let res = timeout(
                std::time::Duration::from_secs(5),
                receiver_handle
            ).await;

            match res {
                Ok(received_data) => {
                    let (data, sender) = received_data.unwrap();

                    // Element hinzufügen
                    self.buffer.push(data);
                    self.sender.push(sender);

                    if self.buffer.len() == self.batch_size {
                        self.calc().await;
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
pub(crate) struct BatchProcessor<
    InputType: InputTypeTraits,
    OutputType: OutputTypeTraits
> {
    // create mpsc channel
    pub mpsc_send: mpsc::Sender<(InputType, oneshot::Sender<OutputType>)>
}


pub trait Processor<InputType, OutputType> : Send {
    fn process_batch(&self, input: &Vec<InputType>) -> Vec<OutputType>;
}

pub fn batch_processor<
    T: Processor<InputType, OutputType> + 'static,
    InputType: InputTypeTraits + 'static,
    OutputType: OutputTypeTraits + 'static
>(
    buffer_size: usize,
    processor: T
) -> BatchProcessor<InputType, OutputType> {
    let (mpsc_sender, mpsc_receiver) = mpsc::channel(
        buffer_size
    );

    tokio::spawn(async move {
        let mut batch_processor_receiver = BatchProcessorReceiver::new(
            mpsc_receiver,
            processor,
            buffer_size
        );

        batch_processor_receiver.handler().await;
    });

    return BatchProcessor::new(
        mpsc_sender
    );
}

impl<
    InputType: InputTypeTraits,
    OutputType: OutputTypeTraits
> BatchProcessor<InputType, OutputType> {
    pub fn new(
        mpsc_send: mpsc::Sender<(InputType, oneshot::Sender<OutputType>)>
    ) -> Self {
        Self {
            mpsc_send
        }
    }

    pub async fn process(
        &mut self,
        input: InputType
    ) -> OutputType {
        let (sender, receiver) = oneshot::channel();

        self
            .mpsc_send
            .send((input, sender))
            .await
            .expect("Failed to send data to mpsc channel");

        let output = receiver.await
            .expect("Failed to receive data from oneshot channel");

        return output;
    }
}
