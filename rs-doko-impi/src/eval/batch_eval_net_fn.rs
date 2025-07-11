use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use tokio::sync::oneshot::error::RecvError;
use tokio::task::JoinSet;
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};
use crate::eval::batch_processor::{create_batch_processor, BPInputTypeTraits, BPOutputTypeTraits, BPProcessor, BatchProcessorSender};
use crate::eval::eval_net_fn::EvaluateNetFn;
use crate::network::{ImpiNetwork};

unsafe impl Send for BatchNetworkEvaluateProcessor {}

#[derive(Clone)]
struct BatchNetworkEvaluateProcessor {
    network: Arc::<Box<dyn ImpiNetwork>>,

    result_memory: Vec<f32>,
}

impl BPProcessor<
    [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],
    [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]
> for BatchNetworkEvaluateProcessor {
    fn process_batch(&mut self, input: &Vec<
        [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]
    >) {
        let flattened_input: Vec<i64> =
            input
                .iter()
                .flat_map(|x| x.iter().copied())
                .collect();

        // let input_t = tch::Tensor::from_slice(flattened_input.as_slice())
        //     .view([-1, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE as i64])
        //     .to_device(self.network.device());

        let result_t = self.network.predict(flattened_input);

        self.result_memory = result_t
    }

    fn get_batch_result_by_index(&self, index: usize) -> [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE] {
        let result = self.result_memory
            [index * FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE..(index + 1) * FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]
            .try_into()
            .unwrap();

        result
    }
}

#[derive(Clone)]
pub struct BatchNetworkEvaluateNetFn {
    pub batch_processor_sender: BatchProcessorSender<
        [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],
        [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]
    >
}

impl BPInputTypeTraits for [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE] {}
impl BPOutputTypeTraits for [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE] {}


impl BatchNetworkEvaluateNetFn {
    pub fn new(
        network: Box<dyn ImpiNetwork>,

        buffer_size: usize,
        batch_timeout: std::time::Duration,
        num_processors: usize
    ) -> (BatchNetworkEvaluateNetFn, JoinSet<()>) {
        let network = Arc::new(network);

        let processors = (0..num_processors)
            .map(|_| BatchNetworkEvaluateProcessor {
                network: network.clone(),
                result_memory: vec![0.0; buffer_size * FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE],
            })
            .collect();

        let (batch_processor_sender, join_set) = create_batch_processor(
           buffer_size,
           batch_timeout,
            processors,
           num_processors,
        );

        (
            BatchNetworkEvaluateNetFn {
                batch_processor_sender
            },
            join_set
        )
    }
}

#[async_trait]
impl EvaluateNetFn for BatchNetworkEvaluateNetFn {
    async fn evaluate(
        &self,
        input: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]
    ) -> [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE] {
        let (sender, receiver) = tokio::sync::oneshot::channel();

        self.batch_processor_sender
            .mpsc_send
            .send((input, sender))
            .await
            .unwrap_or_else(|_| {
            });

        let output =
            receiver
                .await
                .unwrap_or_else(|_| [0.0; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]);

        output
    }
}