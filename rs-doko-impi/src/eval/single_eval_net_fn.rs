use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};
use crate::eval::eval_net_fn::EvaluateNetFn;
use crate::network::ImpiNetwork;

pub struct SingleNetworkEvaluateNetFn {
    pub(crate) network: Arc::<Mutex<Box<dyn ImpiNetwork>>>
}

impl SingleNetworkEvaluateNetFn {
    pub fn new(
        network: Box<dyn ImpiNetwork>
    ) -> SingleNetworkEvaluateNetFn {
        SingleNetworkEvaluateNetFn {
            network: Arc::new(Mutex::new(network))
        }
    }
}

#[async_trait]
impl EvaluateNetFn for SingleNetworkEvaluateNetFn {
    async fn evaluate(
        &self,
        input: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]
    ) -> [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE] {
        let network = self
            .network
            .lock()
            .await;
        //
        // let input = tch::Tensor::from_slice(&input)
        //     .view([1, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE as i64]);

        let output = network.predict(Vec::from(input));

        //     .view([FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE as i64]);
        //
        // let output_tensor = output
        //     .to_kind(tch::Kind::Float)
        //     .to_device(tch::Device::Cpu);

        output
            .try_into()
            .expect("Ausgabe hat nicht die erwartete Länge!")
    }
}