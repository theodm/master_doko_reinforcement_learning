use async_trait::async_trait;
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};

#[async_trait]
pub trait EvaluateNetFn: Send + Sync {
    async fn evaluate(
        &self,
        input: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]
    ) -> [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE];
}