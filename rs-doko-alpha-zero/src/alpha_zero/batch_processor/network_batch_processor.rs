use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;
use bytemuck::Pod;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;
use crate::alpha_zero::batch_processor::batch_processor::{BPInputTypeTraits, BPOutputTypeTraits, BPProcessor};
use crate::alpha_zero::net::network::Network;

// Der Prozessor, der die Batches an das Netzwerk sendet. Das
// Sammeln erfolgt durch den BatchProcessorReceiver.
pub struct NetworkBatchProcessor<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    // Die maximale Größe des Batches, die wir vom
    // BatchProcessor erhalten.
    batch_size: usize,

    network: Arc<dyn Network>,

    // Auch für die Ergebnisse allokieren wir den Speicher einmalig.
    value_result_memory: Vec<f32>,
    policy_result_memory: Vec<f32>,
}

impl <
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>Clone for NetworkBatchProcessor<
    TStateDim,
    TNumberOfPlayers,
    TNumberOfActions,
> {
    fn clone(&self) -> Self {
        Self {
            network: self.network.clone(),

            batch_size: self.batch_size,

            value_result_memory: vec![0.0; self.batch_size * TNumberOfPlayers],
            policy_result_memory: vec![0.0; self.batch_size * TNumberOfActions],
        }
    }
}

impl<const TStateDim: usize, const TNumberOfPlayers: usize, const TNumberOfActions: usize> NetworkBatchProcessor<TStateDim, TNumberOfPlayers, TNumberOfActions>
{
    pub fn new(
        network: Arc<dyn Network>,

        batch_size: usize,
    ) -> Self {
        Self {
            network,

            batch_size,

            value_result_memory: vec![0.0; batch_size * TNumberOfPlayers],
            policy_result_memory: vec![0.0; batch_size * TNumberOfActions],
        }
    }
}


unsafe impl<const TStateDim: usize, const TNumberOfPlayers: usize, const TNumberOfActions: usize>
Send for NetworkBatchProcessor<TStateDim, TNumberOfPlayers, TNumberOfActions>
{
}
impl<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>
BPProcessor<
    [i64; TStateDim],
    ([f32; TNumberOfPlayers], [f32; TNumberOfActions]),
> for NetworkBatchProcessor<TStateDim, TNumberOfPlayers, TNumberOfActions>
{
    fn process_batch(&mut self, input: &Vec<[i64; TStateDim]>) {
        let flattened_input: Vec<i64> =
            input.iter().flat_map(|x| x.iter().copied()).collect();

        let (value_t, policy_t) = self
            .network
            .predict(flattened_input.as_slice().to_vec());

        self.value_result_memory = value_t;
        self.policy_result_memory = policy_t;
    }

    fn get_batch_result_by_index(
        &self,
        index: usize,
    ) -> ([f32; TNumberOfPlayers], [f32; TNumberOfActions]) {
        let value = self.value_result_memory
            [index * TNumberOfPlayers..(index + 1) * TNumberOfPlayers]
            .try_into()
            .unwrap();
        let policy = self.policy_result_memory
            [index * TNumberOfActions..(index + 1) * TNumberOfActions]
            .try_into()
            .unwrap();

        return (value, policy);
    }
}

impl<
    TStateMemoryDataType: Zero + Pod + Clone + Copy + Serialize + DeserializeOwned  + Send + Debug,
    const TStateDim: usize,
> BPInputTypeTraits for [TStateMemoryDataType; TStateDim]
{
}

impl<const TNumberOfPlayers: usize, const TNumberOfActions: usize> BPOutputTypeTraits
for ([f32; TNumberOfPlayers], [f32; TNumberOfActions])
{
}
