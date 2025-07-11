use rand_distr::num_traits::Zero;
use std::fmt::Debug;
use std::rc::Rc;
use bytemuck::Pod;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::alpha_zero::net::network::Network;
use crate::alpha_zero::shared::net_eval::StateNetEvaluator;
use crate::env::env_state::AzEnvState;

pub struct SingleStateNetEvaluator {
    network: Rc<dyn Network>
}


impl<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> StateNetEvaluator<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions> for crate::alpha_zero::shared::single_net_eval::SingleStateNetEvaluator {
    async fn evaluate(&self, state: &TGameState) -> ([f32; TNumberOfPlayers], [f32; TNumberOfActions]) {
        let mut encoded_state = [0; TStateDim];
        state.encode_into_memory(&mut encoded_state);

        let (value_t, policy_t) = self.network.predict(encoded_state.as_slice().to_vec());

        let mut value_result_memory = [0.0f32; TNumberOfPlayers];
        let mut policy_result_memory = [0.0f32; TNumberOfActions];

        value_result_memory.copy_from_slice(&value_t);
        policy_result_memory.copy_from_slice(&policy_t);

        return (value_result_memory, policy_result_memory);
    }
}