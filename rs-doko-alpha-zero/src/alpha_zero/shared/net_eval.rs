use std::fmt::Debug;
use bytemuck::Pod;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::env::env_state::AzEnvState;

pub trait StateNetEvaluator<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> {
    async fn evaluate(&self, state: &TGameState) -> ([f32; TNumberOfPlayers], [f32; TNumberOfActions]);
}
