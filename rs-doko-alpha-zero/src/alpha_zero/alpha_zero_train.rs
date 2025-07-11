use std::cell::RefCell;
use std::fmt::Debug;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;
use async_trait::async_trait;
use bytemuck::Pod;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use tensorboard_rs::summary_writer::SummaryWriter;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::Mutex;
use tokio::task;

use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::batch_processor::batch_processor::{
    create_batch_processor, BatchProcessorReceiver, BatchProcessorSender, BPInputTypeTraits,
    BPOutputTypeTraits,
};
use crate::alpha_zero::batch_processor::network_batch_processor::NetworkBatchProcessor;
use crate::alpha_zero::net::experience_replay_buffer3::ExperienceReplayBuffer3;
use crate::alpha_zero::net::network::{Network};
use crate::alpha_zero::net_trainer::net_trainer::create_network_trainer;
use crate::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
use crate::alpha_zero::train::train_loop::alpha_zero_main_train_loop;
use crate::env::env_state::AzEnvState;

#[async_trait]
pub trait AzEvaluator<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
>: Send + Sync {
    async fn evaluate(
        &self,
        batch_processor_sender: BatchProcessorSender<[i64; TStateDim], ([f32; TNumberOfPlayers], [f32; TNumberOfActions])>,
        u1: usize,
        u2: usize,
        a: &AlphaZeroTrainOptions,
        tensorboard_sender: Arc<dyn TensorboardSender>
    );
}


pub async fn train_alpha_zero<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    create_root_state_fn: fn(&mut SmallRng) -> TGameState,
    initial_network: Box<dyn Network>,

    alpha_zero_options: AlphaZeroTrainOptions,

    evaluate_fn: Arc<dyn AzEvaluator<TStateDim, TNumberOfPlayers, TNumberOfActions>>,
    tensorboard_sender: Arc<dyn TensorboardSender>,
) {
    task::spawn(async move {
        let evaluation_fn_clone = evaluate_fn.clone();

        unsafe {
            alpha_zero_main_train_loop(
                initial_network,
                create_root_state_fn,
                alpha_zero_options,
                evaluation_fn_clone,
                tensorboard_sender
            )
                .await;
        }
    }).await.unwrap();
}


