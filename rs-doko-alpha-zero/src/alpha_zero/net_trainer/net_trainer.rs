use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::net::experience_replay_buffer3::ExperienceReplayBuffer3;
use crate::alpha_zero::net::network::Network;
use crate::env::env_state::AzEnvState;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SmallRng;
use rand::SeedableRng;
use rand_distr::num_traits::Zero;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::cell::RefCell;
use std::cmp::min;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use bytemuck::Pod;
use graphviz_rust::print;
use tensorboard_rs::summary_writer::SummaryWriter;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio::task;
use tokio::task::{JoinHandle, JoinSet};

enum AppendMessage<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    Data(Vec<[i64; TStateDim]>, Vec<[f32; TNumberOfPlayers]>, Vec<[f32; TNumberOfActions]>),
    Finished,
}

#[derive(Debug, Clone)]
pub struct NetworkTrainerSender<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    add_to_replay_buffer_sender: UnboundedSender<AppendMessage<
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    >>,

    num_learned_batches: usize,
    alpha_zero_train_options: AlphaZeroTrainOptions,
}

impl<
        const TStateDim: usize,
        const TNumberOfPlayers: usize,
        const TNumberOfActions: usize,
    > NetworkTrainerSender<TStateDim, TNumberOfPlayers, TNumberOfActions>
{
    pub fn append(
        &self,
        states: Vec<[i64; TStateDim]>,
        values: Vec<[f32; TNumberOfPlayers]>,
        policies: Vec<[f32; TNumberOfActions]>,
    ) {
        self.add_to_replay_buffer_sender
            .send(AppendMessage::Data(states, values, policies))
            .unwrap()
    }

    pub async fn append_finished(
        &mut self
    ) {
        self
            .add_to_replay_buffer_sender
            .send(AppendMessage::Finished)
            .unwrap();

        self.add_to_replay_buffer_sender.closed().await;
    }
}

struct NetworkGathererReceiver<
    TActionType: Copy + Send + Debug,
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> {
    add_to_replay_buffer_receiver: UnboundedReceiver<
        AppendMessage<TStateDim, TNumberOfPlayers, TNumberOfActions>,
    >,

    experience_replay_buffer3: Arc<
        Mutex<
            ExperienceReplayBuffer3<
                TStateDim,
                TNumberOfPlayers,
                TNumberOfActions,
            >,
        >,
    >,

    alpha_zero_train_options: AlphaZeroTrainOptions,

    phantom_1: std::marker::PhantomData<TActionType>,
    phantom_2: std::marker::PhantomData<TGameState>,
}


impl<
        TActionType: Copy + Send + Debug,
        TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
        const TStateDim: usize,
        const TNumberOfPlayers: usize,
        const TNumberOfActions: usize,
    >
    NetworkGathererReceiver<
        TActionType,
        TGameState,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    >
{
    async fn receive_append(&mut self) {
        let mut rng = SmallRng::from_os_rng();

        let mut erb = self.experience_replay_buffer3.lock().await;

        let mut i = 0;
        println!("here?");
        loop {
            let message = self.add_to_replay_buffer_receiver.recv().await.unwrap();

            println!("Received message {}", i);
            i += 1;

            match message {
                AppendMessage::Data(states, values, policies) => {
                    erb.append_slice(
                        &states,
                        &values,
                        &policies,
                    );
                }
                AppendMessage::Finished => {
                    println!("Received Finished");
                    self.add_to_replay_buffer_receiver.close();

                    return;
                }
            }
        }

    }
}


pub fn create_network_trainer<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    erb: Arc<Mutex<ExperienceReplayBuffer3<TStateDim, TNumberOfPlayers, TNumberOfActions>>>,
    alpha_zero_options: &AlphaZeroTrainOptions,
) -> NetworkTrainerSender<TStateDim, TNumberOfPlayers, TNumberOfActions> {
    let (add_to_replay_buffer_sender, add_to_replay_buffer_receiver) =
        tokio::sync::mpsc::unbounded_channel();

    let mut network_gatherer_receiver: NetworkGathererReceiver<
        TActionType,
        TGameState,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    > = NetworkGathererReceiver {
        add_to_replay_buffer_receiver,
        experience_replay_buffer3: erb.clone(),
        alpha_zero_train_options: alpha_zero_options.clone(),
        phantom_1: Default::default(),
        phantom_2: Default::default(),
    };

    task::spawn(async move {
        network_gatherer_receiver.receive_append().await;
    });

    let network_trainer_sender = NetworkTrainerSender {
        add_to_replay_buffer_sender,
        num_learned_batches: 0,
        alpha_zero_train_options: alpha_zero_options.clone(),
    };

    network_trainer_sender
}

pub struct NetworkTrainer<
    TActionType: Copy + Send + Debug,
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    erb: Arc<ExperienceReplayBuffer3<TStateDim, TNumberOfPlayers, TNumberOfActions>>,

    num_learned_batches: usize,

    phantom_data: PhantomData<TActionType>,
    phantom_data2: PhantomData<TGameState>,
}

impl<
    TActionType: Copy + Send + Debug,
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> NetworkTrainer<
    TActionType,
    TGameState,
    TStateDim,
    TNumberOfPlayers,
    TNumberOfActions,
> {
    pub fn new(
        erb: Arc<ExperienceReplayBuffer3<TStateDim, TNumberOfPlayers, TNumberOfActions>>,
    ) -> Self {
        NetworkTrainer {
            erb,
            num_learned_batches: 0,
            phantom_data: PhantomData,
            phantom_data2: PhantomData,
        }
    }

    pub fn train_network(
        &mut self,
        mut network: Box<dyn Network>,
        alpha_zero_train_options: &AlphaZeroTrainOptions,
        epoch: usize,
    ) -> Box<dyn Network> {
        let mut erb = &self.erb;

        let mut rng = SmallRng::from_os_rng();

        let (state_memory, value_memory, policy_memory) = erb.load(&mut rng);

        network.fit(state_memory.iter().flatten().copied().collect(), (value_memory.iter().flatten().copied().collect(), policy_memory.iter().flatten().copied().collect()));

        erb.clear();

        return network;
    }

}
