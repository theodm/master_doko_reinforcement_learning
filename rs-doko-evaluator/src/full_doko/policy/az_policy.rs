use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::Duration;
use async_channel::{RecvError, Sender};
use async_trait::async_trait;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use strum::EnumCount;
use tokio::task::JoinSet;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluate::{evaluate_single, evaluate_single_only_policy_head};
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::{create_batch_processor, BatchProcessorSender};
use rs_doko_alpha_zero::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::batch_processor::network_batch_processor::NetworkBatchProcessor;
use rs_doko_alpha_zero::alpha_zero::net::network::Network;
use rs_doko_alpha_zero::alpha_zero::train::glob::GlobalStats;
use rs_doko_alpha_zero::env::envs::full_doko::full_doko::FdoAzEnvState;
use rs_doko_impi::train_impi::FullDokoPolicy;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_doko_networks::full_doko::var2::encode_action::index2action;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use crate::full_doko::policy::policy::EvFullDokoPolicy;

struct ExecuteAZMessage {
    state: FdoState,

    alpha_zero_evaluation_options: AlphaZeroEvaluationOptions,

    sender: tokio::sync::oneshot::Sender<(FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }])>
}

#[derive(Debug, Clone)]
pub struct AZPolicyFactory {
    sender: Sender<ExecuteAZMessage>
}

impl AZPolicyFactory {
    pub fn policy(
        &mut self,
        alpha_zero_evaluation_options: AlphaZeroEvaluationOptions
    ) -> EvFullDokoAZPolicy {
        let policy = EvFullDokoAZPolicy {
            sender: self.sender.clone(),
            alpha_zero_evaluation_options
        };

        return policy
    }
}

pub struct AZWorkers {
    join_set: JoinSet<()>
}

impl AZWorkers {

    pub fn with_new_batch_processors(
        number_of_az_workers: usize,

        node_arena_capacity: usize,
        states_arena_capacity: usize,

        cache_size: usize,

        buffer_size: usize,
        batch_timeout: Duration,
        num_processors: usize,

        network: Box<dyn Network>

    ) -> (AZWorkers, AZPolicyFactory, JoinSet<()>) {
        let batch_processors = create_batch_processor(
            buffer_size,
            batch_timeout,
            (0..num_processors)
                .map(|i| NetworkBatchProcessor::new(
                    Arc::from(network.clone_network(i)),
                    buffer_size,
                ))
                .collect(),
            num_processors,
            Arc::new(GlobalStats::new()),
        );

        let res = Self::with_existing_batch_processor(
            number_of_az_workers,
            batch_processors.0,
            node_arena_capacity,
            states_arena_capacity,
            cache_size
        );

        return (res.0, res.1, batch_processors.1);
    }

    pub fn with_existing_batch_processor(
        number_of_az_workers: usize,

        batch_processor_sender: BatchProcessorSender<[i64; 311], ([f32; 4], [f32; 39])>,

        node_arena_capacity: usize,
        states_arena_capacity: usize,

        cache_size: usize
    ) -> (AZWorkers, AZPolicyFactory) {
        let (sender, receiver) = async_channel::unbounded();

        let mut join_set = JoinSet::new();

        let batch_processor = CachedNetworkBatchProcessorSender::<
            FdoAzEnvState,
            FdoAction,
            311,
            4,
            {FdoAction::COUNT}
        >::new(
            batch_processor_sender,
            cache_size,
            Arc::new(GlobalStats::new())
        );

        for _ in 0..number_of_az_workers {
            let receiver = receiver.clone();
            let batch_processor = batch_processor.clone();

            join_set.spawn(async move {
                let mut rng = SmallRng::from_os_rng();

                let mut nodes_arena = rs_unsafe_arena::unsafe_arena::UnsafeArena::new(node_arena_capacity);
                let mut states_arena = rs_unsafe_arena::unsafe_arena::UnsafeArena::new(states_arena_capacity);

                loop {
                    let result: ExecuteAZMessage = match receiver
                        .recv()
                        .await {
                        Ok(msg) => msg,
                        Err(recv) => {
                            panic!("Receiver error: {:?}", recv);
                        }
                    };


                    let state = result.state;
                    let alpha_zero_evaluation_options = result.alpha_zero_evaluation_options;

                    let action = if alpha_zero_evaluation_options.iterations == 0 {
                        let single_head_result = evaluate_single_only_policy_head(
                            &FdoAzEnvState::new(state.clone(), None),
                            &batch_processor,
                        ).await;

                        let moves_to_visits = single_head_result.1;

                        (index2action(single_head_result.0), moves_to_visits, [0.0f32; { FdoAction::COUNT }])
                    } else {
                        let result = evaluate_single(
                            &FdoAzEnvState::new(state.clone(), None),
                            alpha_zero_evaluation_options,
                            &batch_processor,
                            &mut nodes_arena,
                            &mut states_arena,
                            &mut rng,
                        ).await;

                        (index2action(result.0), result.1, result.2)
                    };

                    result
                        .sender
                        .send(action)
                        .unwrap();
                }});
        }

        return (AZWorkers { join_set }, AZPolicyFactory { sender })
    }

    pub  fn abort(&mut self) {
        self
            .join_set
            .abort_all();
    }
}

#[derive(Clone, Debug)]
pub struct EvFullDokoAZPolicy {
    sender: Sender<ExecuteAZMessage>,

    alpha_zero_evaluation_options: AlphaZeroEvaluationOptions
}

#[async_trait]
impl FullDokoPolicy for EvFullDokoAZPolicy {
    async fn execute_policy(
        &self,
        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> FdoAction {
        let (sender, receiver) = tokio::sync::oneshot::channel();

        self.sender.send(ExecuteAZMessage {
            state: state.clone(),
            alpha_zero_evaluation_options: self.alpha_zero_evaluation_options.clone(),
            sender
        }).await.unwrap();

        let action = receiver.await.unwrap();

        return action.0
    }
}

#[async_trait]
impl EvFullDokoPolicy for EvFullDokoAZPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut rand::rngs::SmallRng,
    ) -> (FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }]) {
        let (sender, receiver) = tokio::sync::oneshot::channel();

        // print!("Sending message");
        self.sender.send(ExecuteAZMessage {
            state: state.clone(),
            alpha_zero_evaluation_options: self.alpha_zero_evaluation_options.clone(),
            sender
        }).await.unwrap();

        let action = match receiver.await {
            Ok(action) => {action}
            Err(rec) => {
                panic!("Receiver error: {:?}", rec);
                // return (
                //     FdoAction::ReservationHealthy,
                //     [0; { FdoAction::COUNT }],
                //     [0.0f32; { FdoAction::COUNT }]
                // )
            }
        };

        // print!("Received message");

        return action
    }
}
