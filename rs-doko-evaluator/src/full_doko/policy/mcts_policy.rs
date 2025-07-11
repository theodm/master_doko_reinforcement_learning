use async_channel::Sender;
use async_trait::async_trait;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use strum::EnumCount;
use tokio::task;
use tokio::task::{JoinHandle, JoinSet};
use rs_doko::action::action::DoAction;
use rs_doko::state::state::DoState;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::display::display::display_game;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use crate::full_doko::policy::policy::EvFullDokoPolicy;

pub struct MCTSWorkers {
    workers: JoinSet<Result<(), tokio::task::JoinError>>,
}

#[derive(Debug, Clone)]
pub struct MCTSFactory {
    sender: async_channel::Sender<ExecuteMCTSMessage>
}

impl MCTSFactory {

    pub fn policy(
        &self,

        uct_exploration_constant: f32,
        iterations: usize
    ) -> EvFullDokoMCTSPolicy {
        EvFullDokoMCTSPolicy {
            sender: self.sender.clone(),

            uct_exploration_constant: uct_exploration_constant,
            iterations: iterations
        }
    }

}

struct ExecuteMCTSMessage {
    state: FdoState,

    uct_exploration_constant: f32,
    iterations: usize,

    sender: tokio::sync::oneshot::Sender<(FdoAction, [usize; 39], [f32; 39])>
}

impl MCTSWorkers {
    pub fn create_and_run(
        number_of_workers: usize,

        node_arena_capacity: usize,
        state_arena_capacity: usize
    ) -> (MCTSWorkers, MCTSFactory) {
        let (sender, receiver) = async_channel::unbounded();

        let mut join_set = JoinSet::new();

        for _ in 0..number_of_workers {
            let receiver = receiver.clone();
            join_set.spawn(async move {
                let mut mcts: CachedMCTS<McFullDokoEnvState, FdoAction, 4, { FdoAction::COUNT }>
                    = CachedMCTS::new(node_arena_capacity, state_arena_capacity);

                let mut rng = SmallRng::from_os_rng();

                loop {
                    let result: ExecuteMCTSMessage = receiver
                        .recv()
                        .await
                        .unwrap();

                    let state = result.state;
                    let uct_exploration_constant = result.uct_exploration_constant;
                    let iterations = result.iterations;

                    let (_mcts, action) = tokio::task::spawn_blocking(move || {
                        let moves = unsafe {
                            mcts.monte_carlo_tree_search(
                                McFullDokoEnvState::new(state, None),

                                uct_exploration_constant as f64,
                                iterations,

                                &mut SmallRng::from_os_rng(),
                            )
                        };

                        let mut moves_to_visits = [0; { FdoAction::COUNT }];
                        let mut moves_to_values = [0.0f32; { FdoAction::COUNT }];

                        for m in moves.iter() {
                            moves_to_visits[m.action.to_index()] = m.visits;
                        }

                        for m in moves.iter() {
                            moves_to_values[m.action.to_index()] = m.value as f32;
                        }

                        let best_move = moves
                            .iter()
                            .max_by_key(|x| x.visits)
                            .unwrap();

                        (mcts, (best_move.action, moves_to_visits, moves_to_values))
                    }).await.unwrap();

                    result
                        .sender
                        .send(action)
                        .unwrap();

                    mcts = _mcts;
                }});
        }

        (Self {
            workers: join_set
        }, MCTSFactory { sender })
    }


    pub fn abort(mut self) {
        self
            .workers
            .abort_all();
    }
}

#[derive(Debug, Clone)]
pub struct EvFullDokoMCTSPolicy {
    sender: Sender<ExecuteMCTSMessage>,

    uct_exploration_constant: f32,
    iterations: usize
}

#[async_trait]
impl EvFullDokoPolicy for EvFullDokoMCTSPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut rand::rngs::SmallRng,
    ) -> (FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }]) {
        let (sender, receiver) = tokio::sync::oneshot::channel();

        // print!("Sent MCTS message\n");
        self.sender.send(ExecuteMCTSMessage {
            state: state.clone(),
            uct_exploration_constant: self.uct_exploration_constant,
            iterations: self.iterations,
            sender
        })
            .await
            .unwrap();

        let (action, moves_to_visits, moves_to_values) = receiver.await.unwrap();
        // print!("Received MCTS message\n");

        (action, moves_to_visits, moves_to_values)
    }
}
