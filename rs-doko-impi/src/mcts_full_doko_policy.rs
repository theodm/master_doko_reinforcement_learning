use async_trait::async_trait;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use strum::EnumCount;
use tokio::sync::oneshot::Sender;
use tokio::task;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use crate::get_weighted_action;
use crate::train_impi::FullDokoPolicy;

#[derive(Debug, Clone)]
pub struct MCTSFullDokoPolicy {
    sender: async_channel::Sender<(FdoState, tokio::sync::oneshot::Sender<FdoAction>)>
}

impl MCTSFullDokoPolicy {

    pub fn new(
        number_of_az_workers: usize,
        expected_number_of_nodes: usize,
        expected_number_of_states: usize,

        iterations: usize,
        uct: f64,
    ) -> Self {
        let (mcts_sender, mcts_receiver) = async_channel::unbounded();

        for i in 0..number_of_az_workers {
            let mcts_receiver = mcts_receiver
                .clone();

            task::spawn(async move {
                let mut mcts: CachedMCTS<
                    McFullDokoEnvState,
                    FdoAction,
                    4,
                    { FdoAction::COUNT }
                > = CachedMCTS::new(
                    expected_number_of_nodes,
                    expected_number_of_states,
                );

                let mut rng = SmallRng::from_os_rng();

                loop {
                    let (state, callback): (FdoState, Sender<FdoAction>) = mcts_receiver
                        .recv()
                        .await
                        .unwrap();

                    unsafe {
                        let res = tokio::task::spawn_blocking(move || {
                            let mut mcts = mcts;
                            let mut rng = rng;

                            let moves = mcts.monte_carlo_tree_search(
                                McFullDokoEnvState::new(state, None),
                                uct,
                                iterations,
                                &mut rng
                            );

                            let best_move = moves.iter()
                                .max_by_key(|x| x.visits)
                                .unwrap();

                            (mcts, rng, best_move
                                .action)
                        });

                        let (_mcts, _rng, best_move) = res
                            .await
                            .unwrap();

                        mcts = _mcts;
                        rng = _rng;

                        callback
                            .send(best_move)
                            .unwrap();
                    }
                }
            });
        }

        MCTSFullDokoPolicy {
            sender: mcts_sender
        }
    }
}

#[async_trait]
impl FullDokoPolicy for MCTSFullDokoPolicy {
    async fn execute_policy(
        &self,

        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> FdoAction {
        let (sender, receiver) = tokio::sync::oneshot::channel();

        self.sender
            .send((state.clone(), sender))
            .await
            .unwrap();

        receiver
            .await
            .unwrap()
    }
}