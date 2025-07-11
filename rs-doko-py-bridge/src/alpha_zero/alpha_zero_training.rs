use std::sync::Arc;
use std::time::Duration;
use pyo3::PyObject;
use tokio::runtime::Builder;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train::train_alpha_zero;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
use rs_doko_alpha_zero::env::envs::full_doko::full_doko::FdoAzEnvState;
use rs_full_doko::state::state::FdoState;
use rs_tictactoe::tictactoe::TicTacToeState;
use crate::alpha_zero::doppelkopf::AzFullDokoEvaluator;
use crate::alpha_zero::network::PyAlphaZeroNetwork;
use crate::alpha_zero::tic_tac_toe::AzTicTacToeEvaluator;
use crate::tensorboard::TensorboardController;

pub fn alpha_zero_training(
    // "tic-tac-toe" or "doppelkopf"
    game: String,

    number_of_batch_receivers: usize,
    batch_nn_max_delay_in_ms: usize,
    batch_nn_buffer_size: usize,

    checkpoint_every_n_epochs: usize,
    probability_of_keeping_experience: f32,

    games_per_epoch: usize,
    max_concurrent_games: usize,

    mcts_iterations: usize,

    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,

    puct_exploration_constant: f32,

    temperature: Option<f32>,
    min_or_max_value: f32,

    value_target: String,

    node_arena_capacity: usize,
    state_arena_capacity: usize,
    cache_size_batch_processor: usize,

    mcts_workers: usize,
    max_concurrent_games_in_evaluation: usize,

    epoch_start: Option<usize>,
    evaluation_every_n_epochs: usize,
    skip_evaluation: bool,

    network: PyObject,
    tensorboard_controller: PyObject
) {
    let rt = Builder::new_multi_thread()
        .enable_all()
        .enable_time()
        .build()
        .unwrap();

    rt.block_on(async {
        let evaluation_fn = AzFullDokoEvaluator {};

        let tensorboard_sender = TensorboardController::new(tensorboard_controller);

        train_alpha_zero(
           |rng| FdoAzEnvState::new(FdoState::new_game(rng), None),
            Box::new(PyAlphaZeroNetwork::new(network)),
            AlphaZeroTrainOptions {
                number_of_batch_receivers,
                batch_nn_max_delay: Duration::from_millis(batch_nn_max_delay_in_ms as u64),
                batch_nn_buffer_size,

                checkpoint_every_n_epochs,
                probability_of_keeping_experience,

                games_per_epoch,
                max_concurrent_games,

                mcts_iterations,

                dirichlet_alpha,
                dirichlet_epsilon,

                puct_exploration_constant,

                temperature,
                min_or_max_value,

                value_target: match value_target.as_str() {
                    "default" => ValueTarget::Default,
                    "greedy" => ValueTarget::Greedy,
                    "avg" => ValueTarget::Avg,
                    _ => panic!("Invalid value_target")
                },

                node_arena_capacity,
                state_arena_capacity,
                cache_size_batch_processor,
                mcts_workers,
                max_concurrent_games_in_evaluation,
                epoch_start,
                evaluation_every_n_epochs,
                skip_evaluation,
            },
            Arc::new(evaluation_fn),
            Arc::new(tensorboard_sender)
        ).await;

    });
}