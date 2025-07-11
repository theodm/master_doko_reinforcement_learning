extern crate core;


pub mod alpha_zero {
    pub mod alpha_zero_training;

    pub mod tic_tac_toe;
    pub mod doppelkopf;
    pub mod network;

    pub mod csv_writer_thread;
}

pub mod compare_impi {
    pub mod compare_impi;
    pub mod policy_fusion;
    pub mod all_policies;

    pub mod run;

    pub mod evaluate_single_game_impi;

    pub mod run_mcts_vs_previous;
}
pub mod gather_games {
    pub mod gather_games;
    pub mod db;
}

pub mod train_impi2 {
    pub mod train_impi2;
}

pub mod impi {
    pub mod execute;
    pub mod network;
}

pub mod view_server {
    pub mod view_server;
    pub mod api;
}
pub mod tensorboard;

pub mod single_game {
    pub mod play_single_game;
}

use std::io::Write;
use std::sync::Arc;
use numpy::ToPyArray;
use pyo3::prelude::*;
//use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
//use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use rand::prelude::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use tokio::runtime::Builder;
use rs_doko_impi::mcts_full_doko_policy::MCTSFullDokoPolicy;
use rs_doko_impi::modified_random_full_doko_policy::ModifiedRandomFullDokoPolicy;
use rs_doko_impi::network::ImpiNetwork;
use rs_doko_impi::train_impi::train_impi;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::compare_impi::run::run_comparison;
use crate::compare_impi::run_mcts_vs_previous::run_mcts_vs_previous;
use crate::gather_games::gather_games::ext_gather_games;
use crate::impi::network::PyImpiNetwork;
use crate::single_game::play_single_game::{play_single_game, play_single_game2};
use crate::tensorboard::TensorboardController;
use crate::train_impi2::train_impi2::{train_impi2, train_impi2_c};
use crate::view_server::view_server::{create_game_for_web_view, create_game_for_web_view_exec2};

#[pyfunction]
pub fn alpha_zero_training(
    py: Python,

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

    min_or_max_value: f32,

    value_target: String,

    node_arena_capacity: usize,
    state_arena_capacity: usize,
    cache_size_batch_processor: usize,

    mcts_workers: usize,
    max_concurrent_games_in_evaluation: usize,

    evaluation_every_n_epochs: usize,
    skip_evaluation: bool,

    network: PyObject,
    tensorboard_controller: PyObject,

    epoch_start: Option<usize>,
    temperature: Option<f32>,
) -> PyResult<()> {

    py.allow_threads(|| {
        crate::alpha_zero::alpha_zero_training::alpha_zero_training(
            game,

            number_of_batch_receivers,
            batch_nn_max_delay_in_ms,
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

            value_target,

            node_arena_capacity,
            state_arena_capacity,
            cache_size_batch_processor,

            mcts_workers,
            max_concurrent_games_in_evaluation,

            epoch_start,
            evaluation_every_n_epochs,
            skip_evaluation,

            network,
            tensorboard_controller
        );
    });

    Ok(())
}


#[pyfunction]
pub fn execute_impi(
    py: Python,

    simultaneous_games: usize,
    network: PyObject,
    chance_of_keeping_experience: f64,
    learn_after_x_games: usize,
    eval_every_x_epoch: usize,

    tensorboard_controller: PyObject
    // run_name: String
) -> PyResult<()> {
    let network = PyImpiNetwork {
        network: network
    };

    py.allow_threads(|| {
        crate::impi::execute::impi_start(
            simultaneous_games,
            Box::new(network),
            chance_of_keeping_experience,
            learn_after_x_games,
            eval_every_x_epoch,
            "not yet".to_string(),
            TensorboardController {
                py_controller: tensorboard_controller
            }
        );
    });


    Ok(())
}

#[pyfunction]
pub fn p_gather_games(
    py: Python,
    max_games: usize,
    az_network: PyObject,
) -> PyResult<()> {
    py.allow_threads(|| {
        ext_gather_games(max_games, az_network);
    });

    Ok(())
}

#[pyfunction]
pub fn p_create_game_for_web_view(
    py: Python,

    seed: Option<u64>
) -> PyResult<String> {
    let result = py.allow_threads(|| {
        create_game_for_web_view_exec2(seed)
    });

    Ok(result)
}

#[pyfunction]
pub fn p_train_impi2(
    py: Python,

    network: PyObject,
    az_network: PyObject,

    chance_of_keeping_experience: f64,
    number_of_experiences_per_training_step: usize,

    tensorboard_controller: PyObject
) -> PyResult<()> {
    let network = PyImpiNetwork {
        network: network
    };

    py.allow_threads(|| {
        train_impi2(
            chance_of_keeping_experience,
            number_of_experiences_per_training_step,

            Box::new(network),
            az_network,
            Arc::new(TensorboardController {
                py_controller: tensorboard_controller
            })
        );
    });


    Ok(())
}

#[pyfunction]
fn p_run_comparison(
    py: Python,

    az_ar_network: PyObject,
    mcts_ar_network: PyObject,

    az_network: PyObject,

    number_of_games: usize
) -> PyResult<()> {
    py.allow_threads(|| {
        run_comparison(
            az_ar_network,
            mcts_ar_network,

            az_network,

            number_of_games
        )
    });

    Ok(())
}

#[pyfunction]
fn p_run_mcts_vs_previous(
    py: Python,

    number_of_games: usize
) -> PyResult<()> {
    py.allow_threads(|| {
        run_mcts_vs_previous(
            number_of_games
        )
    });

    Ok(())
}

#[pyfunction]
fn p_play_single_game(
    py: Python,

    az_ar_network: PyObject,
    mcts_ar_network: PyObject,

    az_network: PyObject,

    seed: u64
) -> PyResult<()> {
    py.allow_threads(|| {
        play_single_game(
            az_ar_network,
            mcts_ar_network,

            az_network,

            seed
        )
    });

    Ok(())
}

#[pymodule(gil_used = false)]
fn rs_doko_py_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alpha_zero_training, m)?)?;
    m.add_function(wrap_pyfunction!(execute_impi, m)?)?;
    m.add_function(wrap_pyfunction!(p_gather_games, m)?)?;
    m.add_function(wrap_pyfunction!(p_create_game_for_web_view, m)?)?;
    m.add_function(wrap_pyfunction!(p_train_impi2, m)?)?;
    m.add_function(wrap_pyfunction!(p_run_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(p_run_mcts_vs_previous, m)?)?;
    m.add_function(wrap_pyfunction!(p_play_single_game, m)?)?;

    Ok(())
}


mod tests {
    #[test]
    fn test_mulitply() {
        assert_eq!(4, 2 * 2);
    }
}