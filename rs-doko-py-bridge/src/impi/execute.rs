use pyo3::pyfunction;

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
use crate::tensorboard::TensorboardController;


pub(crate) fn impi_start(
    simultaneous_games: usize,
    mut network: Box<dyn ImpiNetwork>,
    chance_of_keeping_experience: f64,
    learn_after_x_games: usize,
    eval_every_x_epoch: usize,

    run_name: String,

    tensorboard: TensorboardController
) {
    let rt = Builder::new_multi_thread()
        .enable_all()
        .enable_time()
        .build()
        .unwrap();

    rt.block_on(async {
        let available_parallelism = std::thread::available_parallelism().unwrap().get();

        println!("Available parallelism: {}", available_parallelism);

        let mcts_policy = MCTSFullDokoPolicy::new(
            available_parallelism - 1,
            45000 * 15,
            45000 * 15,
            45000,
            2f64.sqrt() * 20f64
        );

        // let mcts_policy = ModifiedRandomFullDokoPolicy::new();

        train_impi(
            available_parallelism - 1,
            1000f64,
            1000000,
            network,
            chance_of_keeping_experience,
            PlayerZeroOrientedArr::from_full([
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
            ]),
            run_name,
            learn_after_x_games as u32,
            eval_every_x_epoch as u32,
            Arc::new(tensorboard)
        ).await
    });
}

