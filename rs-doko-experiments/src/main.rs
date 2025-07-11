pub mod experiment_1;
pub mod experiment_2;
mod experiment_1_fdo;
mod experiment_0_5;
mod experiment_0_5_fdo;
mod csv_writer_thread;
mod experiment_2_fdo;
mod experiment_3;
mod experiment_3_ttt;
mod simple_tree;
mod doko_single_game;
mod experiment_4_fdo;

use std::cell::RefCell;
use std::sync::mpsc;
use indicatif::{ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use rand::prelude::{SliceRandom, SmallRng};
use rand::SeedableRng;
use rayon::iter::IntoParallelRefIterator;
use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::random_action;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_mcts::env::env_state::McEnvState;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rayon::iter::ParallelIterator;
use tokio::runtime::{Builder, Runtime};
use rs_doko_mcts::example::tic_tac_toe_player::tic_tac_toe_player;
use crate::simple_tree::doko_mcts_simple_tree;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;


#[tokio::main]
async fn main() {
    doko_mcts_simple_tree().await;
}
