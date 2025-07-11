use std::intrinsics::transmute;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::thread::available_parallelism;
use std::time::Duration;
use async_trait::async_trait;
use indicatif::ProgressStyle;
use pyo3::PyObject;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use tokio::runtime::Builder;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_alpha_zero::alpha_zero::net::network::Network;
use rs_doko_evaluator::full_doko::policy::az_policy::{AZWorkers, EvFullDokoAZPolicy};
use rs_doko_impi::mcts_full_doko_policy::MCTSFullDokoPolicy;
use rs_doko_impi::train_impi::{FullDokoPolicy, _ModifiedRandomFullDokoPolicy};
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::alpha_zero::network::PyAlphaZeroNetwork;
use crate::gather_games::db::SledStateDb;

async fn play_game(
    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,
    rng: &mut SmallRng,

    db: &Arc<SledStateDb>
) {
    let mut state = FdoState::new_game(rng);

    loop {
        let obs = state.observation_for_current_player();

        if obs.finished_stats.is_some() {
            return;
        }

        db.insert_state(&state)
            .unwrap();

        let current_player = obs
            .current_player
            .unwrap();

        if obs.allowed_actions_current_player.len() == 1 {
            unsafe {
                state.play_action(transmute(obs.allowed_actions_current_player.0.0));
            }

            continue;
        }

        let action = policies[current_player]
            .execute_policy(&state, &obs, rng)
            .await;

        state.play_action(action);
    }
}




pub async fn gather_games(
    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,
    simultaneous_games: usize,

    max_games: usize,
) {
    let multi_progress = indicatif::MultiProgress::new();
    let number_of_started_games_progress = multi_progress
        .add(
            indicatif::ProgressBar::new(max_games as u64)
        );
    let number_of_played_games_progress = multi_progress
        .add(
            indicatif::ProgressBar::new(max_games as u64)
        );

    number_of_started_games_progress
        .set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta}) [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"));
    number_of_played_games_progress
        .set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta}) [{per_sec}]")
            .unwrap()
            .progress_chars("#>-"));


    let number_of_started_games = Arc::new(AtomicUsize::new(0));
    let number_of_played_games = Arc::new(AtomicUsize::new(0));
    let db = Arc::new(SledStateDb::new("played_games_db").unwrap());

    // db.clear().unwrap();

    let mut sim_games_join_set =
        tokio::task::JoinSet::new();

    for i in 0..simultaneous_games {
        let number_of_played_games = number_of_played_games
            .clone();
        let number_of_started_games = number_of_started_games
            .clone();

        let policies = policies.clone();
        let db = db.clone();

        let number_of_started_games_progress = number_of_started_games_progress
            .clone();
        let number_of_played_games_progress = number_of_played_games_progress
            .clone();

        sim_games_join_set.spawn(async move {
            let mut rng = SmallRng::from_os_rng();

            loop {
                let mut _number_of_started_games = number_of_started_games
                    .load(std::sync::atomic::Ordering::Relaxed);

                if _number_of_started_games >= max_games {
                    break;
                }

                _number_of_started_games = number_of_started_games
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                number_of_started_games_progress
                    .set_position(_number_of_started_games as u64);

                play_game(
                    policies.clone(),
                    &mut rng,

                    &db
                ).await;

                number_of_played_games
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                tokio::task::yield_now().await;

                number_of_played_games_progress
                    .set_position(number_of_played_games.load(std::sync::atomic::Ordering::Relaxed) as u64);
            }
        });
    }

    sim_games_join_set
        .join_all()
        .await;

    db.close().unwrap();
}

pub fn ext_gather_games(
    max_games: usize,

    az_network: PyObject
) {
    let tokio_runtime = Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    tokio_runtime.block_on(async {
        let available_parallelism = available_parallelism().unwrap().get();

        // let network = PyAlphaZeroNetwork::new(az_network);
        //
        // let (az_workers, mut az_factory, joins) = AZWorkers::with_new_batch_processors(
        //     8192,
        //     200*15,
        //     200*15,
        //     200*2,
        //     3192,
        //     Duration::from_millis(5),
        //     2,
        //     Box::new(network)
        // );
        //
        // let az_policy = az_factory.policy(AlphaZeroEvaluationOptions {
        //     iterations: 200,
        //     dirichlet_alpha: 0.5,
        //     dirichlet_epsilon: 0.0,
        //     puct_exploration_constant: 4.0,
        //     min_or_max_value: 0.0,
        //     par_iterations: 0,
        //     virtual_loss: 0.0,
        // });

        let iterations = 500000;
        let mcts_policy = MCTSFullDokoPolicy::new(
            available_parallelism,
            500001,
            500001,
            iterations,
            3.75f64
        );

        println!("Starting gathering games");

        gather_games(
            PlayerZeroOrientedArr::from_full(
                [
                    Arc::new(mcts_policy.clone()),
                    Arc::new(mcts_policy.clone()),
                    Arc::new(mcts_policy.clone()),
                    Arc::new(mcts_policy.clone())
                ]
            ),
            available_parallelism,
            max_games
        )
            .await;

        println!("Finished gathering games");
    });
}