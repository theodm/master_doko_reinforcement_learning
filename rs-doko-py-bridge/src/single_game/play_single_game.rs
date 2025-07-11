use std::collections::HashMap;
use crate::alpha_zero::network::PyAlphaZeroNetwork;
use crate::impi::network::PyImpiNetwork;
use pyo3::PyObject;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSWorkers;
use rs_doko_impi::eval::batch_eval_net_fn::BatchNetworkEvaluateNetFn;
use std::sync::Arc;
use std::thread::available_parallelism;
use std::time::Duration;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use serde::__private::de::Content::Map;
use strum::IntoEnumIterator;
use tokio::runtime::Builder;
use tokio::sync::Mutex;
use rs_doko_impi::eval::eval_net_fn::EvaluateNetFn;
use rs_doko_impi::forward::{forward_pred_process_with_mask, ForwardPredResult};
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::card::cards::FdoCard::{ClubAce, ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondJack, DiamondKing, DiamondNine, DiamondQueen, DiamondTen, HeartAce, HeartJack, HeartKing, HeartNine, HeartQueen, HeartTen, SpadeAce, SpadeJack, SpadeKing, SpadeNine, SpadeQueen, SpadeTen};
use rs_full_doko::display::display::display_game;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::compare_impi::all_policies::{mcts_ar_avg_policy, mcts_pi_policy};
use crate::compare_impi::compare_impi::EvImpiPolicy;
use crate::compare_impi::evaluate_single_game_impi::full_doko_evaluate_single_game_impi;

pub fn play_single_game(
    az_ar_network: PyObject,
    mcts_ar_network: PyObject,

    az_network: PyObject,

    seed: u64
) {
    let tokio_runtime = Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    tokio_runtime.block_on(async {
        let mut rng = SmallRng::from_os_rng();

        let available_parallelism = 4;

        println!("Available parallelism: {}", available_parallelism);

        let network = PyAlphaZeroNetwork::new(az_network);

        let (mcts_workers, mut mcts_policy_factory) = MCTSWorkers::create_and_run(
            available_parallelism,
            1000001,
            1000001,
        );

        // let (az_workers, mut az_policy_factory, joinset)  = AZWorkers::with_new_batch_processors(
        //     2048,
        //     200 * 15,
        //     200 * 15,
        //     2048*400,
        //     1024,
        //     Duration::from_millis(1),
        //     2,
        //     Box::new(network)
        // );
        //
        // let az_ar_network = PyImpiNetwork {
        //     network: az_ar_network
        // };
        //
        // let (az_ar_evaluate_net_fn, az_ar_evaluate_net_fn_js) = BatchNetworkEvaluateNetFn::new(
        //     Box::new(az_ar_network),
        //     2048,
        //     Duration::from_millis(1),
        //     2
        // );
        //
        // let az_ar_evaluate_net_fn = Arc::new(az_ar_evaluate_net_fn);

        let mcts_ar_network = PyImpiNetwork {
            network: mcts_ar_network
        };

        let (mcts_ar_evaluate_net_fn, mcts_ar_evaluate_net_fn_js) = BatchNetworkEvaluateNetFn::new(
            Box::new(mcts_ar_network),
            512,
            Duration::from_millis(1),
            4,
        );

        let mcts_ar_evaluate_net_fn = Arc::new(mcts_ar_evaluate_net_fn);

        let mut rng_local = SmallRng::seed_from_u64(seed);
        let mut rng_game = SmallRng::seed_from_u64(rng_local.gen());

        let policies: [Arc<dyn EvImpiPolicy>; 4] = [
            // Arc::new(mcts_ar_avg_policy(
            //     mcts_policy_factory.clone(),
            //     200000,
            //     2.5,
            //     10,
            //     mcts_ar_evaluate_net_fn.clone()
            // )),
            //
            // Arc::new(mcts_ar_avg_policy(
            //     mcts_policy_factory.clone(),
            //     200000,
            //     2.5,
            //     10,
            //     mcts_ar_evaluate_net_fn.clone()
            // )),
            //
            // Arc::new(mcts_ar_avg_policy(
            //     mcts_policy_factory.clone(),
            //     200000,
            //     2.5,
            //     10,
            //     mcts_ar_evaluate_net_fn.clone()
            // )),
            // Arc::new(mcts_ar_avg_policy(
            //     mcts_policy_factory.clone(),
            //     200000,
            //     2.5,
            //     10,
            //     mcts_ar_evaluate_net_fn.clone()
            // ))

            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                200000,
                5.5f32,
            )),
            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                200000,
                5.5f32,
            )),
            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                200000,
                5.5f32,
            )),
            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                200000,
                5.5f32,
            )),
        ];

        full_doko_evaluate_single_game_impi(
            policies.clone(),
            &mut rng_local,
            &mut rng_game,
            true,
            false
        ).await;
    });


}

pub fn play_single_game2(
    az_ar_network: PyObject,
    mcts_ar_network: PyObject,

    az_network: PyObject,

    seed: u64
) {
    let tokio_runtime = Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    tokio_runtime.block_on(async {
        let mut rng = SmallRng::from_os_rng();

        let available_parallelism = 1;

        println!("Available parallelism: {}", available_parallelism);

        let network = PyAlphaZeroNetwork::new(az_network);

        let (mcts_workers, mut mcts_policy_factory) = MCTSWorkers::create_and_run(
            available_parallelism,
            2000001,
            2000001,
        );

        // let (az_workers, mut az_policy_factory, joinset)  = AZWorkers::with_new_batch_processors(
        //     2048,
        //     200 * 15,
        //     200 * 15,
        //     2048*400,
        //     1024,
        //     Duration::from_millis(1),
        //     2,
        //     Box::new(network)
        // );
        //
        // let az_ar_network = PyImpiNetwork {
        //     network: az_ar_network
        // };
        //
        // let (az_ar_evaluate_net_fn, az_ar_evaluate_net_fn_js) = BatchNetworkEvaluateNetFn::new(
        //     Box::new(az_ar_network),
        //     2048,
        //     Duration::from_millis(1),
        //     2
        // );
        //
        // let az_ar_evaluate_net_fn = Arc::new(az_ar_evaluate_net_fn);

        let mcts_ar_network = PyImpiNetwork {
            network: mcts_ar_network
        };

        let (mcts_ar_evaluate_net_fn, mcts_ar_evaluate_net_fn_js) = BatchNetworkEvaluateNetFn::new(
            Box::new(mcts_ar_network),
            512,
            Duration::from_millis(1),
            1,
        );

        let mcts_ar_evaluate_net_fn: Arc<dyn EvaluateNetFn> = Arc::new(mcts_ar_evaluate_net_fn);

        // let mut state = FdoState::new_game_from_hand_and_start_player(
        //     PlayerZeroOrientedArr::from_full([
        //         // Bottom
        //         FdoHand::from_vec(vec![SpadeNine, ClubKing, DiamondAce, SpadeNine, SpadeTen, HeartJack, DiamondKing, DiamondQueen, HeartQueen, DiamondTen, HeartTen, DiamondTen]),
        //         // Left
        //         FdoHand::from_vec(vec![SpadeKing, ClubAce, HeartNine, SpadeJack, DiamondJack, DiamondNine, SpadeQueen, ClubQueen, HeartJack, SpadeJack, HeartAce, ClubJack]),
        //         // Top
        //         FdoHand::from_vec(vec![SpadeAce, ClubAce, HeartAce, DiamondQueen, ClubTen, HeartQueen, HeartTen, ClubTen, DiamondAce, SpadeQueen, HeartKing, HeartNine]),
        //         // Right
        //         FdoHand::from_vec(vec![SpadeKing, ClubNine, HeartKing, SpadeTen, ClubNine, ClubQueen, DiamondKing, ClubKing, ClubJack, DiamondNine, SpadeAce, DiamondJack]),
        //     ]),
        //     FdoPlayer::BOTTOM,
        // );
        //
        // state.play_action(FdoAction::ReservationHealthy);
        // state.play_action(FdoAction::ReservationHealthy);
        // state.play_action(FdoAction::ReservationHealthy);
        // state.play_action(FdoAction::ReservationHealthy);
        // state.play_action(FdoAction::NoAnnouncement);
        // state.play_action(FdoAction::NoAnnouncement);
        // state.play_action(FdoAction::NoAnnouncement);
        // state.play_action(FdoAction::NoAnnouncement);
        // state.play_action(FdoAction::CardSpadeNine);

        let mut state = FdoState::new_game_from_hand_and_start_player(
            PlayerZeroOrientedArr::from_full([
                // Bottom
                FdoHand::from_vec(vec![HeartQueen, HeartNine, DiamondNine, HeartTen, ClubKing, SpadeKing, DiamondKing, ClubQueen, ClubKing, HeartQueen, SpadeQueen, DiamondTen]),
                // Left
                FdoHand::from_vec(vec![HeartJack, ClubJack, DiamondAce, SpadeJack, ClubNine, SpadeTen, DiamondJack, ClubAce, ClubTen, ClubQueen, ClubNine, SpadeJack]),
                // Top
                FdoHand::from_vec(vec![HeartNine, HeartJack, DiamondNine, ClubJack, ClubAce, SpadeKing, DiamondQueen, DiamondKing, SpadeNine, HeartTen, SpadeQueen, SpadeTen]),
                // Right
                FdoHand::from_vec(vec![HeartKing, DiamondJack, DiamondQueen, DiamondTen, ClubTen, SpadeAce, DiamondAce, HeartKing, SpadeNine, HeartAce, HeartAce, SpadeAce]),
            ]),
            FdoPlayer::BOTTOM,
        );
        state.play_action(FdoAction::ReservationHealthy);
        state.play_action(FdoAction::ReservationJacksSolo);
        state.play_action(FdoAction::ReservationHealthy);
        state.play_action(FdoAction::ReservationHealthy);


        println!("State: {}", display_game(state.observation_for_current_player()));

        let result_nums = Arc::new(Mutex::new(HashMap::new()));
        for card in FdoCard::iter() {
            result_nums.lock().await.insert(card, 0);
        }

        let num_results = Arc::new(Mutex::new(0));

        let mut tasks = FuturesUnordered::new();

        for i in 0..1000 {
            let state = state.clone(); // Voraussetzung: `FdoState` implementiert `Clone`
            let eval_fn = mcts_ar_evaluate_net_fn.clone();
            let result_nums = Arc::clone(&result_nums);
            let num_results = Arc::clone(&num_results);
            let mut rng = SmallRng::from_os_rng();

            tasks.push(tokio::spawn(async move {
                if i % 50 == 0 {
                    println!("Iteration: {}", i);
                }

                let fp = forward_pred_process_with_mask(&state, 0.5, 1.0, &eval_fn, &mut rng).await;

                if let ForwardPredResult::Consistent(s) = fp {
                    let mut results = result_nums.lock().await;
                    let mut count = num_results.lock().await;
                    *count += 1;

                    for card in FdoCard::iter() {
                        if s.hands[FdoPlayer::LEFT].contains(card) {
                            *results.entry(card).or_insert(0) += 1;
                        }
                    }
                }
            }));
        }

        // Await all tasks
        while let Some(res) = tasks.next().await {
            res.unwrap();
        }

        // Print results
        let results = result_nums.lock().await;
        let total = num_results.lock().await;

        for (card, count) in results.iter() {
            println!("Card: {:?}, Count: {}, Percent: {}", card, count, (*count as f64 / (*total) as f64) * 100f64);
        }
        println!("Num results: {}", total);

    })


}