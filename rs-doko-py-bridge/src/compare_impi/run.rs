use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::available_parallelism;
use std::time::Duration;
use pyo3::PyObject;
use rand::prelude::SliceRandom;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use tokio::io::join;
use tokio::runtime::Builder;
use tokio::sync::Mutex;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_evaluator::full_doko::policy::az_policy::{AZPolicyFactory, AZWorkers};
use rs_doko_evaluator::full_doko::policy::mcts_policy::{MCTSFactory, MCTSWorkers};
use rs_doko_impi::csv_writer_thread::CSVWriterThread;
use rs_doko_impi::eval::batch_eval_net_fn::BatchNetworkEvaluateNetFn;
use crate::alpha_zero::network::PyAlphaZeroNetwork;
use crate::compare_impi::all_policies::{az_ar_avg_policy, az_ar_maxn_policy, az_pi_policy, mcts_ar_avg_policy, mcts_ar_maxn_policy, mcts_cap_avg_policy, mcts_cap_maxn_policy, mcts_pi_policy, random_policy};
use crate::compare_impi::compare_impi::{DefaultImpiPolicy, EvImpiPolicy, FullPolicy, RandomImpiPolicy};
use crate::compare_impi::evaluate_single_game_impi::full_doko_evaluate_single_game_impi;
use crate::impi::network::PyImpiNetwork;
use itertools::Itertools;
use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;

pub fn all_gegen_all(policies: &[Arc<dyn EvImpiPolicy>]) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let mut valid_combinations = Vec::new();

    for combi in policies.iter().combinations_with_replacement(4) {
        let mut counts = HashMap::new();
        let mut valid = true;

        for policy in &combi {
            let counter = counts.entry(policy.name()).or_insert(0);
            *counter += 1;
            if *counter > 3 {
                valid = false;
                break;
            }
        }

        if valid {
            let arr = [
                combi[0].clone(),
                combi[1].clone(),
                combi[2].clone(),
                combi[3].clone(),
            ];

            valid_combinations.push(arr);
        }
    }

    println!("Valid combinations: {}", valid_combinations.len());

    valid_combinations
}

pub fn only_self_play(
    policies: &[DefaultImpiPolicy]
) -> Vec<[DefaultImpiPolicy; 4]> {
    let mut valid_combinations = Vec::new();

    for policy in policies.iter() {
        valid_combinations.push([
            policy.clone(),
            policy.clone(),
            policy.clone(),
            policy.clone(),
        ]);
    }

    valid_combinations
}

// Funktion für eine statische Liste
pub fn only(policies: &[DefaultImpiPolicy]) -> Vec<[DefaultImpiPolicy; 4]> {
    let mut valid_combinations = Vec::new();

    if let Some(policy) = policies.get(0) {
        valid_combinations.push([
            policy.clone(),
            policy.clone(),
            policy.clone(),
            policy.clone(),
        ]);
    }

    valid_combinations
}

struct GameToPlay {
    game_id: usize,
    policies: [Arc<dyn EvImpiPolicy>; 4],

    player_1_name: String,
    player_2_name: String,
    player_3_name: String,
    player_4_name: String,

}

pub fn create_with_iterations_outplay(
    mcts_policy_factory: MCTSFactory
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let iterations = [
        10,
        20,
        100,
        1000,
        10000,
        100000,
        200000,
        500000,
        1000000,
        2000000
    ];

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    for i in 0..iterations.len() {
        let mcts_policy = Arc::new(mcts_pi_policy(
            mcts_policy_factory.clone(),
            iterations[i],
            5f32
        ));

        let mcts_policy2 = Arc::new(mcts_pi_policy(
            mcts_policy_factory.clone(),
            iterations[i],
            3.75f32
        ));

        let mcts_policy3 = Arc::new(mcts_pi_policy(
            mcts_policy_factory.clone(),
            iterations[i],
            2.5f32
        ));

        // Alle Kombinationen
        let policies = [
            mcts_policy.clone(),
            mcts_policy2.clone(),
            mcts_policy3.clone(),
        ];

        for a in 0..policies.len() {
            for b in a..policies.len() {
                for c in b..policies.len() {
                    for d in c..policies.len() {
                        // überspringe die Fälle, in denen a == b == c == d
                        if a == d {
                            continue;
                        }
                        valid_combinations.push([
                            policies[a].clone(),
                            policies[b].clone(),
                            policies[c].clone(),
                            policies[d].clone(),
                        ]);
                    }
                }
            }
        }
    }

    return valid_combinations;
}

pub fn create_only_with_iterations(
    mcts_policy_factory: MCTSFactory,
    mcts_ar_evaluate_net_fn: BatchNetworkEvaluateNetFn
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let iterations = [
        10,
        100,
        1000,
        10000,
        100000,
        200000,
        500000,
        1000000
    ];

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    for i in 0..iterations.len() {
        let mcts_policy = Arc::new(mcts_ar_avg_policy(
            mcts_policy_factory.clone(),
            iterations[i],
            2.5f32,
            15,
            Arc::new(mcts_ar_evaluate_net_fn.clone())
        ));

        valid_combinations.push([
            mcts_policy.clone(),
            mcts_policy.clone(),
            mcts_policy.clone(),
            mcts_policy.clone(),
        ]);
    }

    return valid_combinations;
}

pub fn create_az_vs_mcts(
    mcts_policy_factory: MCTSFactory,
    az_policy_factory: AZPolicyFactory
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let iterations = [
        0,
        100,
        1000,
        10000,
        100000,
    ];

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    for i in 0..iterations.len() {
        let mcts_policy = if iterations[i] == 0 {
            Arc::new(
                random_policy()
            )
        } else {
            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                iterations[i],
                2.5f32
            ))
        };

        let az_policy = Arc::new(az_pi_policy(
            az_policy_factory.clone(),
        ));

        valid_combinations.push([
            mcts_policy.clone(),
            mcts_policy.clone(),
            mcts_policy.clone(),
            az_policy.clone(),
        ]);

        valid_combinations.push([
            mcts_policy.clone(),
            mcts_policy.clone(),
            az_policy.clone(),
            az_policy.clone(),
        ]);

        valid_combinations.push([
            mcts_policy.clone(),
            az_policy.clone(),
            az_policy.clone(),
            az_policy.clone(),
        ]);
    }

    println!("Valid combinations: {}", valid_combinations.len());

    return valid_combinations;
}

pub fn create_vs_lesser_samples(
    mcts_policy_factory: MCTSFactory,
    mcts_ar_evaluate_net_fn: BatchNetworkEvaluateNetFn
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let samples = [5, 10, 15, 25, 50];
    let uct_value = 2.5f32;

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    for i in 0..samples.len() - 1 {
        let lesser = samples[i];
        let higher = samples[i + 1];

        let mcts_policy_lesser = Arc::new(mcts_ar_avg_policy(
            mcts_policy_factory.clone(),
            200000, // Fixed iterations
            uct_value,
            lesser,
            Arc::new(mcts_ar_evaluate_net_fn.clone()),
        ));

        let mcts_policy_higher = Arc::new(mcts_ar_avg_policy(
            mcts_policy_factory.clone(),
            200000, // Fixed iterations
            uct_value,
            higher,
            Arc::new(mcts_ar_evaluate_net_fn.clone()),
        ));

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
        ]);

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
        ]);

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
        ]);
    }

    valid_combinations
}

pub fn create_vs_lesser(
    mcts_policy_factory: MCTSFactory
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let iterations = [
        10,
        20,
        100,
        200,
        1000,
        10000,
        100000,
        500000,
        1000000,
        2000000
    ];

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    for i in 0..iterations.len() - 1 {
        let lesser = iterations[i];
        let higher = iterations[i + 1];

        let mcts_policy_lesser = if lesser == 0 {
            Arc::new(
                random_policy()
            )
        } else {
            Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                lesser,
                5f32
            ))
        };

        let mcts_policy_higher = Arc::new(mcts_pi_policy(
            mcts_policy_factory.clone(),
            higher,
            5f32
        ));

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
        ]);

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
            mcts_policy_higher.clone(),
        ]);

        valid_combinations.push([
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_lesser.clone(),
            mcts_policy_higher.clone(),
        ]);
    }

    return valid_combinations;
}

pub fn create_random_combinations(
    mcts_policy_factory: MCTSFactory
) -> Vec<[Arc<dyn EvImpiPolicy>; 4]> {
    let uct_values = [
        2.5f32
    ];

    let iterations = [
        10,
        20,
        100,
        200,
        1000,
        10000,
        100000,
        200000
    ];

    let mut valid_combinations: Vec<[Arc<dyn EvImpiPolicy>; 4]> = vec![];

    let random_policy: Arc<dyn EvImpiPolicy> = Arc::new(random_policy());

    for i in 0..uct_values.len() {
        for j in 0..iterations.len() {
            let mcts_policy = Arc::new(mcts_pi_policy(
                mcts_policy_factory.clone(),
                iterations[j],
                uct_values[i]
            ));

            valid_combinations.push([
                mcts_policy.clone(),
                mcts_policy.clone(),
                mcts_policy.clone(),
                random_policy.clone(),
            ]);

            valid_combinations.push([
                mcts_policy.clone(),
                mcts_policy.clone(),
                random_policy.clone(),
                random_policy.clone(),
            ]);

            valid_combinations.push([
                mcts_policy.clone(),
                random_policy.clone(),
                random_policy.clone(),
                random_policy.clone(),
            ]);
        }
    }

    return valid_combinations;
}

// Erster Run:
//
// let valid_combinations = all_gegen_all(
//     &[
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             2f32.sqrt()
//         )),
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             5f32.sqrt()
//         )),
//
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             10f32 * 2f32.sqrt()
//         )),
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             10f32 * 5f32.sqrt()
//         )),
//
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             20f32 * 2f32.sqrt()
//         )),
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             20f32 * 5f32.sqrt()
//         )),
//
//
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             30f32 * 2f32.sqrt()
//         )),
//         Arc::new(mcts_pi_policy(
//             mcts_policy_factory.clone(),
//             10000,
//             30f32 * 5f32.sqrt()
//         )),
//     ]
// );

pub fn run_comparison(
    az_ar_network: PyObject,
    mcts_ar_network: PyObject,

    az_network: PyObject,

    number_of_games: usize,
) {
    let tokio_runtime = Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    tokio_runtime.block_on(async {
        let mut rng = SmallRng::from_os_rng();

        let available_parallelism = available_parallelism()
            .unwrap()
            .get();

        println!("Available parallelism: {}", available_parallelism);

        let network = PyAlphaZeroNetwork::new(az_network);

        let (mcts_workers, mut mcts_policy_factory) = MCTSWorkers::create_and_run(
            available_parallelism,
            1000001,
            1000001
        );

        // let (az_workers, mut az_policy_factory, joinset)  = AZWorkers::with_new_batch_processors(
        //     4096,
        //     200 * 15,
        //     200 * 15,
        //     4096*400,
        //     2048,
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
            2048,
            Duration::from_millis(1),
            2
        );


        let all_policies: [Arc<dyn EvImpiPolicy>; 9] = [
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                500000,
                2.5f32,
                5,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                500000,
                2.5f32,
                10,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                500000,
                2.5f32,
                15,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),

            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                200000,
                2.5f32,
                5,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                200000,
                2.5f32,
                10,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                200000,
                2.5f32,
                15,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),

            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                100000,
                2.5f32,
                5,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                100000,
                2.5f32,
                10,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
            Arc::new(mcts_ar_avg_policy(
                mcts_policy_factory.clone(),
                100000,
                2.5f32,
                15,
                Arc::new(mcts_ar_evaluate_net_fn.clone())
            )),
        ];

        // let mcts_ar_evaluate_net_fn = Arc::new(mcts_ar_evaluate_net_fn);

        let valid_combinations = create_only_with_iterations(
            mcts_policy_factory.clone(),
            mcts_ar_evaluate_net_fn
        );

        println!("Valid combinations: {}", valid_combinations.len());

        let mut games_to_play = vec![];

        for combi in valid_combinations.iter() {
            for i in 0..number_of_games {
                let policies = [
                    combi[0].clone(),
                    combi[1].clone(),
                    combi[2].clone(),
                    combi[3].clone(),
                ];

                let player_1_name = policies[0].name().clone();
                let player_2_name = policies[1].name().clone();
                let player_3_name = policies[2].name().clone();
                let player_4_name = policies[3].name().clone();

                let game = GameToPlay {
                    game_id: i,

                    policies: policies,

                    player_1_name: player_1_name,
                    player_2_name: player_2_name,
                    player_3_name: player_3_name,
                    player_4_name: player_4_name
                };

                games_to_play.push(game);
            }
        }

        games_to_play.shuffle(&mut rng);

        let csv_writer = CSVWriterThread::new(
            PathBuf::from("create_only_with_iterationswhatimpilol.csv"), &[
                "game_id",

                "player_1_name",
                "player_2_name",
                "player_3_name",
                "player_4_name",

                "player_1_points",
                "player_2_points",
                "player_3_points",
                "player_4_points",

                "player_1_number_executed_actions",
                "player_2_number_executed_actions",
                "player_3_number_executed_actions",
                "player_4_number_executed_actions",

                "number_of_actions",

                "played_game_mode",

                "lowest_announcement_re",
                "lowest_announcement_contra",

                "branching_factor",

                "player_1_reservation_made",
                "player_2_reservation_made",
                "player_3_reservation_made",
                "player_4_reservation_made",

                "player_1_lowest_announcement_made",
                "player_2_lowest_announcement_made",
                "player_3_lowest_announcement_made",
                "player_4_lowest_announcement_made",

                "player_1_number_consistent",
                "player_2_number_consistent",
                "player_3_number_consistent",
                "player_4_number_consistent",

                "player_1_number_not_consistent",
                "player_2_number_not_consistent",
                "player_3_number_not_consistent",
                "player_4_number_not_consistent",

                "player_1_number_was_random_because_not_consistent",
                "player_2_number_was_random_because_not_consistent",
                "player_3_number_was_random_because_not_consistent",
                "player_4_number_was_random_because_not_consistent",

                "player_1_number_not_consistentHandSizeMismatch",
                "player_2_number_not_consistentHandSizeMismatch",
                "player_3_number_not_consistentHandSizeMismatch",
                "player_4_number_not_consistentHandSizeMismatch",

                "player_1_number_not_consistentRemainingCardsLeft",
                "player_2_number_not_consistentRemainingCardsLeft",
                "player_3_number_not_consistentRemainingCardsLeft",
                "player_4_number_not_consistentRemainingCardsLeft",

                "player_1_number_not_consistentNotInRemainingCards",
                "player_2_number_not_consistentNotInRemainingCards",
                "player_3_number_not_consistentNotInRemainingCards",
                "player_4_number_not_consistentNotInRemainingCards",

                "player_1_number_not_consistentAlreadyDiscardedColor",
                "player_2_number_not_consistentAlreadyDiscardedColor",
                "player_3_number_not_consistentAlreadyDiscardedColor",
                "player_4_number_not_consistentAlreadyDiscardedColor",

                "player_1_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_2_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_3_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_4_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",

                "player_1_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_2_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_3_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_4_number_not_consistentHasNoClubQueenButAnnouncedRe",

                "player_1_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_2_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_3_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_4_number_not_consistentHasClubQueenButAnnouncedKontra",

                "player_1_number_not_consistentWrongReservation",
                "player_2_number_not_consistentWrongReservation",
                "player_3_number_not_consistentWrongReservation",
                "player_4_number_not_consistentWrongReservation",

                "player_1_number_not_consistentWrongReservationClubQ",
                "player_2_number_not_consistentWrongReservationClubQ",
                "player_3_number_not_consistentWrongReservationClubQ",
                "player_4_number_not_consistentWrongReservationClubQ",

                "player_1_was_re",
                "player_2_was_re",
                "player_3_was_re",
                "player_4_was_re",
            ],
            Some(games_to_play.len() as u64),
            false
        );

        let parallel_tasks = available_parallelism * 2;
        let total_games = games_to_play.len();

        // Spiele-Queue in ein Arc<Mutex<..>>, damit Tasks sich daraus bedienen können
        let shared_games = Arc::new(Mutex::new(VecDeque::from(games_to_play)));
        let finished = Arc::new(AtomicUsize::new(0));

        let mut joinset = tokio::task::JoinSet::new();

        for _ in 0..parallel_tasks {
            let shared_games = Arc::clone(&shared_games);

            let csv_writer = csv_writer.clone();

            joinset.spawn(async move {
                loop {
                    let mut rng = SmallRng::from_os_rng();
                    let mut game_rng = SmallRng::from_os_rng();

                    // Ein Spiel aus der Queue ziehen
                    let maybe_game = {
                        let mut locked = shared_games
                            .lock()
                            .await;

                        locked.pop_front()
                    };

                    // Keine Spiele mehr?
                    let game = match maybe_game {
                        Some(g) => g,
                        None => break,
                    };

                    let game_id = game.game_id;

                    let result = full_doko_evaluate_single_game_impi(
                        game.policies,
                        &mut rng,
                        &mut game_rng,
                        false,
                        false
                    ).await;

                    csv_writer.write_row(
                        vec![
                            game_id.to_string(),

                            game.player_1_name.clone(),
                            game.player_2_name.clone(),
                            game.player_3_name.clone(),
                            game.player_4_name.clone(),

                            result.points[0].to_string(),
                            result.points[1].to_string(),
                            result.points[2].to_string(),
                            result.points[3].to_string(),

                            result.number_executed_actions[0].to_string(),
                            result.number_executed_actions[1].to_string(),
                            result.number_executed_actions[2].to_string(),
                            result.number_executed_actions[3].to_string(),

                            result.number_of_actions.to_string(),

                            result.played_game_mode.to_string(),

                            result.lowest_announcement_re.map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_contra.map(|x| x.to_string()).unwrap_or("".to_string()),

                            result.branching_factor.to_string(),

                            result.reservation_made[0].to_string(),
                            result.reservation_made[1].to_string(),
                            result.reservation_made[2].to_string(),
                            result.reservation_made[3].to_string(),

                            result.lowest_announcement_made[0].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[1].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[2].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[3].map(|x| x.to_string()).unwrap_or("".to_string()),

                            result.number_consistent[0].to_string(),
                            result.number_consistent[1].to_string(),
                            result.number_consistent[2].to_string(),
                            result.number_consistent[3].to_string(),

                            result.number_not_consistent[0].to_string(),
                            result.number_not_consistent[1].to_string(),
                            result.number_not_consistent[2].to_string(),
                            result.number_not_consistent[3].to_string(),

                            result.number_was_random_because_not_consistent[0].to_string(),
                            result.number_was_random_because_not_consistent[1].to_string(),
                            result.number_was_random_because_not_consistent[2].to_string(),
                            result.number_was_random_because_not_consistent[3].to_string(),

                            result.number_not_consistentHandSizeMismatch[0].to_string(),
                            result.number_not_consistentHandSizeMismatch[1].to_string(),
                            result.number_not_consistentHandSizeMismatch[2].to_string(),
                            result.number_not_consistentHandSizeMismatch[3].to_string(),

                            result.number_not_consistentRemainingCardsLeft[0].to_string(),
                            result.number_not_consistentRemainingCardsLeft[1].to_string(),
                            result.number_not_consistentRemainingCardsLeft[2].to_string(),
                            result.number_not_consistentRemainingCardsLeft[3].to_string(),

                            result.number_not_consistentNotInRemainingCards[0].to_string(),
                            result.number_not_consistentNotInRemainingCards[1].to_string(),
                            result.number_not_consistentNotInRemainingCards[2].to_string(),
                            result.number_not_consistentNotInRemainingCards[3].to_string(),

                            result.number_not_consistentAlreadyDiscardedColor[0].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[1].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[2].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[3].to_string(),

                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[0].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[1].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[2].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[3].to_string(),

                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[0].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[1].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[2].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[3].to_string(),

                            result.number_not_consistentHasClubQueenButAnnouncedKontra[0].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[1].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[2].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[3].to_string(),

                            result.number_not_consistentWrongReservation[0].to_string(),
                            result.number_not_consistentWrongReservation[1].to_string(),
                            result.number_not_consistentWrongReservation[2].to_string(),
                            result.number_not_consistentWrongReservation[3].to_string(),

                            result.number_not_consistentWrongReservationClubQ[0].to_string(),
                            result.number_not_consistentWrongReservationClubQ[1].to_string(),
                            result.number_not_consistentWrongReservationClubQ[2].to_string(),
                            result.number_not_consistentWrongReservationClubQ[3].to_string(),

                            result.player_was_re[0].to_string(),
                            result.player_was_re[1].to_string(),
                            result.player_was_re[2].to_string(),
                            result.player_was_re[3].to_string(),
                        ]
                    )
                }
            });
        }

        // Alle Tasks abwarten
        while let Some(res) = joinset.join_next().await {
            if let Err(err) = res {
                eprintln!("Ein Task ist fehlgeschlagen: {:?}", err);
            }
        }
    });



}