// use indicatif::{ProgressBar, ProgressStyle};
// use rand::prelude::{SliceRandom, SmallRng};
// use rand::SeedableRng;
// use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
// use rayon::iter::ParallelIterator;
// use rs_doko::action::action::DoAction;
// use rs_doko::observation::observation::DoObservation;
// use rs_doko::state::state::DoState;
// use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use std::cell::RefCell;
// use std::path::PathBuf;
// use itertools::Itertools;
//
// use rs_doko_evaluator::doko::evaluate_single_game::doko_evaluate_single_game;
// use rs_doko_evaluator::doko::policy::policy::EvDokoPolicy;
// use rs_doko_evaluator::doko::policy::random_policy::doko_random_policy;
// use crate::csv_writer_thread::CSVWriterThread;
//
// pub(crate) fn repeat_policy_remaining<'a>(
//     number_of_mcts_players: usize,
//     policy: &'a EvDokoPolicy<'a>,
//     remaining_policy: &'a EvDokoPolicy<'a>,
// ) -> [&'a EvDokoPolicy<'a>; 4] {
//     match number_of_mcts_players {
//         1 => [policy, remaining_policy, remaining_policy, remaining_policy],
//         2 => [policy, policy, remaining_policy, remaining_policy],
//         3 => [policy, policy, policy, remaining_policy],
//         _ => panic!("Invalid number of MCTS players: must be between 1 and 4"),
//     }
// }
//
// pub fn experiment_2() {
//     // Alle möglichen UCT-Konstanten
//     let uct_params = [1f64, 2f64.sqrt(), 5f64.sqrt(), 10f64.sqrt(), 20f64.sqrt(), 50f64.sqrt()];
//     // Anzahl Iterationen pro MCTS-Durchlauf
//     let iterations = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
//     // Wie viele MCTS-Spieler sind im Matchup? (1 oder 2 in diesem Beispiel)
//     let number_of_mcts_player = [3, 2, 1];
//     let number_of_games = 10000;
//
//     // Zufalls-Generator für Spielreihenfolge
//     let mut rng = SmallRng::from_os_rng();
//
//     pub fn mcts_sdo_policy_gen(
//         iterations: usize,
//         uct_exploration_constant: f64,
//     ) -> impl Fn(
//         &DoState,
//         &DoObservation,
//         &mut SmallRng,
//         &mut CachedMCTS<McDokoEnvState, DoAction, 4, 26>,
//     ) -> DoAction {
//         let uct_exploration_constant = uct_exploration_constant;
//         move |state, observation, rng, cached_mcts| unsafe {
//             let moves = cached_mcts.monte_carlo_tree_search(
//                 McDokoEnvState::new(state.clone(), None),
//                 uct_exploration_constant,
//                 iterations,
//                 rng,
//             );
//             let best_move = moves.iter().max_by_key(|x| x.visits).unwrap();
//             best_move.action
//         }
//     }
//
//     thread_local! {
//         static THREAD_LOCAL_INSTANCE: RefCell<CachedMCTS<McDokoEnvState, DoAction, 4, 26>> =
//             RefCell::new(CachedMCTS::new(5_000_000, 5_000_000));
//     }
//
//     // Gesamte Anzahl erwarteter Spiele (für CSVWriterThread + Fortschrittsbalken)
//     let total_number_of_games = uct_params.len()
//         * iterations.len()
//         * number_of_mcts_player.len()
//         * number_of_games;
//
//     let mut csv_writer = CSVWriterThread::new(
//         PathBuf::from("experiment_2_small_doko.csv"),
//         &[
//             "uct_param",
//             "number_of_iterations",
//             "number_of_mcts_player",
//             "game_number",
//             "points_player_0",
//             "points_player_1",
//             "points_player_2",
//             "points_player_3",
//             "total_execution_time_player_0",
//             "total_execution_time_player_1",
//             "total_execution_time_player_2",
//             "total_execution_time_player_3",
//             "avg_execution_time_player_0",
//             "avg_execution_time_player_1",
//             "avg_execution_time_player_2",
//             "avg_execution_time_player_3",
//             "previous_best_uct_param",
//             "previous_iterations",
//             "number_of_actions"
//         ],
//         Some(total_number_of_games as u64),
//         false
//     );
//
//     let mut previous_iterations = 0;
//     let mut previous_best_uct_param = 0.0;
//
//     for (index, iteration) in iterations.iter().enumerate() {
//         let mut games = Vec::new();
//
//         println!("Iteration: {}", iteration);
//         println!("Previous Iteration: {}", previous_iterations);
//         println!("Previous Best UCT Param: {}", previous_best_uct_param);
//
//         // Liste aller Spiele vorbereiten
//         for uct_param in uct_params {
//             for matchup in number_of_mcts_player {
//                 for i in 0..number_of_games {
//                     games.push((
//                         i,
//                         previous_iterations,
//                         previous_best_uct_param,
//                         uct_param,
//                         *iteration,
//                         matchup,
//                     ));
//                 }
//             }
//         }
//
//         games.shuffle(&mut rng);
//
//         println!("Number of games: {}", games.len());
//
//         let results = games
//             .par_iter()
//             .map(
//                 |(game_id, prev_it, prev_best_uct, uct_param, iteration, number_of_mcts_player)| {
//                     THREAD_LOCAL_INSTANCE.with(|instance| {
//                         let mut cached_mcts = instance.borrow_mut();
//
//                         let previous: &EvDokoPolicy = if *prev_it == 0 {
//                             &doko_random_policy
//                         } else {
//                             &mcts_sdo_policy_gen(*prev_it, *prev_best_uct)
//                         };
//
//                         let policy = mcts_sdo_policy_gen(*iteration, *uct_param);
//                         let policies = repeat_policy_remaining(*number_of_mcts_player, &policy, previous);
//
//                         let mut rng = SmallRng::from_os_rng();
//                         let mut game_creation_rng = SmallRng::seed_from_u64(*game_id as u64 * 1000);
//
//                         let result = doko_evaluate_single_game(
//                             policies,
//                             &mut game_creation_rng,
//                             &mut rng,
//                             &mut cached_mcts,
//                         );
//
//                         csv_writer.write_row(vec![
//                             uct_param.to_string(),
//                             iteration.to_string(),
//                             number_of_mcts_player.to_string(),
//                             game_id.to_string(),
//                             result.points[0].to_string(),
//                             result.points[1].to_string(),
//                             result.points[2].to_string(),
//                             result.points[3].to_string(),
//                             result.total_execution_time[0].to_string(),
//                             result.total_execution_time[1].to_string(),
//                             result.total_execution_time[2].to_string(),
//                             result.total_execution_time[3].to_string(),
//                             result.avg_execution_time[0].to_string(),
//                             result.avg_execution_time[1].to_string(),
//                             result.avg_execution_time[2].to_string(),
//                             result.avg_execution_time[3].to_string(),
//                             prev_best_uct.to_string(),
//                             prev_it.to_string(),
//                             result.number_of_actions.to_string()
//                         ]);
//
//                         // Rückgabe für spätere Aggregation
//                         (*uct_param, result.points[0])
//                     })
//                 },
//             )
//             .collect::<Vec<(f64, i32)>>();
//
//         let grouped_results = results
//             .into_iter()
//             .into_group_map_by(|(uct_param_value, _)| uct_param_value.to_string());
//
//         let aggregated_results: Vec<(String, i32)> = grouped_results
//             .into_iter()
//             .map(|(uct_param_str, group)| {
//                 let total_points = group.iter().map(|(_, points)| points).sum::<i32>();
//                 (uct_param_str, total_points)
//             })
//             .collect();
//
//         let best_uct_param = aggregated_results
//             .iter()
//             .max_by_key(|(_, total_points)| *total_points)
//             .map(|(param_str, _)| param_str.parse::<f64>().unwrap());
//
//         println!("Iterations: {:?} - The best UCT param is: {:?}", iterations, best_uct_param);
//
//         previous_best_uct_param = best_uct_param.unwrap();
//         previous_iterations = *iteration;
//     }
//
//     csv_writer.finish();
// }
