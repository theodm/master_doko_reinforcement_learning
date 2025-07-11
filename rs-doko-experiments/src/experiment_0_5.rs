// use std::cell::RefCell;
// use std::path::PathBuf;
// use std::sync::mpsc;
// use indicatif::{ProgressBar, ProgressStyle};
// use rand::prelude::{SliceRandom, SmallRng};
// use rand::SeedableRng;
// use rayon::iter::IntoParallelRefIterator;
// use rs_doko::action::action::DoAction;
// use rs_doko::observation::observation::DoObservation;
// use rs_doko::state::state::DoState;
// use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use rayon::iter::ParallelIterator;
// use rs_doko_evaluator::doko::evaluate_single_game::doko_evaluate_single_game;
// use rs_doko_evaluator::doko::policy::policy::EvDokoPolicy;
// use rs_doko_evaluator::doko::policy::random_policy::doko_random_policy;
// use crate::csv_writer_thread::CSVWriterThread;
//
// pub fn experiment_0_5() {
//     rayon::ThreadPoolBuilder::new()
//         .num_threads(2)
//         .build_global()
//         .unwrap();
//
//     let uct_params = [1f64.sqrt(), 2f64.sqrt(), 5f64.sqrt(), 10f64.sqrt(), 50f64.sqrt()];
//     let iterations = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000];
//     let number_of_games = 25;
//
//     let mut rng = SmallRng::from_os_rng();
//
//     pub fn mcts_sdo_policy_gen(
//         iterations: usize,
//         uct_exploration_constant: f64
//     ) -> impl Fn(&DoState, &DoObservation, &mut SmallRng, &mut CachedMCTS<McDokoEnvState, DoAction, 4, 26>) -> DoAction {
//         let uct_exploration_constant = uct_exploration_constant;
//
//         move |state, observation, rng, cached_mcts| {
//             unsafe {
//                 let mut moves = cached_mcts.monte_carlo_tree_search(
//                     McDokoEnvState::new(state.clone(), None),
//                     uct_exploration_constant,
//                     iterations,
//                     rng
//                 );;
//
//                 let best_move = moves.iter()
//                     .max_by_key(|x| x.visits)
//                     .unwrap();
//
//                 return best_move.action;
//             }
//         }
//     }
//
//     let mut games = Vec::with_capacity(uct_params.len() * iterations.len() * number_of_games);
//
//     for uct_param in uct_params {
//         for iteration in iterations {
//             for i in 0..number_of_games {
//                 games.push((
//                     i,
//                     uct_param,
//                     iteration
//                 ))
//             }
//         }
//     }
//
//     games.shuffle(&mut rng);
//
//     thread_local! {
//         static THREAD_LOCAL_INSTANCE: RefCell<CachedMCTS<McDokoEnvState, DoAction, 4, 26>> = RefCell::new(CachedMCTS::new(8_000_000, 8_000_000));
//     }
//
//     let number_of_games = games.len();
//
//     let csv_writer_thread = CSVWriterThread::new(
//         PathBuf::from("experiment0.5_small_doko.csv"),
//         &[
//             "uct_param",
//             "number_of_iterations",
//             "game_number",
//             "game_execution_time",
//             "total_execution_time_player_0",
//             "total_execution_time_player_1",
//             "total_execution_time_player_2",
//             "total_execution_time_player_3",
//             "avg_execution_time_player_0",
//             "avg_execution_time_player_1",
//             "avg_execution_time_player_2",
//             "avg_execution_time_player_3"
//         ],
//         Some(number_of_games as u64),
//         false
//     );
//
//     games
//         .par_iter()
//         .for_each(|(game_id, uct_param, iteration)| {
//             THREAD_LOCAL_INSTANCE.with(|instance| {
//                 let mut cached_mcts = instance.borrow_mut();
//
//                 let game_creation_seed = *game_id as u64 * 1000;
//
//                 let policy = mcts_sdo_policy_gen(*iteration, *uct_param);
//                 let policies: [&EvDokoPolicy; 4] = if *iteration == 0 {
//                     [&doko_random_policy, &doko_random_policy, &doko_random_policy, &doko_random_policy]
//                 } else {
//                     [&policy, &policy, &policy, &policy]
//                 };
//
//                 let mut rng = SmallRng::from_os_rng();
//                 let mut game_creation_rng = SmallRng::seed_from_u64(game_creation_seed);
//
//                 let start_time = std::time::Instant::now();
//
//                 let result = doko_evaluate_single_game(
//                     policies,
//
//                     &mut game_creation_rng,
//                     &mut rng,
//
//                     &mut cached_mcts
//                 );
//
//                 let end_time = std::time::Instant::now();
//                 let execution_time = end_time
//                     .duration_since(start_time)
//                     .as_secs_f64();
//
//                 csv_writer_thread.write_row(vec![
//                     uct_param.to_string(),
//                     iteration.to_string(),
//                     game_id.to_string(),
//                     execution_time.to_string(),
//                     result.total_execution_time[0].to_string(),
//                     result.total_execution_time[1].to_string(),
//                     result.total_execution_time[2].to_string(),
//                     result.total_execution_time[3].to_string(),
//                     result.avg_execution_time[0].to_string(),
//                     result.avg_execution_time[1].to_string(),
//                     result.avg_execution_time[2].to_string(),
//                     result.avg_execution_time[3].to_string()
//                 ]);
//             });
//
//         });
// }