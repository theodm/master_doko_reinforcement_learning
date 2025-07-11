// use std::cell::RefCell;
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
// use strum::EnumCount;
// use rs_doko_evaluator::full_doko::evaluate_single_game::{full_doko_evaluate_single_game, FdoPlayedGameType};
// use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
// use rs_doko_evaluator::full_doko::policy::random_policy::full_doko_random_policy;
// use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
// use rs_full_doko::action::action::FdoAction;
// use rs_full_doko::announcement::announcement::FdoAnnouncement;
// use rs_full_doko::observation::observation::FdoObservation;
// use rs_full_doko::reservation::reservation::FdoReservation;
// use rs_full_doko::state::state::FdoState;
//
// /// Wiederholt die gegebene Policy für die angegebene Anzahl von MCTS-Spielern und füllt die restlichen Slots mit zufälligen Policies.
// fn repeat_policy_remaining_random<'a>(
//     number_of_mcts_players: usize,
//     policy: &'a EvFullDokoPolicy<'a>,
// ) -> [&'a EvFullDokoPolicy<'a>; 4] {
//     match number_of_mcts_players {
//         1 => [
//             policy,
//             &full_doko_random_policy,
//             &full_doko_random_policy,
//             &full_doko_random_policy,
//         ],
//         2 => [
//             policy,
//             policy,
//             &full_doko_random_policy,
//             &full_doko_random_policy,
//         ],
//         3 => [
//             policy,
//             policy,
//             policy,
//             &full_doko_random_policy,
//         ],
//         _ => panic!("Invalid number of MCTS players: must be between 1 and 4"),
//     }
// }
// pub fn experiment_1_fdo() {
//     // Das erste Experiment: Es soll der MCTS-Algorithmus gegen den Zufallsspieler antreten. Dabei
//     // soll eine informierte Grid-Suche durchgeführt werden, um die besten Parameter für den MCTS-Algorithmus
//     // zu finden.
//     //
//     // Folgende Werte für den UCT-Parameter sollen getestet werden: 1, 2, 5, 10, 20, 50, 100
//     // Folgende Werte für die Anzahl der Iterationen sollen getestet werden: 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000
//     //
//     // Da Doppelkopf ein 4-Spieler-Spiel ist soll die Performance jeweils in folgenden Kombinationen getestet werden:
//     //
//     // 1. 3 MCTS-Spieler gegen 1 Zufallsspieler
//     // 2. 2 MCTS-Spieler gegen 2 Zufallsspieler
//     // 3. 1 MCTS-Spieler gegen 3 Zufallsspieler
//     //
//     // Dabei wird für jede Kombination jeweils 150 Spiele durchgeführt. Die Performance wird anhand der
//     // durchschnittlichen Punktzahl eines MCTS-Spieler gemessen.
//     // Für jede Ausführung des MCTS soll die Ausführungszeit gemessen werden.
//     let uct_params = [25.0 * 2f64.sqrt(), 20.0 * 2f64.sqrt(), 20.0 * 5f64.sqrt(), 25.0 * 5f64.sqrt(), 30.0f64 * 5f64.sqrt()];
//     let iterations = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
//     let number_of_mcts_player = [3, 2, 1];
//     let number_of_games = 10000;
//
//     let mut rng = SmallRng::from_os_rng();
//
//     pub fn mcts_sdo_policy_gen(
//         iterations: usize,
//         uct_exploration_constant: f64
//     ) -> impl Fn(&FdoState, &FdoObservation, &mut SmallRng, &mut CachedMCTS<McFullDokoEnvState, FdoAction, 4, { FdoAction::COUNT }>) -> FdoAction {
//         let uct_exploration_constant = uct_exploration_constant;
//
//         move |state, observation, rng, cached_mcts| {
//             unsafe {
//                 let mut moves = cached_mcts.monte_carlo_tree_search(
//                     McFullDokoEnvState::new(state.clone(), None),
//                     uct_exploration_constant,
//                     iterations,
//                     rng
//                 );
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
//     let mut games = Vec::with_capacity(uct_params.len() * iterations.len() * number_of_mcts_player.len() * number_of_games * number_of_mcts_player.len());
//
//     for uct_param in uct_params {
//         for iteration in iterations {
//             for matchup in number_of_mcts_player {
//                 for i in 0..number_of_games {
//                     games.push((
//                         i,
//                         uct_param,
//                         iteration,
//                         matchup
//                     ))
//                 }
//
//             }
//         }
//     }
//
//     games.shuffle(&mut rng);
//
//     thread_local! {
//         static THREAD_LOCAL_INSTANCE: RefCell<CachedMCTS<McFullDokoEnvState, FdoAction, 4, { FdoAction::COUNT }>> = RefCell::new(CachedMCTS::new(200000, 200000));
//     }
//
//     let number_of_games = games.len();
//
//     let (tx, rx) = mpsc::channel::<(usize, f64, usize, usize, u64, rs_doko_evaluator::full_doko::evaluate_single_game::EvFullDokoSingleGameEvaluationResult)>();
//
//     std::thread::spawn(move || {
//         let mut pb = ProgressBar::new(number_of_games as u64);
//
//         pb
//             .set_style(
//                 ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
//                     .unwrap()
//                     .progress_chars("#>-")
//             );
//
//         let mut number_of_finished_games = 0;
//
//         let mut writer = csv::WriterBuilder::new()
//             .has_headers(true)
//             .flexible(false)
//             .from_path("experiment1_full_doko.csv")
//             .unwrap();
//
//         writer.serialize(
//             &[
//                 "uct_param",
//                 "number_of_iterations",
//                 "number_of_mcts_player",
//                 "game_number",
//                 "points_player_0",
//                 "points_player_1",
//                 "points_player_2",
//                 "points_player_3",
//                 "total_execution_time_player_0",
//                 "total_execution_time_player_1",
//                 "total_execution_time_player_2",
//                 "total_execution_time_player_3",
//                 "avg_execution_time_player_0",
//                 "avg_execution_time_player_1",
//                 "avg_execution_time_player_2",
//                 "avg_execution_time_player_3",
//                 "game_creation_seed",
//                 "number_of_actions",
//
//                 "played_game_mode",
//                 "lowest_announcement_re",
//                 "lowest_announcement_contra",
//
//                 "reservation_made_0",
//                 "reservation_made_1",
//                 "reservation_made_2",
//                 "reservation_made_3",
//
//                 "lowest_announcement_made_0",
//                 "lowest_announcement_made_1",
//                 "lowest_announcement_made_2",
//                 "lowest_announcement_made_3",
//
//                 "branching_factor"
//             ]
//         ).unwrap();
//
//         for result in rx {
//             number_of_finished_games += 1;
//
//             pb.inc(1);
//
//             writer.serialize(
//                 &[
//                     result.1.to_string(),
//                     result.2.to_string(),
//                     result.3.to_string(),
//                     result.0.to_string(),
//                     result.5.points[0].to_string(),
//                     result.5.points[1].to_string(),
//                     result.5.points[2].to_string(),
//                     result.5.points[3].to_string(),
//                     result.5.total_execution_time[0].to_string(),
//                     result.5.total_execution_time[1].to_string(),
//                     result.5.total_execution_time[2].to_string(),
//                     result.5.total_execution_time[3].to_string(),
//                     result.5.avg_execution_time[0].to_string(),
//                     result.5.avg_execution_time[1].to_string(),
//                     result.5.avg_execution_time[2].to_string(),
//                     result.5.avg_execution_time[3].to_string(),
//                     result.4.to_string(),
//                     result.5.number_of_actions.to_string(),
//
//                     result.5.played_game_mode.to_string(),
//                     result.5.lowest_announcement_re.map(|x| x.to_string()).unwrap_or("".to_string()),
//                     result.5.lowest_announcement_contra.map(|x| x.to_string()).unwrap_or("".to_string()),
//
//                     result.5.reservation_made[0].to_string(),
//                     result.5.reservation_made[1].to_string(),
//                     result.5.reservation_made[2].to_string(),
//                     result.5.reservation_made[3].to_string(),
//
//                     result.5.lowest_announcement_made[0].map(|x| x.to_string()).unwrap_or("".to_string()),
//                     result.5.lowest_announcement_made[1].map(|x| x.to_string()).unwrap_or("".to_string()),
//                     result.5.lowest_announcement_made[2].map(|x| x.to_string()).unwrap_or("".to_string()),
//                     result.5.lowest_announcement_made[3].map(|x| x.to_string()).unwrap_or("".to_string()),
//
//                     result.5.branching_factor.to_string()
//                 ]
//
//             ).unwrap();
//             writer.flush().unwrap()
//         }
//
//         pb.finish();
//     });
//
//
//     games
//         .par_iter()
//         .for_each(|(game_id, uct_param, iteration, number_of_mcts_player)| {
//             THREAD_LOCAL_INSTANCE.with(|instance| {
//                 let mut cached_mcts = instance.borrow_mut();
//
//                 let game_creation_seed = *game_id as u64 * 1000;
//
//                 let policy = mcts_sdo_policy_gen(*iteration, *uct_param);
//                 let policies = repeat_policy_remaining_random(*number_of_mcts_player, &policy);
//                 let mut rng = SmallRng::from_os_rng();
//                 let mut game_creation_rng = SmallRng::seed_from_u64(game_creation_seed);
//
//                 let result = full_doko_evaluate_single_game(
//                     policies,
//
//                     &mut game_creation_rng,
//                     &mut rng,
//
//                     &mut cached_mcts
//                 );
//
//                 tx.send((*game_id, *uct_param, *iteration, *number_of_mcts_player, game_creation_seed, result)).unwrap();
//             });
//
//         });
// }