// use std::cell::RefCell;
// use std::path::PathBuf;
//
// use rand::prelude::{SmallRng, SliceRandom};
// use rand::SeedableRng;
// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
// use itertools::Itertools;
// use strum::EnumCount;
//
// use rs_full_doko::state::state::FdoState;
// use rs_full_doko::observation::observation::FdoObservation;
// use rs_full_doko::action::action::FdoAction;
// use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use rs_doko_evaluator::full_doko::evaluate_single_game::full_doko_evaluate_single_game;
// use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
//
// use crate::csv_writer_thread::CSVWriterThread;
//
// #[derive(Clone)]
// pub struct PolicyConfig {
//     pub uct: f64,
//     pub iterations: usize,
//     pub desc: &'static str,
// }
//
// pub fn mcts_sdo_policy_gen(
//     iterations: usize,
//     uct_exploration_constant: f64,
// ) -> Box<EvFullDokoPolicy<'static>> {
//     Box::new(move |state, _observation, rng, cached_mcts| unsafe {
//         let moves = cached_mcts.monte_carlo_tree_search(
//             McFullDokoEnvState::new(state.clone(), None),
//             uct_exploration_constant,
//             iterations,
//             rng,
//         );
//         let best_move = moves.iter().max_by_key(|x| x.visits).unwrap();
//         best_move.action
//     })
// }
//
// thread_local! {
//     static THREAD_LOCAL_INSTANCE: RefCell<CachedMCTS<McFullDokoEnvState, FdoAction, 4, {FdoAction::COUNT}>> =
//         RefCell::new(CachedMCTS::new(10_000, 10_000));
// }
//
// pub fn experiment_4_fdo() {
//     let policy_configs = vec![
//         PolicyConfig {
//             uct: (1.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=sqrt(1.0), it=1000",
//         },
//         PolicyConfig {
//             uct: (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=sqrt(10.0), it=1000",
//         // },
//         PolicyConfig {
//             uct: 5f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=5*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 5f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=5*sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: 5f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=5*sqrt(10.0), it=1000",
//         // },
//         PolicyConfig {
//             uct: 10f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=10*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 10f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=10*sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: 10f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=10*sqrt(10.0), it=1000",
//         // },
//         PolicyConfig {
//             uct: 15f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=15*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 15f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=15*sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: 15f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=15*sqrt(10.0), it=1000",
//         // },
//         // PolicyConfig {
//         //     uct: 10f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=10*sqrt(10.0), it=1000",
//         // },
//         PolicyConfig {
//             uct: 20f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=20*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 20f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=20*sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: 20f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=20*sqrt(10.0), it=1000",
//         // },
//         PolicyConfig {
//             uct: 25f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=25*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 25f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=25*sqrt(5.0), it=1000",
//         },
//         // PolicyConfig {
//         //     uct: 25f64 * (10.0f64).sqrt(),
//         //     iterations: 1000,
//         //     desc: "UCT=25*sqrt(10.0), it=1000",
//         // },
//
//         PolicyConfig {
//             uct: 30f64 * (2.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=30*sqrt(2.0), it=1000",
//         },
//         PolicyConfig {
//             uct: 30f64 * (5.0f64).sqrt(),
//             iterations: 1000,
//             desc: "UCT=30*sqrt(5.0), it=1000",
//         },
//         // usw... (beliebig erweiterbar)
//     ];
//
//     let all_combinations = policy_configs.iter().combinations(4).collect::<Vec<_>>();
//
//     let number_of_games = 400;
//
//     let total_number_of_games = all_combinations.len() * number_of_games;
//     let mut csv_writer = CSVWriterThread::new(
//         PathBuf::from("experiment_4_all_against_all.csv"),
//         &[
//             // Neben UCT und Iterationen packen wir jetzt auch "desc_pX" in die CSV
//             "desc_p0", "uct_param_p0", "iterations_p0",
//             "desc_p1", "uct_param_p1", "iterations_p1",
//             "desc_p2", "uct_param_p2", "iterations_p2",
//             "desc_p3", "uct_param_p3", "iterations_p3",
//             "game_number",
//             "points_player_0",
//             "points_player_1",
//             "points_player_2",
//             "points_player_3",
//             "total_execution_time_0",
//             "total_execution_time_1",
//             "total_execution_time_2",
//             "total_execution_time_3",
//             "avg_execution_time_0",
//             "avg_execution_time_1",
//             "avg_execution_time_2",
//             "avg_execution_time_3",
//             "number_of_actions",
//             "played_game_mode",
//             "lowest_announcement_re",
//             "lowest_announcement_contra",
//             "reservation_made_0",
//             "reservation_made_1",
//             "reservation_made_2",
//             "reservation_made_3",
//             "lowest_announcement_made_0",
//             "lowest_announcement_made_1",
//             "lowest_announcement_made_2",
//             "lowest_announcement_made_3",
//             "branching_factor",
//         ],
//         Some(total_number_of_games as u64),
//         false
//     );
//
//     let mut rng = SmallRng::from_os_rng();
//     let mut scenario_list = Vec::with_capacity(total_number_of_games);
//     for combo in &all_combinations {
//         for game_id in 0..number_of_games {
//             let c = combo.iter().map(|pc| (*pc).clone()).collect::<Vec<_>>();
//             scenario_list.push((c, game_id));
//         }
//     }
//
//     scenario_list.shuffle(&mut rng);
//
//     scenario_list.par_iter().for_each(|(combo_of_four, game_id)| {
//         THREAD_LOCAL_INSTANCE.with(|instance| {
//             let mut cached_mcts = instance.borrow_mut();
//
//             let mut rng = SmallRng::from_os_rng();
//
//             let mut policies = combo_of_four
//                 .iter()
//                 .map(|pc| {
//                     mcts_sdo_policy_gen(pc.iterations, pc.uct)
//                 })
//                 .collect::<Vec<_>>();
//
//             policies.shuffle(&mut rng);
//
//             let policy_refs: [&EvFullDokoPolicy<'static>; 4] = [
//                 &*policies[0],
//                 &*policies[1],
//                 &*policies[2],
//                 &*policies[3],
//             ];
//
//             let mut game_creation_rng = SmallRng::seed_from_u64(*game_id as u64);
//
//             let result = full_doko_evaluate_single_game(
//                 policy_refs,
//                 &mut game_creation_rng,
//                 &mut rng,
//                 &mut cached_mcts,
//             );
//
//             let seat_pc0 = &combo_of_four[0];
//             let seat_pc1 = &combo_of_four[1];
//             let seat_pc2 = &combo_of_four[2];
//             let seat_pc3 = &combo_of_four[3];
//
//             csv_writer.write_row(vec![
//                 // p0
//                 seat_pc0.desc.to_string(),
//                 seat_pc0.uct.to_string(),
//                 seat_pc0.iterations.to_string(),
//
//                 // p1
//                 seat_pc1.desc.to_string(),
//                 seat_pc1.uct.to_string(),
//                 seat_pc1.iterations.to_string(),
//
//                 // p2
//                 seat_pc2.desc.to_string(),
//                 seat_pc2.uct.to_string(),
//                 seat_pc2.iterations.to_string(),
//
//                 // p3
//                 seat_pc3.desc.to_string(),
//                 seat_pc3.uct.to_string(),
//                 seat_pc3.iterations.to_string(),
//
//                 game_id.to_string(),
//
//                 result.points[0].to_string(),
//                 result.points[1].to_string(),
//                 result.points[2].to_string(),
//                 result.points[3].to_string(),
//                 result.total_execution_time[0].to_string(),
//                 result.total_execution_time[1].to_string(),
//                 result.total_execution_time[2].to_string(),
//                 result.total_execution_time[3].to_string(),
//                 result.avg_execution_time[0].to_string(),
//                 result.avg_execution_time[1].to_string(),
//                 result.avg_execution_time[2].to_string(),
//                 result.avg_execution_time[3].to_string(),
//                 result.number_of_actions.to_string(),
//                 result.played_game_mode.to_string(),
//                 result.lowest_announcement_re.map(|x| x.to_string()).unwrap_or_default(),
//                 result.lowest_announcement_contra.map(|x| x.to_string()).unwrap_or_default(),
//                 result.reservation_made[0].to_string(),
//                 result.reservation_made[1].to_string(),
//                 result.reservation_made[2].to_string(),
//                 result.reservation_made[3].to_string(),
//                 result.lowest_announcement_made[0].map(|x| x.to_string()).unwrap_or_default(),
//                 result.lowest_announcement_made[1].map(|x| x.to_string()).unwrap_or_default(),
//                 result.lowest_announcement_made[2].map(|x| x.to_string()).unwrap_or_default(),
//                 result.lowest_announcement_made[3].map(|x| x.to_string()).unwrap_or_default(),
//                 result.branching_factor.to_string(),
//             ]);
//         });
//     });
//
//     csv_writer.finish();
// }
