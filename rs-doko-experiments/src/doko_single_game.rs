// use rand::prelude::SmallRng;
// use rand::SeedableRng;
// use strum::EnumCount;
// use rs_doko_evaluator::full_doko::evaluate_single_game::full_doko_evaluate_single_game;
// use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
// use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use rs_full_doko::action::action::FdoAction;
// use rs_full_doko::card::cards::FdoCard;
// use rs_full_doko::display::display::display_game;
// use rs_full_doko::hand::hand::FdoHand;
// use rs_full_doko::observation::observation::FdoObservation;
// use rs_full_doko::player::player::FdoPlayer;
// use rs_full_doko::state::state::FdoState;
// use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
//
// pub fn doko_mcts_single_game() {
//     let mut rng = SmallRng::from_os_rng();
//     let mut game_creation_rng = SmallRng::seed_from_u64(15);
//
//     pub fn mcts_sdo_policy_gen(
//         iterations: usize,
//         uct_exploration_constant: f64,
//     ) -> impl Fn(
//         &FdoState,
//         &FdoObservation,
//         &mut SmallRng,
//         &mut CachedMCTS<McFullDokoEnvState, FdoAction, 4, { FdoAction::COUNT }>,
//     ) -> FdoAction {
//         let uct_exploration_constant = uct_exploration_constant;
//
//         move |state, observation, rng, cached_mcts| unsafe {
//             let mut moves = cached_mcts.monte_carlo_tree_search(
//                 McFullDokoEnvState::new(state.clone(), None),
//                 uct_exploration_constant,
//                 iterations,
//                 rng,
//             );
//
//             println!("Moves: {:?}", moves);
//
//             let best_move = moves.iter().max_by_key(|x| x.visits).unwrap();
//
//             println!("Best move: {:?}", best_move);
//
//             return best_move.action;
//         }
//     }
//
//     let mut state = FdoState::new_game(&mut game_creation_rng);
//
//     println!("{}", display_game(state.observation_for_current_player()));
//
//     let mut game_creation_rng = SmallRng::seed_from_u64(20);
//
//     let policy = mcts_sdo_policy_gen(10000, 12f64 * 2f64.sqrt());
//
//     let policies: [&EvFullDokoPolicy; 4] = [&policy, &policy, &policy, &policy];
//
//     let mut cached_mcts = CachedMCTS::new(800000, 800000);
//
//     let result =
//         full_doko_evaluate_single_game(policies, &mut game_creation_rng, &mut rng, &mut cached_mcts);
//
//
// }
