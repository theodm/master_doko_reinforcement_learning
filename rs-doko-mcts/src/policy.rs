use std::collections::HashMap;
use rand::prelude::SmallRng;
use rs_doko::action::action::DoAction;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_assignment::assignment::{sample_assignment, sample_assignment_full};

// pub fn mcts_policy_best_of_n_generator<
//     const TNumberOfAssignments: usize,
//     const TNumberOfIterations: usize
// >(
//     uct_exploration_constant: f64
// ) -> impl Fn(&DoState, &DoObservation, &mut SmallRng) -> DoAction {
//     let uct_exploration_constant = uct_exploration_constant;
//
//     move |state, observation, rng| {
//         mcts_policy_best_of_n::<TNumberOfAssignments, TNumberOfIterations>(state, observation, rng, uct_exploration_constant)
//     }
// }
//
// pub fn mcts_policy_merge_probs_generator<
//     const TNumberOfAssignments: usize,
//     const TNumberOfIterations: usize
// >(
//     uct_exploration_constant: f64
// ) -> impl Fn(&DoState, &DoObservation, &mut SmallRng) -> DoAction {
//     let uct_exploration_constant = uct_exploration_constant;
//
//     move |state, observation, rng| {
//         mcts_policy_merge_probs::<TNumberOfAssignments, TNumberOfIterations>(state, observation, rng, uct_exploration_constant)
//     }
// }
//
// fn mcts_policy_best_of_n<
//     const TNumberOfAssignments: usize,
//     const TNumberOfIterations: usize
// >(
//     state: &DoState,
//     observation: &DoObservation,
//     rng: &mut rand::rngs::SmallRng,
//
//     uct_exploration_constant: f64
// ) -> DoAction {
//     unsafe {
//         let mut move_map: HashMap<DoAction, usize> = std::collections::HashMap::new();
//
//         for i in 0..TNumberOfAssignments {
//             let state_with_assignment =
//                 sample_assignment_full(state, observation, rng);
//
//             let mut moves = monte_carlo_tree_search(
//                 DokoState::new(state_with_assignment, None),
//                 uct_exploration_constant,
//                 TNumberOfIterations,
//                 rng
//             );
//
//             moves
//                 .sort_by_key(|m| m.visits as i32);
//
//             moves
//                 .reverse();
//
//             for i in 0 .. moves.len() {
//                 let m = &moves[i];
//
//                 let count = move_map
//                     .entry(m.action)
//                     .or_insert(0);
//
//                 *count += i;
//             }
//         }
//
//         let mut min = std::usize::MAX;
//         let mut best_move = DoAction::CardSpadeNine;
//
//         for (_move, count) in &move_map {
//             if *count < min {
//                 min = *count;
//                 best_move = *_move;
//             }
//         }
//
//         println!("Move map: {:?}", move_map);
//         println!("Best move: {:?}", best_move);
//
//         return best_move;
//     }
// }
//
// fn mcts_policy_merge_probs<
//     const TNumberOfAssignments: usize,
//     const TNumberOfIterations: usize
// >(
//     state: &DoState,
//     observation: &DoObservation,
//     rng: &mut rand::rngs::SmallRng,
//     uct_exploration_constant: f64
// )
// -> DoAction {
//     unsafe {
//         let mut move_map: HashMap<DoAction, usize> = std::collections::HashMap::new();
//
//         for i in 0..TNumberOfAssignments {
//             let state_with_assignment =
//                 sample_assignment_full(state, observation, rng);
//
//             let moves = monte_carlo_tree_search(
//                 DokoState::new(state_with_assignment, None),
//                 uct_exploration_constant,
//                 TNumberOfIterations,
//                 rng
//             );
//
//             for m in moves {
//                 let count = move_map
//                     .entry(m.action)
//                     .or_insert(0);
//
//                 *count += m.visits;
//             }
//         }
//
//         let mut max = 0;
//         let mut best_move = DoAction::CardSpadeNine;
//
//         for (_move, count) in &move_map {
//             if *count > max {
//                 max = *count;
//                 best_move = *_move;
//             }
//         }
//
//         println!("Move map: {:?}", move_map);
//         println!("Best move: {:?}", best_move);
//
//         return best_move;
//     }
// }