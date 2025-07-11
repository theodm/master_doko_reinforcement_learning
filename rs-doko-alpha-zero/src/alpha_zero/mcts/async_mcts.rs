use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use crate::alpha_zero::mcts::mcts::{backpropagate, expand_node, print_tree, select_action, select_promising_node, tree_to_string};
use crate::alpha_zero::mcts::node::{new_root_node, new_with_prior_prob, AzNode};
use crate::alpha_zero::utils::float_array_utils::to_vec_with_indices_removed;
use crate::alpha_zero::utils::float_vec_utils::{softmax_inplace, vec_with_non_indices_removed};
use crate::alpha_zero::utils::rot_arr::RotArr;
use crate::env::env_state::AzEnvState;
use rand::distr::Distribution;
use rand::prelude::{IteratorRandom, SmallRng};
use rand_distr::num_traits::Zero;
use rand_distr::Dirichlet;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use std::fmt::Debug;
use std::fs;
use std::fs::File;
use std::ptr::null_mut;
use bytemuck::Pod;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use tokio::task::JoinSet;
use chrono::Local;
use std::io::Write;
use rs_full_doko::display::display::display_game;

pub fn save_log_maybe(
    log: Option<String>,
    folder: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(text) = log {
        fs::create_dir_all(folder)?;

        let now = Local::now();
        let timestamp = now.format("%d.%m.%Y_%H-%M-%S").to_string();

        let filename = format!("{}/log_{}.txt", folder, timestamp);

        let mut file = File::create(&filename)?;
        file.write_all(text.as_bytes())?;

        let mut entries: Vec<_> = fs::read_dir(folder)?
            .filter_map(|res| res.ok()) 
            .filter_map(|entry| {
                let path = entry.path();
                // Nur Dateien berücksichtigen
                match fs::metadata(&path) {
                    Ok(metadata) if metadata.is_file() => Some(path),
                    _ => None,
                }
            })
            .collect();

        entries.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        while entries.len() > 50 {
            if let Some(oldest) = entries.first() {
                let _ = fs::remove_file(oldest);
            }
            entries.remove(0);
        }
    }

    Ok(())
}

pub async unsafe fn expand_root_node<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    root_node: &mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    batch_processor: &CachedNetworkBatchProcessorSender<
        TGameState,
        TActionType,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    rng: &mut SmallRng,
    node_arena: &mut UnsafeArena<
        AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
    >,
    state_arena: &mut UnsafeArena<TGameState>,
    epoch: usize,
) {
    let state = (*(root_node.state)).clone();

    let allowed_actions = state.allowed_actions_by_action_index(false, epoch);

    let (initial_value, initial_policy_logits) = batch_processor.process(state.clone()).await;

    let mut policy_logits: heapless::Vec<f32, TNumberOfActions> =
        heapless::Vec::from_slice(&initial_policy_logits).unwrap();

    softmax_inplace(&mut policy_logits);

    let mut noise = Dirichlet::new([dirichlet_alpha; TNumberOfActions])
        .unwrap()
        .sample(rng);

    fn mul_inplace<const N: usize>(a: f32, b: &mut heapless::Vec<f32, N>) {
        for i in 0..b.len() {
            b[i] *= a;
        }
    }

    // ((1.0 - dirichlet_epsilon) * policy_logits)
    mul_inplace(1.0 - dirichlet_epsilon, &mut policy_logits);

    fn mul_inplace2<const N: usize>(a: f32, b: &mut [f32; N]) {
        for i in 0..b.len() {
            b[i] *= a;
        }
    }

    // dirichlet_epsilon * noise_t;
    mul_inplace2(dirichlet_epsilon, &mut noise);

    fn add_inplace<const N: usize>(a: &mut heapless::Vec<f32, N>, b: &[f32; N]) {
        for i in 0..a.len() {
            a[i] += b[i];
        }
    }

    add_inplace(&mut policy_logits, &noise);
    let action_probs = policy_logits;

    let mut action_probs = vec_with_non_indices_removed(&action_probs, &allowed_actions);

    let x = action_probs.iter().sum::<f32>();

    fn div_inplace<const N: usize>(a: &mut heapless::Vec<f32, N>, b: f32) {
        for i in 0..a.len() {
            a[i] /= b;
        }
    }

    div_inplace(&mut action_probs, x);

    for (i, allowed_action) in allowed_actions.iter().enumerate() {
        let action = allowed_action;

        let action_prob = action_probs[i];

        let new_node = new_with_prior_prob(
            state.take_action_by_action_index(*action, true, epoch),
            action_prob,
            root_node
                as *const AzNode<
                    TGameState,
                    TActionType,
                    TNumberOfPlayers,
                    TNumberOfActions,
                >
                as *mut AzNode<
                    TGameState,
                    TActionType,
                    TNumberOfPlayers,
                    TNumberOfActions,
                >,
            state_arena,
        );

        let new_node_ptr = node_arena.alloc(new_node);

        root_node.children[*action] = new_node_ptr;
        root_node.has_children = true;
    }


    root_node.visits = 1;

    root_node.win_score = RotArr::new_from_pov(state.current_player(), initial_value);

    return;
}

struct PointerWrapper<T> {
    ptr: *mut T,
}

impl<T> Debug for PointerWrapper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointerWrapper")
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl<T> Clone for PointerWrapper<T> {
    fn clone(&self) -> Self {
        PointerWrapper { ptr: self.ptr }
    }
}

// ToDo: Erklären warum benötigt!
impl<T> PointerWrapper<T> {
    pub fn new(ptr: *mut T) -> PointerWrapper<T> {
        PointerWrapper { ptr }
    }

    pub fn get(&self) -> *mut T {
        self.ptr
    }
}

unsafe impl<T> Sync for PointerWrapper<T> {}
unsafe impl<T> Send for PointerWrapper<T> {}

pub async unsafe fn mcts_search<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    state: TGameState,
    batch_processor: &CachedNetworkBatchProcessorSender<
        TGameState,
        TActionType,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    iterations: usize,

    // MCTSOptions, aber Iterationen werden nicht übergeben
    mcts_options: &AlphaZeroTrainOptions,
    rng: &mut SmallRng,
    node_arena: &mut UnsafeArena<
        AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
    >,
    state_arena: &mut UnsafeArena<TGameState>,
    epoch: usize,
) -> (
    usize,
    AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
) {
    let state = state.clone();
    let state2 = state.clone();
    let mut root_node = new_root_node(state, state_arena);

    expand_root_node::<
        TGameState,
        TActionType,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions,
    >(
        &mut root_node,
        batch_processor,
        mcts_options.dirichlet_alpha,
        mcts_options.dirichlet_epsilon,
        rng,
        node_arena,
        state_arena,
        epoch
    )
    .await;

    for i in 0..iterations {
        let mut promising_node = select_promising_node(
            &mut root_node,
            mcts_options.puct_exploration_constant,
            mcts_options.min_or_max_value,
        );

        let promising_node = PointerWrapper::new(
            promising_node
        );

        if !(*promising_node.get()).is_terminal {
            expand_node(promising_node.get(), node_arena, state_arena, i, epoch);
        }

        let rewards_rot = if (*promising_node.get()).is_terminal {
            let promising_state = (*promising_node.get()).state.as_ref().unwrap();

            // Leaf Node
            let rewards = promising_state.rewards_or_none().unwrap();

            RotArr::new_from_0(promising_state.current_player(), rewards)
        } else {
            // Calc value with NN
            let mut promising_state_encoded: [i64; TStateDim] =
                [0; TStateDim];

            let promising_state = (*promising_node.get()).state.as_ref().unwrap();

            promising_state.encode_into_memory(&mut promising_state_encoded);

            let allowed_actions = promising_state.allowed_actions_by_action_index(true, epoch);

            let (value, policy_logits) = batch_processor.process(promising_state.clone()).await;

            let mut action_probs = to_vec_with_indices_removed(&policy_logits, &allowed_actions);

            softmax_inplace(&mut action_probs);

            let mut i = 0;

            for child in (*promising_node.get()).children.iter_mut() {
                let child = *child;

                if child.is_null() {
                    continue;
                }

                let action_prob = action_probs[i];

                (*child).prior_prob = action_prob;
                i += 1;
            }

            // Von Nicht-Blättern brauchen wir den Zustand nicht mehr und
            // können den entsprechenden Speicherplatz freigeben.
            state_arena.free_single((*promising_node.get()).state_index_in_arena);
            (*promising_node.get()).state = null_mut();

            let rewards = value.clone();

            RotArr::new_from_pov((*promising_node.get()).current_player, rewards)
        };

        backpropagate(promising_node.get(), rewards_rot);
    }

    if rng.random_bool(1f64 / 5000f64) {
        let game = state2.display_game();
        let tree_str = tree_to_string(
            &mut root_node,
            0,
            2,
            mcts_options
        );

        // println!("logged");

        save_log_maybe(
            Some(format!("{} {}\n{}\n\n{}", iterations, mcts_options.temperature.unwrap_or(0.0),
                         game, tree_str)),
            "mcts_tree_logs"
        ).unwrap();
    }

    // println!("===");
    // root_node.print_graphviz(2, mcts_options.puct_exploration_constant);
    // println!("===");

    return (
        select_action(&root_node, mcts_options.temperature, rng),
        root_node,
    );
}

// struct MCTS<
//     TGameState: AzEnvState<TActionType, TStateMemoryDataType, TNumberOfPlayers, TNumberOfActions>,
//     TActionType: Copy + Send + Debug,
//     TStateMemoryDataType: Zero + Pod + Clone + Copy + tch::kind::Element + Serialize + DeserializeOwned  + Debug + Send,
//     const TStateDim: usize,
//     const TNumberOfPlayers: usize,
//     const TNumberOfActions: usize,
// > {}

struct Part1Result {

}

// impl<
//         TGameState: AzEnvState<TActionType, TStateMemoryDataType, TNumberOfPlayers, TNumberOfActions>,
//         TActionType: Copy + Send + Debug,
//         TStateMemoryDataType: Zero + Pod + Clone + Copy + tch::kind::Element + Serialize + DeserializeOwned  + Debug + Send,
//         const TStateDim: usize,
//         const TNumberOfPlayers: usize,
//         const TNumberOfActions: usize,
//     >
//     MCTS<
//         TGameState,
//         TActionType,
//         TStateMemoryDataType,
//         TStateDim,
//         TNumberOfPlayers,
//         TNumberOfActions,
//     >
// {
// }


// pub async unsafe fn mcts_search_parallel<
//     TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
//     TActionType: Copy + Send + Debug + 'static,
//     const TStateDim: usize,
//     const TNumberOfPlayers: usize,
//     const TNumberOfActions: usize,
// >(
//     state: TGameState,
//     batch_processor: &CachedNetworkBatchProcessorSender<
//         TGameState,
//         TActionType,
//         TStateDim,
//         TNumberOfPlayers,
//         TNumberOfActions,
//     >,
//     iterations: usize,
//
//
//     // MCTSOptions, aber Iterationen werden nicht übergeben
//     mcts_options: &AlphaZeroTrainOptions,
//     rng: &mut SmallRng,
//     node_arena: &mut UnsafeArena<
//         AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
//     >,
//     state_arena: &mut UnsafeArena<TGameState>,
//
//     epoch: usize
// ) -> (
//     usize,
//     AzNode<
//         TGameState,
//         TActionType,
//         { TNumberOfPlayers },
//         { TNumberOfActions },
//     >,
// ) {
//     let state = state.clone();
//     let mut root_node = new_root_node(state.clone(), state_arena);
//
//     expand_root_node::<
//         TGameState,
//         TActionType,
//         TStateDim,
//         TNumberOfPlayers,
//         TNumberOfActions,
//     >(
//         &mut root_node,
//         batch_processor,
//         mcts_options.dirichlet_alpha,
//         mcts_options.dirichlet_epsilon,
//         rng,
//         node_arena,
//         state_arena,
//         epoch
//     )
//     .await;
//
//     let mut processed_iterations = 0;
//
//     while processed_iterations < iterations {
//         let mut promising_nodes = heapless::Vec::<
//             PointerWrapper<
//                 AzNode<
//                     TGameState,
//                     TActionType,
//                     TNumberOfPlayers,
//                     TNumberOfActions,
//                 >,
//             >,
//             64,
//         >::new();
//
//         // let is_random = rng.random_ratio(1, 50000);
//
//         for i in 0..10 {
//             let mut promising_node = select_promising_node(
//                 &mut root_node,
//                 mcts_options.puct_exploration_constant,
//                 mcts_options.min_or_max_value,
//             );
//
//             let promising_node = promising_node
//                 as *const AzNode<
//                     TGameState,
//                     TActionType,
//                     TNumberOfPlayers,
//                     TNumberOfActions,
//                 >
//                 as *mut AzNode<
//                     TGameState,
//                     TActionType,
//                     TNumberOfPlayers,
//                     TNumberOfActions,
//                 >;
//
//             if promising_node.is_null() {
//                 break;
//             }
//
//             // set virtual loss (to parents)
//             let mut node = promising_node;
//
//             while !node.is_null() {
//                 let win_score = (*node).win_score.clone();
//
//                 (*node).virtual_loss.add_other_in_place(
//                     &(win_score / (*node).visits as f32),
//                 );
//                 (*node).virtual_visits += 1;
//
//                 node = (*node).parent;
//             }
//
//             if (*promising_node).virtual_visits > 1 {
//                 continue;
//             }
//             // if is_random {
//             //     print_tree(
//             //         &mut root_node,
//             //         0,
//             //         5,
//             //         mcts_options
//             //     );
//             // }
//
//             let promising_node = PointerWrapper::new(promising_node);
//
//             promising_nodes.push(promising_node).unwrap();
//         }
//
//         // if (processed_iterations > 10) {
//         //     print_tree(
//         //         &mut root_node,
//         //         0,
//         //         1,
//         //         mcts_options
//         //     );
//         //
//         //     std::process::exit(1);
//         // }
//         //
//         // if (is_random) {
//         //     println!("promising_nodes.len(): {}", promising_nodes.len());
//         //
//         //     if (promising_nodes.len() > 1) {
//         //         println!("promising_nodes[0]: {:?}", promising_nodes[0]);
//         //         println!("promising_nodes[1]: {:?}", promising_nodes[1]);
//         //
//         //         std::process::exit(1);
//         //     }
//         // }
//
//         for (i, promising_node) in promising_nodes.iter().enumerate() {
//             if !(*promising_node.get()).is_terminal {
//                 expand_node(promising_node.get(), node_arena, state_arena, i, epoch);
//             }
//         }
//
//         let mut joinSet = JoinSet::new();
//
//         for (i, promising_node) in promising_nodes.iter().enumerate() {
//             if (*promising_node.get()).is_terminal {
//                 joinSet.spawn(tokio::spawn(async move {
//                     return ([0.0f32; TNumberOfPlayers], [0.0f32; TNumberOfActions])
//                 }));
//             } else {
//                 let state = (*promising_node.get()).state.as_ref().unwrap().clone();
//                 let mut batch_processor = batch_processor.clone();
//
//                 joinSet.spawn(tokio::spawn(async move {
//                     let (value, policy_logits) = batch_processor.process(
//                         state,
//                     ).await;
//
//                     return (value, policy_logits);
//                 }));
//             }
//         }
//         let results = joinSet.join_all().await;
//
//         for (i, result) in results.into_iter().enumerate() {
//             let result = result.unwrap();
//
//             let promising_node = promising_nodes[i].get();
//
//             if !(*promising_node).is_terminal {
//                 let allowed_actions = (*promising_node)
//                     .state
//                     .as_ref()
//                     .unwrap()
//                     .allowed_actions_by_action_index(true, epoch);
//
//                 let mut action_probs = to_vec_with_indices_removed(&result.1, &allowed_actions);
//
//                 softmax_inplace(&mut action_probs);
//
//                 let mut i = 0;
//
//                 for child in (*promising_node).children.iter_mut() {
//                     let child = *child;
//
//                     if child.is_null() {
//                         continue;
//                     }
//
//                     let action_prob = action_probs[i];
//
//                     (*child).prior_prob = action_prob;
//                     i += 1;
//                 }
//
//                 // Von Nicht-Blättern brauchen wir den Zustand nicht mehr und
//                 // können den entsprechenden Speicherplatz freigeben.
//                 state_arena.free_single((*promising_node).state_index_in_arena);
//                 (*promising_node).state = null_mut();
//
//                 let rewards = result.0.clone();
//
//                 backpropagate(promising_node, RotArr::new_from_pov((*promising_node).current_player, rewards));
//             } else{
//                 let promising_state = (*promising_node).state.as_ref().unwrap();
//
//                 // Leaf Node
//                 let rewards = promising_state.rewards_or_none().unwrap();
//
//                 RotArr::new_from_0(promising_state.current_player(), rewards);
//
//                 backpropagate(promising_node, RotArr::new_from_0(promising_state.current_player(), rewards));
//             }
//         }
//
//         processed_iterations += promising_nodes.len();
//     }
//
//     if rng.random_bool(1f64 / 5000f64) {
//         let game = state.clone().display_game();
//         let tree_str = tree_to_string(
//             &mut root_node,
//             0,
//             2,
//             mcts_options
//         );
//
//         // println!("logged");
//
//         save_log_maybe(
//             Some(format!("{} {}\n{}\n\n{}", iterations, mcts_options.temperature.unwrap_or(0.0),
//                          game, tree_str)),
//             "mcts_tree_logs"
//         ).unwrap();
//     }
//
//     // println!("processed_iterations: {}", processed_iterations);
//     // if (rng.random_ratio(1, 10000) && mcts_options.par_iterations > 1) {
//     //     print_tree(
//     //         &mut root_node,
//     //         0,
//     //         5,
//     //         mcts_options
//     //     );
//     //     std::process::exit(1);
//     // }
//
//     return (
//         select_action(&root_node, mcts_options.temperature, rng),
//         root_node,
//     );
// }
