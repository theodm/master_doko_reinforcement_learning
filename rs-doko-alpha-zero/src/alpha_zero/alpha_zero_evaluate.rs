use std::fmt::Debug;
use bytemuck::Pod;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use crate::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use crate::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
use crate::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use crate::alpha_zero::mcts::async_mcts::{mcts_search};
use crate::alpha_zero::mcts::node::{value_for_player, AzNode};
use crate::env::env_state::AzEnvState;


pub fn apply_mask_inplace<const N: usize>(
    data: &mut [f32; N],
    mask: &[bool; N],
) {
    for i in 0..data.len() {
        if !mask[i] {
            data[i] = f32::MIN;
        }
    }
}

/// Führt den AlphaZero-MCTS (nur Policy-Head) für einen einzelnen Zug aus. Dabei muss ein
/// CachedNetworkBatchProcessorSender und Arenen für die Nodes und States
/// übergeben werden. (Kann im Rahmen der Evaluation verwendet werden.)
pub async fn evaluate_single_only_policy_head<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    state: &TGameState,

    batch_processor: &CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions>,

) -> (usize, [usize; TNumberOfActions]) {
    let (value, mut policy) = batch_processor
        .process(state.clone())
        .await;

    let mut allowed_actions = state.allowed_actions_by_action_index(
        false,
        1000
    );
    let mut mask = [false; TNumberOfActions];
    for i in 0..TNumberOfActions {
        mask[i] = allowed_actions.contains(&i);
    }

    apply_mask_inplace(&mut policy, &mask);

    let next_action_logits = policy
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect::<Vec<_>>();

    let next_action = next_action_logits
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Finde das Minimum, um alle Werte positiv zu machen
    let min_logit = policy.iter().cloned().fold(f32::INFINITY, f32::min);
    let offset = if min_logit < 0.0 { -min_logit } else { 0.0 };

    let mut moves_to_visits = [0; TNumberOfActions];

    for (i, &logit) in policy.iter().enumerate() {
        moves_to_visits[i] = ((logit + offset) * 1000.0) as usize;
    }

    (*next_action, moves_to_visits)
}

/// Führt den AlphaZero-MCTS für einen einzelnen Zug aus. Dabei muss ein
/// CachedNetworkBatchProcessorSender und Arenen für die Nodes und States
/// übergeben werden. (Kann im Rahmen der Evaluation verwendet werden.)
pub async fn evaluate_single<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    current_state: &TGameState,

    alpha_zero_evaluation_options: AlphaZeroEvaluationOptions,

    batch_processor: &CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions>,

    node_arena: &mut UnsafeArena<AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>>,
    state_arena: &mut UnsafeArena<TGameState>,

    rng: &mut rand::rngs::SmallRng,
) -> (usize, [usize; TNumberOfActions], [f32; TNumberOfActions]) {
    // if current_state.number_of_allowed_actions(1000) == 1 {
    //
    //     // Wenn wir nur eine mögliche Aktion haben, dann kann man sich
    //     // den MCTS-Algorithmus sparen.
    //     return current_state.allowed_actions_by_action_index(false, 1000)[0];
    // }

    unsafe {
        node_arena.free_all();
        state_arena.free_all();

        let mut state = [0; TStateDim];
        current_state.encode_into_memory(&mut state);

        let mcts_result = mcts_search(
            // ToDo: Kann auch übernommen werden?
            current_state.clone(),
            batch_processor,
            alpha_zero_evaluation_options.iterations,
            &AlphaZeroTrainOptions {
                dirichlet_alpha: alpha_zero_evaluation_options.dirichlet_alpha,
                dirichlet_epsilon: 0f32,

                puct_exploration_constant: alpha_zero_evaluation_options.puct_exploration_constant,

                temperature: None,

                min_or_max_value: alpha_zero_evaluation_options.min_or_max_value,


                // unused here
                batch_nn_max_delay: Default::default(),
                batch_nn_buffer_size: 0,
                checkpoint_every_n_epochs: 0,
                probability_of_keeping_experience: 0.0,
                games_per_epoch: 0,
                max_concurrent_games: 0,
                number_of_batch_receivers: 0,
                value_target: ValueTarget::Default,
                node_arena_capacity: 0,
                state_arena_capacity: 0,
                cache_size_batch_processor: 0,
                mcts_workers: 0,
                max_concurrent_games_in_evaluation: 0,
                epoch_start: None,
                evaluation_every_n_epochs: 0,
                skip_evaluation: false,
                mcts_iterations: alpha_zero_evaluation_options.iterations,
            },
            rng,
            node_arena,
            state_arena,
            1000
        )
            .await;

        let (action_to_take, root_node) = mcts_result;

        let mut moves_to_visits = [0; TNumberOfActions];
        let mut moves_to_values = [0.0; TNumberOfActions];

        for (i, child) in root_node.children.iter().enumerate() {
            if child.is_null() {
                continue;
            }

            let visits = (*(*child)).visits;

            moves_to_visits[i] = visits;
            moves_to_values[i] = value_for_player(
                *child,
                current_state.current_player()
            );
        }

        // Die Aktion zurückgeben.
        (action_to_take, moves_to_visits, moves_to_values)
    }
}