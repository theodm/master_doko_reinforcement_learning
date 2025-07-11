use std::cmp::PartialEq;
use std::fmt::Debug;
use std::sync::Arc;
use bytemuck::Pod;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
use crate::alpha_zero::mcts::node::{value_for_player_full, AzNode};
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use crate::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use crate::alpha_zero::mcts::async_mcts::{mcts_search};
use crate::alpha_zero::train::glob::GlobalStats;
use crate::alpha_zero::utils::rot_arr::RotArr;
use crate::env::env_state::AzEnvState;

pub async fn self_play<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
>(
    create_root_state_fn: fn(&mut SmallRng) -> TGameState,

    batch_processor: &mut CachedNetworkBatchProcessorSender<
        TGameState,
        TActionType,
        TStateDim,
        TNumberOfPlayers,
        TNumberOfActions
    >,

    iterations: usize,
    alpha_zero_options: AlphaZeroTrainOptions,

    node_arena: &mut UnsafeArena<AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >>,
    state_arena: &mut UnsafeArena<
        TGameState
    >,

    current_epoch: usize,

    mut states_buffer: &mut std::vec::Vec<[i64; TStateDim]>,
    mut value_targets_buffer: &mut std::vec::Vec<[f32; TNumberOfPlayers]>,
    mut policy_targets_buffer: &mut std::vec::Vec<[f32; TNumberOfActions]>,
    glob_stats: Arc<GlobalStats>
) {
    let mut rng = SmallRng::from_os_rng();

    let mut current_state = create_root_state_fn(&mut rng);

    let mut current_player_vec = heapless::Vec::<usize, 250>::new();

    let value_target = if current_epoch == 0 {
        ValueTarget::Default
    } else {
        alpha_zero_options.value_target
    };

    // println!("value_target: {:?}", value_target);

    while !&current_state.is_terminal() {
        glob_stats.number_of_turns_in_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            node_arena.free_all();
            state_arena.free_all();

            if current_state.number_of_allowed_actions(current_epoch) == 1 && value_target == ValueTarget::Default {
                // Wenn wir nur eine mögliche Aktion haben, dann können wir uns
                // bei dem Standard-AlphaZero-Value-Target den MCTS sparen.

                let action = current_state.allowed_actions_by_action_index(false, current_epoch)[0];

                let mut state = [0; TStateDim];
                current_state.encode_into_memory(&mut state);

                let mut policy_target = [0.0f32; TNumberOfActions];
                policy_target[action] = 1.0;

                if rng.gen::<f32>() < alpha_zero_options.probability_of_keeping_experience {
                    states_buffer.push(state);
                    // value_targets_buffer.push([0.0; TNumberOfPlayers]);
                    policy_targets_buffer.push(policy_target);

                    current_player_vec.push(current_state.current_player())
                        .unwrap();

                    glob_stats.number_of_experiences_added_in_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                current_state = current_state.take_action_by_action_index(action, false, current_epoch);

                continue;
            }

            let mcts_result = mcts_search(
                // ToDo: Kann auch übernommen werden?
                current_state.clone(),
                batch_processor,
                iterations,
                &alpha_zero_options,
                &mut rng,
                node_arena,
                state_arena,
                current_epoch
            )
                .await;

            let (action_to_take, root_node) = mcts_result;

            let mut policy_target = [0.0f32; TNumberOfActions];
            let mut visit_sum = 0.0f32;

            for (index, child) in root_node
                .children
                .iter()
                .enumerate() {
                let child = *child;

                if !child.is_null() {
                    policy_target[index] = (*child).visits as f32;
                    visit_sum += (*child).visits as f32;
                }
            }

            policy_target = policy_target
                .iter()
                .map(|x| x / visit_sum)
                .collect::<Vec<f32>>()
                .as_slice()
                .try_into()
                .unwrap();

            // println!("policy_target: {:?}", policy_target);

            let mut state = [0; TStateDim];

            (*root_node
                .state)
                .clone()
                .encode_into_memory(&mut state);


            if (value_target == ValueTarget::Greedy) {
                let greedy_value_target = value_for_player_full(
                    root_node
                        .select_greedy_node(),
                    root_node
                        .current_player,
                );

                value_targets_buffer.push(greedy_value_target);
            }

            if (value_target == ValueTarget::Avg) {
                let avg_value_target = value_for_player_full(
                    &root_node,
                    root_node
                        .current_player,
                );

                value_targets_buffer.push(avg_value_target);
            }

            //
            // // predict
            // let (v, r) = batch_processor.process(state).await;
            //
            // if rng.gen_bool(0.001) {
            //     println!("value_target: {:?} value_2_target: {:?} maybe_value_target: {:?} predicted: {:?}", value_target, value_2_target, maybe_value_target, v);
            // }

            states_buffer.push(state);
            policy_targets_buffer.push(policy_target);

            glob_stats.number_of_experiences_added_in_epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            current_player_vec.push(current_state.current_player())
                .unwrap();

            // Die Aktion ausführen.
            current_state = current_state.take_action_by_action_index(action_to_take, false, current_epoch);
        }
    }

    if value_target == ValueTarget::Default {
        let real_rewards = current_state
            .rewards_or_none()
            .unwrap();

        for (i, cp) in current_player_vec
            .iter()
            .enumerate() {
            let mut value_target = RotArr::new_from_0(*cp, real_rewards);

            value_targets_buffer.push(value_target.extract());
        }
    }
}
