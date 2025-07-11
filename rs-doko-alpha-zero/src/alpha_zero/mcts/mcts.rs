use std::fmt::Debug;
use rand::distr::Distribution;
use rand::prelude::IndexedRandom;
use rand::rngs::SmallRng;
use rand::Rng;
use rand_distr::num_traits::Zero;
use rand_distr::Dirichlet;
use rs_doko::player::player::DoPlayer;
use std::mem::{ManuallyDrop, transmute};
use std::ops::{Add, Div, Mul};
use std::ptr::null_mut;
use bytemuck::Pod;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::alpha_zero::mcts::node::{AzNode, new_with_prior_prob, value_for_player, value_from_pov};
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::utils::rot_arr::RotArr;
use crate::env::env_state::AzEnvState;



pub unsafe fn expand_node<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,

    node_arena: &mut UnsafeArena<AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >>,

    state_arena: &mut UnsafeArena<TGameState>,

    num_iteration: usize,
    epoch: usize,
) {
    let state = (*node)
        .state;

    for action_index in (*state).allowed_actions_by_action_index(true, epoch) {
        let mut child_state = (*state).take_action_by_action_index(action_index, true, epoch);

        let new_node = new_with_prior_prob(
            child_state,
            0.0,
            node as *const AzNode<
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
            state_arena
        );

        // println!("Action: {:?}", action_index);

        (*node).children[action_index] = node_arena.alloc(new_node);
        (*node).has_children = true;
    }
}

pub unsafe fn tree_to_string<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    depth: usize,
    max_depth: usize,
    alpha_zero_options: &AlphaZeroTrainOptions
) -> String {
    let mut output = String::new();

    let uct = if (*node).parent.is_null() {
        0.0
    } else {
        puct_for_child((*node).parent, node, alpha_zero_options.puct_exploration_constant, alpha_zero_options.min_or_max_value)
    };

    output.push_str(&"    ".repeat(depth));

    output.push_str(&format!(
        "Action: {} Value: {:?} UCT: {:?} Visits: {:?} Win Score: {:?} Prior Prob: {:?} Player: {:?} Virtual Loss: {:?} Virtual Visits: {:?} Win Score scale: {:?} Value Scale {:?} node_ptr {:?} \n",
        (*node).last_action.map(|x| format!("{:?}", x)).unwrap_or("None".to_string()),
        value_from_pov(node),
        uct,
        (*node).visits,
        (*node).win_score,
        (*node).prior_prob,
        (*node).current_player,
        (*node).virtual_loss,
        (*node).virtual_visits,
        (*node).win_score.clone() * 8f32,
        value_from_pov(node).map(|x| x * 8f32),
        node
    ));

    if depth < max_depth {
        for child in (*node).children {
            if !child.is_null() {
                output.push_str(&tree_to_string(child, depth + 1, max_depth, alpha_zero_options));
            }
        }
    }

    output
}


pub unsafe fn print_tree<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    depth: usize,
    max_depth: usize,

    alpha_zero_options: &AlphaZeroTrainOptions
) {
    let uct = if (*node).parent.is_null() {
        0.0
    } else {
        puct_for_child((*node).parent, node, alpha_zero_options.puct_exploration_constant, alpha_zero_options.min_or_max_value)
    };

    for i in 0..depth {
        print!("    ");
    }

    print!(
        "Action: {} Value: {:?} UCT: {:?} Visits: {:?} Win Score: {:?} Prior Prob: {:?} Player: {:?} Virtual Loss: {:?} Virtual Visits: {:?} node_ptr {:?} \n",
        0,
        value_from_pov(node),
        uct,
        (*node).visits,
        (*node).win_score,
        (*node).prior_prob,
        (*node).current_player,
        (*node).virtual_loss,
        (*node).virtual_visits,
        node
    );

    if depth < max_depth {
        for child in (*node).children {
            if child.is_null() {
                continue;
            }

            print_tree(child, depth + 1, max_depth, alpha_zero_options);
        }
    }
}

pub unsafe fn select_action<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    root_node: &AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    temperature: Option<f32>,
    rng: &mut SmallRng,
) -> usize {
    // println!("Selecting action with temperature: {:?}", temperature);
    // println!("Root node is: {}", root_node.state.as_ref().unwrap());

    let mut action_counts = heapless::Vec::<(usize, usize), { TNumberOfActions }>::new();

    // println!("children:");
    for (index, child) in root_node.children.iter().enumerate() {
        let child = *child;

        if child.is_null() {
            continue;
        }

        action_counts.push((index, (*child).visits)).unwrap();
    }

    match temperature {
        None => {
            // println!("No temperature");

            return action_counts.iter().max_by_key(|x| x.1).unwrap().0;
        }
        Some(temperature) => {
            return if (temperature.is_infinite()) {
                // println!("Temperature is infinite");
                action_counts.choose(rng).unwrap().0
            } else {
                let distribution: heapless::Vec<f32, TNumberOfActions> = action_counts
                    .iter()
                    .map(|(index, count)| (*count as f32).powf(1.0 / temperature))
                    .collect();

                // println!("action counts: {:?}", action_counts);
                // println!("distribution: {:?}", distribution);

                let random_index = rand_distr::weighted::WeightedIndex::new(distribution)
                    .unwrap()
                    .sample(rng);

                let result = action_counts[random_index].0;

                // println!("Random index: {}", random_index);
                // println!("result: {}", result);

                return result;
            };
        }
    }
}

pub(crate) unsafe fn backpropagate<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    promising_node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    rewards: RotArr<f32, TNumberOfPlayers>,
) {
    let mut temp_node = promising_node;

    loop {
        (*temp_node).visits += 1;
        (*temp_node).win_score.add_other_in_place(&rewards);
        (*temp_node).virtual_loss = RotArr::zeros((*temp_node).current_player);
        (*temp_node).virtual_visits = 0;

        let parent_node = (*temp_node).parent;

        if parent_node.is_null() {
            break;
        }

        temp_node = parent_node;
    }
}

pub unsafe fn has_children<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
) -> bool {
    for child in (*node).children {
        if child.is_null() {
            continue;
        }

        return true;
    }

    return false;
}

pub(crate) unsafe fn select_promising_node<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    uct_exploration_constant: f32,
    min_or_max_value: f32,
) -> *mut AzNode<
    TGameState,
    TActionType,
    { TNumberOfPlayers },
    { TNumberOfActions },
> {
    let mut node = node;

    while has_children(node) {
        node = find_best_child(
            node,
            uct_exploration_constant,
            min_or_max_value
        );

        if node.is_null() {
            break;
        }
    }

    return node;
}

unsafe fn find_best_child<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    >,
    uct_exploration_constant: f32,
    min_or_max_value: f32,
) -> *mut AzNode<
    TGameState,
    TActionType,
    { TNumberOfPlayers },
    { TNumberOfActions },
> {
    let mut best_score = f32::MIN;
    let mut best_child = null_mut();

    for child in (*node).children {
        if child.is_null() {
            continue;
        }

        let score = puct_for_child(node, child, uct_exploration_constant, min_or_max_value);

        if score > best_score {
            best_score = score;
            best_child = child;
        }
    }

    return best_child;
}

pub unsafe fn puct_for_child<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    parent: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    child: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    uct_exploration_constant: f32,

    min_or_max_value: f32,
) -> f32 {
    let current_player = (*parent).current_player;
    let child_player = (*child).current_player;

    //  ToDo in methode integrieren
    let value = value_for_player(child, current_player);

    let exploitation_term = value;

    let child_visits = (*child).visits as f32 + (*child).virtual_visits as f32;
    let parent_visits = (*parent).visits as f32 + (*parent).virtual_visits as f32;

    let exploration_term =
        (*child).prior_prob * (parent_visits).sqrt() / (1.0f32 + child_visits);

    return exploitation_term + uct_exploration_constant * exploration_term;
}

fn get_relative<const TNumberOfPlayers: usize>(current: usize, target: usize) -> usize {
    (target + TNumberOfPlayers - current) % TNumberOfPlayers
}
fn apply<const TNumberOfPlayers: usize>(
    node_win_score: &mut [f32; TNumberOfPlayers],
    rewards_from_rewards_player_pov: [f32; TNumberOfPlayers],
    rewards_player: usize,
    node_player: usize,
) {
    for i in 0..TNumberOfPlayers {
        let ri = get_relative::<TNumberOfPlayers>(rewards_player, i);
        let wsi = get_relative::<TNumberOfPlayers>(node_player, i);

        node_win_score[wsi] += rewards_from_rewards_player_pov[ri];
    }
}

fn get_index<const N: usize>(points_of: usize, array_of: usize) -> usize {
    let mut a0_i = get_relative::<N>(array_of, 0);

    return (a0_i + points_of) % N;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_with_result(
        initial: [f32; 4],
        rewards_from_rewards_player_pov: [f32; 4],
        rewards_player: usize,
        node_player: usize,
    ) -> [f32; 4] {
        let mut node_win_score = initial;

        apply(
            &mut node_win_score,
            rewards_from_rewards_player_pov,
            rewards_player,
            node_player,
        );

        return node_win_score;
    }

    #[test]
    fn test_get_relative() {
        let value = [2, 3, 0, 1];

        assert_eq!(value[get_index::<4>(0, 2)], 0);
        assert_eq!(value[get_index::<4>(1, 2)], 1);
        assert_eq!(value[get_index::<4>(2, 2)], 2);
        assert_eq!(value[get_index::<4>(3, 2)], 3);
    }

    #[test]
    fn test_apply() {
        let r = apply_with_result([10.0, 11.0, 12.0, 13.0], [2.0, 3.0, 0.0, 1.0], 2, 0);
        assert_eq!(r, [10.0, 12.0, 14.0, 16.0]);

        let r = apply_with_result([11.0, 12.0, 13.0, 10.0], [2.0, 3.0, 0.0, 1.0], 2, 1);
        assert_eq!(r, [12.0, 14.0, 16.0, 10.0]);
    }
}
