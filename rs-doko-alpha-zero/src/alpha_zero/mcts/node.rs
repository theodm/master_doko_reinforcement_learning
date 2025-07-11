use std::fmt::{Debug, format};
use std::ptr::null_mut;
use std::usize;
use bytemuck::Pod;
use graphviz_rust::attributes::{id, label};
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::dot_structures::Graph;
use graphviz_rust::printer::PrinterContext;
use rand_distr::num_traits::Zero;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use graphviz_rust::printer::DotPrinter;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::alpha_zero::mcts::mcts::puct_for_child;
use crate::alpha_zero::utils::pretty_print_utils::pretty_print_values_array;
use crate::alpha_zero::utils::rot_arr::RotArr;
use crate::env::env_state::AzEnvState;

// Es sollte eigentlich kein Unsafe notwendig sein, na gut...
unsafe impl<
        TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
        TActionType: Copy + Send + Debug,
        const TNumberOfPlayers: usize,
        const TNumberOfActions: usize,
    > Send
    for AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>
{
}

pub struct AzNode<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    pub state_index_in_arena: usize,
    pub state: *mut TGameState,

    pub is_terminal: bool,
    pub current_player: usize,
    pub last_action: Option<TActionType>,

    pub prior_prob: f32,
    pub visits: usize,
    pub win_score: RotArr<f32, TNumberOfPlayers>,

    pub children: [*mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >; TNumberOfActions],
    pub has_children: bool,

    pub parent: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,

    pub virtual_loss: RotArr<f32, TNumberOfPlayers>,
    pub virtual_visits: usize
}


impl <
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> AzNode<
    TGameState,
    TActionType,
    TNumberOfPlayers,
    TNumberOfActions,
> {

    // pub unsafe fn print_graphviz(
    //     &self,
    //
    //     max_depth: usize,
    //
    //     puct_exploration_constant: f32,
    //     min_or_max_value: f32,
    // ) -> Graph {
    //     let mut g = graph!(strict di id!("mcts_tree"));
    //
    //     self.internal_print_graphviz(
    //         0,
    //         max_depth,
    //         &mut g,
    //         None,
    //         puct_exploration_constant,
    //         min_or_max_value
    //     );
    //
    //     // print
    //     println!("{}", g.print(&mut PrinterContext::default()));
    //
    //     return g
    // }
    // unsafe fn internal_print_graphviz(
    //     &self,
    //
    //     depth: usize,
    //     max_depth: usize,
    //
    //     graph: &mut Graph,
    //     // Einduetiger Bezeichner des Elternknotens (falls vorhanden)
    //     parent_id: Option<NodeId>,
    //
    //     puct_exploration_constant: f32,
    //     min_or_max_value: f32,
    // ) {
    //     // Eindeutiger Bezeichner des aktuellen Knotens
    //     let node_id_str = format!("node_{:p}", self);
    //
    //     match parent_id {
    //         Some(parent_id) => {
    //             let puct =
    //                 puct_for_child(self.parent, self, puct_exploration_constant, min_or_max_value);
    //
    //             let edge_label = format!("<Action: {} <BR/> PUCT: {:.2}>",
    //                                      self.last_action.map_or("None".to_string(), |action| format!("{:?}", action)),
    //                                      puct);
    //
    //             graph.add_stmt(Stmt::Edge(edge!(
    //                 parent_id => node_id!(node_id_str);
    //                 attr!("label", edge_label)
    //             )));
    //
    //
    //         },
    //         None => {}
    //     }
    //
    //     let node_text = format!("<Value: {}<BR/> Visits: {:?}<BR/> Prior Prob: {:?}<BR/> Player: {:?}>",
    //         pretty_print_values_array(RotArr::new_from_pov(self.current_player, value_from_pov(self))),
    //         self.visits,
    //         self.prior_prob,
    //         self.current_player
    //     );
    //
    //     graph.add_stmt(stmt!(node!(node_id_str; attr!("label", node_text))));
    //
    //     if depth < max_depth {
    //         for child in self.children {
    //             if child.is_null() {
    //                 continue;
    //             }
    //
    //             (*child).internal_print_graphviz(depth + 1, max_depth, graph, Some(node_id!(node_id_str)), puct_exploration_constant, min_or_max_value);
    //         }
    //     }
    // }

    pub unsafe fn find_greedy_child(
        &self
    ) -> *mut AzNode<
        TGameState,
        TActionType,
        { TNumberOfPlayers },
        { TNumberOfActions },
    > {
        let mut best_score = usize::MIN;
        let mut best_child = null_mut();

        for child in &self.children {
            let child = *child;

            if child.is_null() {
                continue;
            }

            let score = (*child).visits;

            if score > best_score {
                best_score = score;
                best_child = child;
            }
        }


        return best_child;
    }


    pub unsafe fn select_greedy_node(
        &self
    ) -> &AzNode<TGameState, TActionType, { TNumberOfPlayers }, { TNumberOfActions }> {
        let mut node = self;

        loop {
            let node_greedy_child = node.find_greedy_child();

            if node_greedy_child.is_null() {
                break;
            }

            node = &*node_greedy_child;
        }

        return node;
    }
}

pub fn new_root_node<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    state: TGameState,
    state_arena: &mut UnsafeArena<TGameState>,
) -> AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
    let current_player = state.current_player();

    let is_terminal = state
        .is_terminal();

    let (state_index_in_arena, state_ptr) = state_arena
        .alloc_with_index(state);

    return AzNode {
        is_terminal: is_terminal,
        current_player,
        last_action: None,

        state_index_in_arena: state_index_in_arena,
        state: state_ptr,

        prior_prob: 0.0,
        visits: 0,
        win_score: RotArr::zeros(current_player),

        children: [std::ptr::null_mut(); TNumberOfActions],
        has_children: false,
        parent: std::ptr::null_mut(),
        virtual_loss: RotArr::zeros(current_player),
        virtual_visits: 0
    };
}

pub fn new_without_prior_prob<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    state: TGameState,
    parent: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,

    state_arena: &mut UnsafeArena<TGameState>,
) -> AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
    return new_with_prior_prob(state, 0.0, parent, state_arena);
}

pub fn new_with_prior_prob<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    state: TGameState,
    prior_prob: f32,
    parent: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,

    state_arena: &mut UnsafeArena<TGameState>,
) -> AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
    let current_player = state.current_player();

    let is_terminal = state.is_terminal();
    let last_action = state.last_action();
    let (state_index_in_arena, state_ptr) = state_arena
        .alloc_with_index(state);

    AzNode {
        is_terminal: is_terminal,
        current_player: current_player,
        last_action: last_action,

        state: state_ptr,
        state_index_in_arena: state_index_in_arena,

        prior_prob,
        visits: 0,
        win_score: RotArr::zeros(current_player),

        children: [std::ptr::null_mut(); TNumberOfActions],
        has_children: false,
        parent,
        virtual_loss: RotArr::zeros(current_player),
        virtual_visits: 0
    }
}

/// Löschen??? Funktioniert nicht zuverlässig
pub unsafe fn value_from_pov<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
) -> [f32; TNumberOfPlayers] {
    if (*node).visits == 0 {
        return [0.0; TNumberOfPlayers];
    }

    let mut value = [0.0; TNumberOfPlayers];

    for player in 0..TNumberOfPlayers {
        value[player] = (*node).win_score.from_pov_index(player) / (*node).visits as f32;
    }

    return value;
}

pub fn value_for_player_full<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: &AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    for_player: usize,
) -> [f32; TNumberOfPlayers] {
    if node.visits == 0 {
        return [0.0; TNumberOfPlayers];
    }

    let mut value = node
        .win_score
        .rotated_for_i(for_player) / node.visits as f32;

    return value.extract();
}

pub unsafe fn value_for_player<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    node: *mut AzNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >,
    player: usize,
) -> f32 {
    if (*node).visits == 0 {
        return 0.0;
    }

    return ((*node).win_score.get_for_i(player) + (*node).virtual_loss.get_for_i(player)) / (*node).visits as f32;
}
