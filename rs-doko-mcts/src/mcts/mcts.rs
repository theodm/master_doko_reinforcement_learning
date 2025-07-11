use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr::null_mut;
use graphviz_rust::printer::{DotPrinter, PrinterContext};
use rand::prelude::SmallRng;
use rand::Rng;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use crate::env::env_state::McEnvState;
use crate::mcts::node::{print_tree, McNode};
use crate::save_log_maybe::save_log_maybe;

#[derive(Debug)]
pub struct Move<TActionType: Copy> {
    pub action: TActionType,
    pub visits: usize,
    pub value: f64
}


struct MCTS<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> {
    state: PhantomData<TGameState>,
    action: PhantomData<TActionType>,
    number_of_players: PhantomData<heapless::Vec<usize, TNumberOfPlayers>>,
    number_of_actions: PhantomData<heapless::Vec<usize, TNumberOfActions>>

}

impl<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> MCTS<
    TGameState,
    TActionType,
    TNumberOfPlayers,
    TNumberOfActions
> {
    /// Selektiert den vielversprechendsten Knoten im MCTS-Baum.
    unsafe fn select_promising_node(
        node: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,

        uct_exploration_constant: f64
    ) -> *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
        let mut node = node;

        while (*node).children.len() > 0 {
            if (*node).unexpanded_actions.len() > 0 {
                return node;
            }

            node = (*node).find_best_child(uct_exploration_constant);
        }

        return node;
    }

    unsafe fn expand_single(
        promising_node: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,

        node_arena: &mut UnsafeArena<McNode<
            TGameState,
            TActionType,
            TNumberOfPlayers,
            TNumberOfActions,
        >>,
        state_arena: &mut UnsafeArena<TGameState>,
        rng: &mut SmallRng
    ) -> *mut McNode<TGameState, TActionType, { TNumberOfPlayers }, { TNumberOfActions }> {
        let possible_action = (*promising_node)
            .unexpanded_actions
            .random(rng);

        (*promising_node)
            .unexpanded_actions
            .remove(possible_action);

        let new_state = (*(*promising_node)
            .state)
            .by_action(
                possible_action
            );

        let new_node = node_arena.alloc(McNode::new(
            new_state,
            promising_node,
            state_arena
        ));

        (*promising_node)
            .children
            .push(new_node)
            .unwrap();

        return new_node;

    }

    /// Erweitert einen Knoten im MCTS-Baum um alle von dort aus erreichbaren Zustände.
    unsafe fn expand_node(
        promising_node: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
        is_first_expansion: bool,

        node_arena: &mut UnsafeArena<McNode<
            TGameState,
            TActionType,
            TNumberOfPlayers,
            TNumberOfActions,
        >>,
        state_arena: &mut UnsafeArena<TGameState>
    ) {
        let possible_states = (*promising_node)
            .state
            .as_ref()
            .unwrap()
            .possible_states(is_first_expansion);

        for state in possible_states {
            let new_node = node_arena.alloc(McNode::new(state, promising_node, state_arena));

            (*promising_node)
                .children
                .push(new_node)
                .unwrap();
        }

        // Von Nicht-Blättern brauchen wir den Zustand nicht mehr und
        // können den entsprechenden Speicherplatz freigeben.
        state_arena.free_single((*promising_node).state_index_in_arena);
        (*promising_node).state = null_mut();

        return;
    }


    /// Die Backpropagation im MCTS-Baum.
    unsafe fn backpropagate(
        // Der Knoten, von dem aus die Backpropagation startet. Das ist in der Regel der
        // Knoten unterhalb des vielversprechendsten Knotens, der im Selektionsschritt
        // gefunden wurde oder der vielversprechendste Knoten selbst.
        node_to_explore: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,

        result: [f64; TNumberOfPlayers]
    ) {
        let mut temp_node = node_to_explore;

        loop {
            let parent_node = (*temp_node).parent;

            (*temp_node).visits += 1;
            //(*temp_node).wins.update(result[(*temp_node).current_player]);

            if parent_node.is_null() {
                break;
            }

            (*temp_node).win_score += result[(*parent_node).current_player];

            temp_node = parent_node;
        }
    }

    /// Der Einstiegspunkt in den MCTS.
    ///
    /// Achtung: Die Arena für die Zustände und Knoten müssen bereits im Vorhinein
    /// eine geeignete Größe haben. Außerdem muss die Arena danach zurückgesetzt werden.
    pub unsafe fn monte_carlo_tree_search(
        state: TGameState,

        uct_exploration_constant: f64,
        iterations: usize,

        random: &mut SmallRng,

        node_arena: &mut UnsafeArena<McNode<
            TGameState,
            TActionType,
            TNumberOfPlayers,
            TNumberOfActions,
        >>,
        state_arena: &mut UnsafeArena<
            TGameState
        >,

    ) -> heapless::Vec<Move<TActionType>, TNumberOfActions> {
        let root_node = node_arena.alloc(McNode::new_root_node(state, state_arena));

        for it in 0..iterations {
            // Selektion
            //
            // Wir wählen den vielversprechendsten Knoten im MCTS-Baum aus.
            let promising_node = MCTS::select_promising_node(
                root_node,
                uct_exploration_constant
            );

            // Expansion
            let node_to_explore = if (*promising_node).unexpanded_actions.len() > 0 {
                MCTS::expand_single(
                    promising_node,
                    node_arena,
                    state_arena,
                    random
                )
            } else {
                promising_node
            };

            // Und tatsächlich eine Simulation durchführen.
            let result: [f64; TNumberOfPlayers] = (*(*node_to_explore)
                .state)
                .random_rollout(random);

            // Backpropagation
            MCTS::backpropagate(node_to_explore, result);

            if it == iterations - 1 {
                let mut buf = String::new();

                print_tree(root_node, 0, 2, uct_exploration_constant, &mut buf)
                    .expect("Fehler beim Serialisieren des MCTS-Baums");
                // In Datei speichern

                // println!("{}", buf);

                if random.random_bool((1f32 / 1000f32) as f64) {
                    save_log_maybe(Some(buf), "mcts_logs")
                        .expect("Konnte MCTS-Log nicht speichern");
                }
            }
        }

        let mut moves = heapless::Vec::new();

        for child in (*root_node).children.clone() {
            moves.push(Move {
                action: (*child).last_action.unwrap(),
                visits: (*child).visits,
                value: (*child).win_score / (*child).visits as f64
            }).unwrap();
        }

        return moves;
    }
}

pub struct CachedMCTS<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy,

    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> {
    node_arena: UnsafeArena<McNode<
        TGameState,
        TActionType,
        TNumberOfPlayers,
        TNumberOfActions,
    >>,
    state_arena: UnsafeArena<
        TGameState
    >,
}

unsafe impl<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
>
Send for CachedMCTS<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {}

impl<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> CachedMCTS<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
    pub fn new(
        expected_number_of_nodes: usize,
        expected_number_of_states: usize
    ) -> Self {
        Self {
            node_arena: UnsafeArena::new(expected_number_of_nodes),
            state_arena: UnsafeArena::new(expected_number_of_states),
        }
    }

    pub unsafe fn monte_carlo_tree_search(
        &mut self,
        state: TGameState,

        uct_exploration_constant: f64,
        iterations: usize,

        random: &mut SmallRng,
    ) -> heapless::Vec<Move<TActionType>, TNumberOfActions> {
        let result = MCTS::monte_carlo_tree_search(
            state,
            uct_exploration_constant,
            iterations,
            random,
            &mut self.node_arena,
            &mut self.state_arena
        );

        self.reset();

        return result;
    }

    fn reset(&mut self) {
        self.node_arena.free_all();
        self.state_arena.free_all();
    }
}







//
// unsafe fn find_most_visited_child<
//     TGameState: McEnvState<TActionType, TNumberOfPlayers>,
//     TActionType: Copy,
//     const TNumberOfPlayers: usize
// >(
//     node: &McNode<TGameState, TActionType, TNumberOfPlayers>
// ) -> &McNode<TGameState, TActionType, TNumberOfPlayers> {
//     let mut most_visits = 0;
//     let mut most_visited_child = None;
//
//     for child in &node.children {
//         if child.visits > most_visits {
//             most_visits = child.visits;
//             most_visited_child = Some(child);
//         }
//     }
//
//     return most_visited_child.unwrap();
// }
//
