use std::any::Any;
use std::cmp::{max, min};
use std::fmt::Debug;
use std::os::unix::raw::mode_t;
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use crate::env::env_state::McEnvState;
use rolling_stats::Stats;

use std::fmt::{self, Write as FmtWrite};
use rs_full_doko::action::allowed_actions::FdoAllowedActions;

/// Ein Knoten im MCTS-Baum.
pub struct McNode<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy,

    const TNumberOfPlayers: usize,
    /// Die Menge an möglichen Aktionen der ganzen Umgebung.
    const TNumberOfActions: usize
> {
    pub state_index_in_arena: usize,
    pub state: *mut TGameState,

    pub is_terminal: bool,
    pub current_player: usize,

    /// Die Aktion, die zum aktuellen Zustand geführt hat.
    /// Zurzeit: Nur für die Ausgabe und Debugging-Zwecke. Passt eigentlich nicht so wirklich.
    pub last_action: Option<TActionType>,

    pub visits: usize,
    pub win_score: f64,

    pub unexpanded_actions: FdoAllowedActions,

    pub children: heapless::Vec<*mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>, TNumberOfActions>,

    pub parent: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
}

impl<
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Debug,

    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {


    pub unsafe fn print_graphviz(
        &self,

        max_depth: usize,

        uct_exploration_constant: f64
    ) -> Graph {
        let mut g = graph!(strict di id!("mcts_tree"));

        self.internal_print_graphviz(
            0,
            max_depth,
            &mut g,
            None,
            uct_exploration_constant
        );

        // print
        // println!("{}", g.print(&mut PrinterContext::default()));

        return g
    }
    unsafe fn internal_print_graphviz(
        &self,

        depth: usize,
        max_depth: usize,

        graph: &mut Graph,
        // Einduetiger Bezeichner des Elternknotens (falls vorhanden)
        parent_id: Option<NodeId>,

        uct_exploration_constant: f64
    ) {
        // Eindeutiger Bezeichner des aktuellen Knotens
        let node_id_str = format!("node_{:p}", self);

        match parent_id {
            Some(parent_id) => {
                let uct = self.uct(self.visits, uct_exploration_constant);

                let edge_label = format!(
                    "<Action: {} <BR/> UCT: {:.2}>",
                    self.last_action.map_or("None".to_string(), |action| format!("{:?}", action)),
                    uct
                );

                graph.add_stmt(stmt!(edge!(
                    parent_id => node_id!(node_id_str);
                    attr!("label", edge_label)
                )));


            },
            None => {}
        }

        let node_text = format!("<Value: {}<BR/> Visits: {:?}<BR/> Player: {:?}>",
            self.win_score,
            self.visits,
            self.current_player
        );

        graph.add_stmt(stmt!(node!(
            node_id_str;
            attr!("label", node_text)
        )));

        if depth < max_depth {
            for child in &self.children {
                if child.is_null() {
                    continue;
                }

                (*(*child)).internal_print_graphviz(depth + 1, max_depth, graph, Some(node_id!(node_id_str)), uct_exploration_constant);
            }
        }
    }

    /// Erstellt den Wurzelknoten des MCTS-Baums.
    pub unsafe fn new_root_node(
        state: TGameState,

        state_arena: &mut UnsafeArena<TGameState>
    ) -> McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {

        let is_terminal = state.is_terminal();
        let current_player = state.current_player();
        let last_action = state.last_action();

        let (state_index_in_arena, state) = state_arena.alloc_with_index(state);

        return McNode {
            is_terminal,
            current_player,
            last_action,

            state_index_in_arena,
            state,


            visits: 0,
            win_score: 0.0,
            //wins: Stats::new(),
            children: heapless::Vec::new(),
            parent: std::ptr::null_mut(),

            unexpanded_actions: (*state).allowed_actions(true),
        };
    }

    /// Erstellt einen neuen Knoten im MCTS-Baum (muss im
    /// Rahmen der Expansion aufgerufen werden).
    pub unsafe fn new(
        state: TGameState,

        parent: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,

        state_arena: &mut UnsafeArena<TGameState>
    ) -> McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
        let is_terminal = state.is_terminal();
        let current_player = state.current_player();
        let last_action = state.last_action();

        let (state_index_in_arena, state) = state_arena.alloc_with_index(state);

        return McNode {
            is_terminal,
            current_player,
            last_action,

            state_index_in_arena,
            state,

            visits: 0,
            win_score: 0.0,
            //wins: Stats::new(),
            children: heapless::Vec::new(),
            parent,

            unexpanded_actions: (*state).allowed_actions(false),
        };
    }

    pub unsafe fn min_max_normalized_q(
        &self,
        child: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>
    ) -> f64 {
        if self.children.is_empty() {
            return 1.0;
        }

        let mut min_q = f64::INFINITY;
        let mut max_q = f64::NEG_INFINITY;

        for &c in &self.children {
            let q = if (*c).visits > 0 {
                (*c).win_score / (*c).visits as f64
            } else {
                0.0
            };
            if q < min_q { min_q = q; }
            if q > max_q { max_q = q; }
        }

        // Falls alle Werte gleich sind oder noch nicht gesetzt, gib 1 zurück
        if (max_q - min_q).abs() < std::f64::EPSILON {
            return 1.0;
        }

        let q = if (*child).visits > 0 {
            (*child).win_score / (*child).visits as f64
        } else {
            0.0
        };

        2.0 * (q - min_q) / (max_q - min_q) - 1.0
    }

    /// Gibt den aktuellen UCT-Wert des Knotens zurück.
    pub unsafe fn uct(
        &self,
        total_visits: usize,
        uct_exploration_constant: f64
    ) -> f64 {
        if self.parent.is_null() {
            return 0.0;
        }

        if self.visits == 0 {
            return f64::INFINITY;
        }

        let norm_q = (*self.parent).min_max_normalized_q(self as *const _ as *mut _);

        return norm_q + uct_exploration_constant * ((total_visits as f64).ln() / self.visits as f64).sqrt();
    }

    /// Gibt das Kind mit dem höchsten UCT-Wert zurück. (muss im
    /// Rahmen der Selektion aufgerufen werden).
    pub unsafe fn find_best_child(
        &self,
        uct_exploration_constant: f64
    ) -> *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions> {
        let mut best_uct = f64::NEG_INFINITY;
        let mut best_child = std::ptr::null_mut();

        for child in self.children.clone() {
           // let uct_exploration_constant = uct_exploration_constant * f64::max(1f64, (self.wins.max - self.wins.min).abs());

            let uct = (*child).uct(self.visits, uct_exploration_constant);

            if uct > best_uct {
                best_uct = uct;
                best_child = child;
            }
        }

        return best_child;
    }
}

// pub unsafe fn print_tree<
//     TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
//     TActionType: Copy + Debug,
//     const TNumberOfPlayers: usize,
//     const TNumberOfActions: usize
// >(
//     node: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
//     depth: usize,
//     max_depth: usize,
//     uct_exploration_constant: f64
// ) {
//     let uct = if !(*node).parent.is_null() {
//         (*node).uct((*(*node).parent).visits, uct_exploration_constant)
//     } else {
//         0.0
//     };
//
//     // prepend spaces by depth
//     for _ in 0..depth {
//         print!(" ");
//     }
//
//     println!("Last Action: {} Visits: {}, Win Score: {}, Win Ratio: {}, Current Player: {},  UCT: {}, StdDev: {}, MinMax: {}",
//              {
//                 match (*node).last_action {
//                     Some(state) => format!("{:?}", state),
//                     None => "None".to_string()
//                 }
//
//              },
//              (*node).visits,
//              (*node).win_score,
//              (*node).win_score / max(1, (*node).visits) as f64,
//              (*node).current_player,
//              uct,
//              (*node).wins.std_dev,
//              f64::max(1f64, ((*node).wins.max - (*node).wins.min).abs())
//     );
//
//     if depth == max_depth {
//         return;
//     }
//
//     for child in (*node).children.clone() {
//         print_tree(child, depth + 1, max_depth, uct_exploration_constant);
//     }
// }

/// Drucke den MCTS-Baum in `out` statt auf stdout.
///
/// `W` kann z.B. ein `String` sein.
pub unsafe fn print_tree<W,
    TGameState,
    TActionType,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
>(
    node: *mut McNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>,
    depth: usize,
    max_depth: usize,
    uct_exploration_constant: f64,
    out: &mut W,
) -> fmt::Result
where
    W: FmtWrite,
    TGameState: McEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Debug,
{
    // Berechne UCT-Wert
    let uct = if !(*node).parent.is_null() {
        (*node).uct( (*(*node).parent).visits, uct_exploration_constant )
    } else {
        0.0
    };

    // Einrückung
    for _ in 0..depth {
        write!(out, " ")?;
    }

    let min_max_q = if !(*node).parent.is_null() {
        (*(*node).parent).min_max_normalized_q(node)
    } else {
        0.0
    };

    // Knotendaten
    writeln!(
        out,
        "Last Action: {} Visits: {} Win Score: {} Win Ratio: {:.3} Current Player: {}  UCT: {:.3} Blub-Wert: {:.3}", //, StdDev: {:.3}, MinMax: {:.3}",
        match (*node).last_action {
            Some(act) => format!("{:?}", act),
            None      => "None".to_string(),
        },
        (*node).visits,
        (*node).win_score,
        (*node).win_score / max(1, (*node).visits) as f64,
        (*node).current_player,
        uct,
        min_max_q
        //(*node).wins.std_dev,
        //f64::max(1.0, ((*node).wins.max - (*node).wins.min).abs()),
    )?;

    // Abbruch bei Erreichen der maximalen Tiefe
    if depth == max_depth {
        return Ok(());
    }

    // Rekursiv über Kinder
    for &child in &(*node).children {
        print_tree(child, depth + 1, max_depth, uct_exploration_constant, out)?;
    }

    Ok(())
}