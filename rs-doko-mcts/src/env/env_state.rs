use std::fmt::Display;
use rand::prelude::SmallRng;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::action::allowed_actions::FdoAllowedActions;

/// Ein Zustand einer Umgebung für Zwecke des MCTS.
pub trait McEnvState<
    TActionType,

    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
>: Sized + Clone + Display {
    fn current_player(&self) -> usize;
    fn is_terminal(&self) -> bool;
    fn last_action(&self) -> Option<TActionType>;
    fn possible_states(&self, first_expansion:bool) -> heapless::Vec<Self, TNumberOfActions>;
    fn allowed_actions(&self, first_expansion: bool) -> FdoAllowedActions;
    fn by_action(
        &self,
        action: FdoAction
    ) -> Self;

    fn rewards_or_none(&self) -> Option<[f64; TNumberOfPlayers]>;
    fn random_rollout(
        &self,
        rng: &mut SmallRng
    ) -> [f64; TNumberOfPlayers];
}