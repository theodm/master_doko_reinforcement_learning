use std::fmt::{Debug, Display};
use std::hash::Hash;

pub trait AzEnvState<
    TActionType: Copy + Send + Debug,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>: Sized + Clone + Display + Send + Eq + Hash
{
    const GAME_NAME: &'static str;

    fn current_player(&self) -> usize;
    fn is_terminal(&self) -> bool;

    fn rewards_or_none(&self) -> Option<[f32; TNumberOfPlayers]>;

    fn encode_into_memory(&self, memory: &mut [i64]);

    fn allowed_actions_by_action_index(
        &self,
        is_secondary: bool,
        epoch: usize,
    ) -> heapless::Vec<usize, TNumberOfActions>;

    fn number_of_allowed_actions(
        &self,
        epoch: usize,
    ) -> usize;

    fn take_action_by_action_index(
        &self,
        action: usize,
        skip_single: bool,
        epoch: usize,
    ) -> Self;

    fn id(&self) -> u64;

    fn last_action(&self) -> Option<TActionType>;

    fn display_game(&self) -> String;
}
