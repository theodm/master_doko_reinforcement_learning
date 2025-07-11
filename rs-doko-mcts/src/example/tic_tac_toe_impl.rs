use std::fmt::{Display, Formatter};
use std::rc::Rc;
use rand::prelude::{IndexedRandom, SmallRng};
use rand::Rng;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::action::allowed_actions::FdoAllowedActions;
use crate::env::env_state::McEnvState;
use crate::example::tic_tac_toe::{board_to_str, TicTacToeFinishState, TicTacToeState};
use crate::example::tic_tac_toe;

impl Display for TicTacToeState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{}", board_to_str(self.board));
    }
}

impl McEnvState<usize, 2, 9> for TicTacToeState {
    fn current_player(&self) -> usize {
        self.current_player()
    }

    fn is_terminal(&self) -> bool {
        self.finish_state() != tic_tac_toe::TicTacToeFinishState::NotFinished
    }

    fn last_action(&self) -> Option<usize> {
        return self.last_action;
    }

    fn possible_states(&self, first_expansion: bool) -> heapless::Vec<Self, 9> {
        self.possible_states()
    }

    fn allowed_actions(&self, first_expansion: bool) -> FdoAllowedActions {
        todo!()
    }

    fn by_action(&self, action: FdoAction) -> Self {
        todo!()
    }

    fn rewards_or_none(&self) -> Option<[f64; 2]> {
        return if self.finish_state() == TicTacToeFinishState::NotFinished {
            None
        } else {
            Some(self.rewards())
        }
    }

    // fn random_state(&self, random: &mut SmallRng) -> Self {
    //     return self.random_state(random);
    // }

    fn random_rollout(&self, rng: &mut SmallRng) -> [f64; 2] {
        let mut state = self.clone();

        while state.finish_state() == TicTacToeFinishState::NotFinished {
            state = state.random_state(rng);
        }

        return state.rewards();
    }
}