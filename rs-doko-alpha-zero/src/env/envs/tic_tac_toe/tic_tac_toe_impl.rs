use rand::prelude::{IndexedRandom, SmallRng};
use rand::Rng;
use rs_tictactoe::tictactoe;
use rs_tictactoe::tictactoe::TicTacToeFinishState;
use rs_tictactoe::tictactoe::{board_to_str, FieldState, TicTacToeState};
use std::fmt::{Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use crate::env::env_state::AzEnvState;

impl AzEnvState<usize, 2, 9> for TicTacToeState {
    const GAME_NAME: &'static str = "ttt_new_1";

    fn current_player(&self) -> usize {
        match self.finish_state {
            TicTacToeFinishState::NotFinished => self.current_player(),
            _ => 0,
        }
    }

    fn is_terminal(&self) -> bool {
        self.finish_state() != TicTacToeFinishState::NotFinished
    }

    fn rewards_or_none(&self) -> Option<[f32; 2]> {
        return if self.finish_state() == TicTacToeFinishState::NotFinished {
            None
        } else {
            Some(self.rewards())
        };
    }

    fn encode_into_memory(&self, memory: &mut [i64]) {
        if self.current_player() == 1 {
            for i in 0..9 {
                memory[i] = match self.board[i] {
                    FieldState::Empty => 0,
                    FieldState::X => -1,
                    FieldState::O => 1,
                };
            }
        } else {
            for i in 0..9 {
                memory[i] = match self.board[i] {
                    FieldState::Empty => 0,
                    FieldState::X => 1,
                    FieldState::O => -1,
                };
            }
        }
    }


    fn allowed_actions_by_action_index(&self, secondary: bool, epoch: usize) -> heapless::Vec<usize, 9> {
        return self.allowed_actions();
    }

    fn number_of_allowed_actions(&self, epoch: usize) -> usize {
        return self.allowed_actions().len();
    }

    fn take_action_by_action_index(&self, action: usize, skip_single: bool, epoch: usize) -> Self {
        let mut new_board = self.board.clone();

        new_board[action] = match self.current_player() {
            0 => FieldState::X,
            1 => FieldState::O,
            _ => panic!("Invalid player"),
        };

        return TicTacToeState {
            board: new_board,
            finish_state: tictactoe::calc_finish_state(new_board),
        };
    }

    fn id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        self.hash(&mut hasher);

        hasher.finish()
    }

    fn last_action(&self) -> Option<usize> {
        None
    }

    fn display_game(&self) -> String {
        "TicTacToe".to_string()
    }
}
