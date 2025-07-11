// use crate::node::McState;

use log::debug;
use rand::Rng;

#[repr(usize)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum FieldState {
    Empty,
    X,
    O
}

#[derive(Debug, PartialEq, Clone)]
pub struct TicTacToeState {
    pub board: [FieldState; 9],
    pub finish_state: TicTacToeFinishState,
    pub last_action: Option<usize>
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum TicTacToeFinishState {
    Xwon,
    Owon,
    Draw,
    NotFinished
}

fn calc_finish_state(board: [FieldState; 9]) -> TicTacToeFinishState {
    let winning_lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],

        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],

        [0, 4, 8],
        [2, 4, 6]
    ];

    let mut atleast_one_field_remaining = false;

    for line in winning_lines {
        let mut all_x = true;
        let mut all_o = true;

        for index in line {
            let pos_char = board[index];

            if pos_char != FieldState::X {
                all_x = false;
            }

            if pos_char != FieldState::O {
                all_o = false;
            }

            if pos_char == FieldState::Empty {
                atleast_one_field_remaining = true;
            }
        }

        if all_x {
            return TicTacToeFinishState::Xwon;
        }

        if all_o {
            return TicTacToeFinishState::Owon;
        }
    }

    if atleast_one_field_remaining {
        return TicTacToeFinishState::NotFinished;
    }

    TicTacToeFinishState::Draw
}

impl TicTacToeState {
    pub fn new() -> Self {
        Self {
            board: [FieldState::Empty; 9],
            finish_state: TicTacToeFinishState::NotFinished,
            last_action: None
        }
    }

    pub fn do_action(&mut self, index: usize) {
        if self.finish_state != TicTacToeFinishState::NotFinished {
            panic!("Cannot make a move: the game has already finished.");
        }

        if self.board[index] != FieldState::Empty {
            panic!("Cannot make a move: the chosen field is not empty.");
        }

        let symbol_to_play = match self.current_player() {
            0 => FieldState::X,
            1 => FieldState::O,
            _ => panic!("Invalid player."),
        };

        self.board[index] = symbol_to_play;

        self.finish_state = calc_finish_state(self.board);
        self.last_action = Some(index);
    }
    pub fn current_player(&self) -> usize {
        let mut x_count = 0;
        let mut o_count= 0;

        for field in &self.board {
            match field {
                FieldState::X => x_count += 1,
                FieldState::O => o_count += 1,
                FieldState::Empty => {}
            }
        }

        if x_count == o_count {
            0
        } else {
            1
        }
    }
    pub fn finish_state(&self) -> TicTacToeFinishState {
        return self.finish_state
    }

    pub fn rewards(&self) -> [f64; 2] {
        assert!(self.board.len() == 9);

        match self.finish_state() {
            TicTacToeFinishState::Xwon => [1.0, -1.0],
            TicTacToeFinishState::Owon => [-1.0, 1.0],
            TicTacToeFinishState::Draw => [0.0, 0.0],
            TicTacToeFinishState::NotFinished => [0.0, 0.0]
        }
    }

    pub fn random_state(&self, random: &mut rand::rngs::SmallRng) -> Self {
        let mut random_fields: [isize; 9] = [-1; 9];

        let mut index_in_random_fields = 0;
        for (index, field) in self.board.iter().enumerate() {
            if field == &FieldState::Empty {
                random_fields[index_in_random_fields] = index as isize;
                index_in_random_fields += 1;
            }
        }

        let random_index = random.gen_range(0..index_in_random_fields);

        let mut new_board = self.board.clone();

        new_board[random_fields[random_index] as usize] = match self.current_player() {
            0 => FieldState::X,
            1 => FieldState::O,
            _ => panic!("Invalid player")
        };

        return TicTacToeState { board: new_board, finish_state: calc_finish_state(new_board), last_action: Some(random_index) };
    }

    pub fn possible_states(&self) -> heapless::Vec<TicTacToeState, 9> {
        assert!(self.board.len() == 9);

        if self.finish_state() != TicTacToeFinishState::NotFinished {
            return heapless::Vec::new();
        }

        let current_player = self.current_player();
        let mut possible_states = heapless::Vec::new();

        for (index, field) in self.board.iter().enumerate() {
            if field == &FieldState::Empty {
                let mut new_board = self.board.clone();
                new_board[index] = match current_player {
                    0 => FieldState::X,
                    1 => FieldState::O,
                    _ => panic!("Invalid player")
                };

                possible_states.push(TicTacToeState { board: new_board, finish_state: calc_finish_state(new_board), last_action: Some(index) });
            }
        }

        return possible_states;
    }

}

pub fn board_to_str(
    board: [FieldState; 9]
) -> String {
    let mut board_str = String::new();

    for field in board {
        match field {
            FieldState::Empty => board_str.push_str(" "),
            FieldState::X => board_str.push_str("X"),
            FieldState::O => board_str.push_str("O"),
        }
    }

    return board_str;
}

fn str_to_state(
    board_str: &str
) -> TicTacToeState {
    let board = str_to_board(board_str);
    TicTacToeState { board, finish_state: calc_finish_state(board), last_action: None }
}

fn str_to_board(
    board_str: &str
) -> [FieldState; 9] {
    let mut board = [FieldState::Empty; 9];

    for (index, field) in board_str.chars().enumerate() {
        match field {
            ' ' => board[index] = FieldState::Empty,
            'X' => board[index] = FieldState::X,
            'O' => board[index] = FieldState::O,
            _ => panic!("Invalid field state")
        }
    }

    board
}

#[cfg(test)]
mod tests {
    use crate::example::tic_tac_toe::{str_to_board, str_to_state, TicTacToeFinishState, TicTacToeState};


    #[test]
    fn test_possible_states_empty_board() {
        let state = TicTacToeState::new();
        let possible_states = state.possible_states();

        assert_eq!(str_to_state("X        "), possible_states[0]);
        assert_eq!(str_to_state(" X       "), possible_states[1]);
        assert_eq!(str_to_state("  X      "), possible_states[2]);
        assert_eq!(str_to_state("   X     "), possible_states[3]);
        assert_eq!(str_to_state("    X    "), possible_states[4]);
        assert_eq!(str_to_state("     X   "), possible_states[5]);
        assert_eq!(str_to_state("      X  "), possible_states[6]);
        assert_eq!(str_to_state("       X "), possible_states[7]);
        assert_eq!(str_to_state("        X"), possible_states[8]);
        assert_eq!(9, possible_states.len());
    }

    #[test]
    fn test_possible_states_full_board() {
        let state =  str_to_state("XXOOOXXXO");
        let possible_states = state.possible_states();

        assert!(possible_states.is_empty());
    }

    #[test]
    fn test_possible_states_some_on_board() {
        let state = str_to_state("X  OX    ");
        let possible_states = state.possible_states();

        assert_eq!(str_to_state("XO OX    "), possible_states[0]);
        assert_eq!(str_to_state("X OOX    "), possible_states[1]);
        assert_eq!(str_to_state("X  OXO   "), possible_states[2]);
        assert_eq!(str_to_state("X  OX O  "), possible_states[3]);
        assert_eq!(str_to_state("X  OX  O "), possible_states[4]);
        assert_eq!(str_to_state("X  OX   O"), possible_states[5]);
        assert_eq!(6, possible_states.len());
    }

    #[test]
    fn test_finish_state_empty_board() {
        let state = TicTacToeState::new();
        let finish_state = state.finish_state();

        assert_eq!(TicTacToeFinishState::NotFinished, finish_state);
    }

    #[test]
    fn test_finish_state_draw() {
        let state = str_to_state("XXOOOXXXO");
        let finish_state = state.finish_state();

        assert_eq!(TicTacToeFinishState::Draw, finish_state);
    }

    #[test]
    fn test_finish_state_x_won() {
        let state = str_to_state("XXXOO    ");
        let finish_state = state.finish_state();

        assert_eq!(TicTacToeFinishState::Xwon, finish_state);
    }

    #[test]
    fn test_finish_state_o_won() {
        let state = str_to_state("OOOXX    ");
        let finish_state = state.finish_state();

        assert_eq!(TicTacToeFinishState::Owon, finish_state);
    }

    #[test]
    fn test_finish_state_test_case_x_won() {
        let state = str_to_state("OXOXXOOXX");
        let finish_state = state.finish_state();

        assert_eq!(TicTacToeFinishState::Xwon, finish_state);
    }
}