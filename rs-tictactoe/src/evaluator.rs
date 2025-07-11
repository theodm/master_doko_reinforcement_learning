use rand::prelude::IndexedRandom;
use crate::tictactoe::{TicTacToeFinishState, TicTacToeState};

pub type EvTicTacToePolicy<'a> = dyn Fn(
    &TicTacToeState,
    &mut rand::rngs::SmallRng
) -> usize + 'a;

pub fn random_policy(
    tic_tac_toe_state: &TicTacToeState,
    rng: &mut rand::rngs::SmallRng
) -> usize {
    return *tic_tac_toe_state
        .allowed_actions()
        .choose(rng)
        .unwrap();
}

pub fn evaluate_policy_against_random<'a>(
    policy: &EvTicTacToePolicy,

    number_of_matchups: usize,

    // Dieser RNG wird für alle anderen Zufallsentscheidungen, z.B. die der Policies
    // und deren internes Verhalten, verwendet.
    rng: &'a mut rand::rngs::SmallRng
) -> (i32, i32, i32) {
    let mut x_win = 0;
    let mut o_win = 0;
    let mut draws = 0;

    let policies = [
        &random_policy,
        policy
    ];

    for i in 0..number_of_matchups {
        let mut current_state = TicTacToeState::new();

        loop {
            if (current_state.finish_state() != TicTacToeFinishState::NotFinished) {
                break;
            }

            let current_player = current_state.current_player();

            let action = policies[current_player as usize](
                &current_state,
                rng
            );

            current_state.take_action(action);

        }

        match current_state.finish_state {
            TicTacToeFinishState::Xwon => {
                x_win += 1
            },
            TicTacToeFinishState::Owon => {
                o_win += 1
            },
            TicTacToeFinishState::Draw => {
                draws += 1
            },
            TicTacToeFinishState::NotFinished => panic!("Game not finished")
        };
    }

    return (
        x_win,
        o_win,
        draws
    );
}
