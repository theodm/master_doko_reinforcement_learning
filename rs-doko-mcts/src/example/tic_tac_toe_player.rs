
// use crate::mcts::find_next_move;

use rand::SeedableRng;
use crate::{example, mcts};
use crate::example::tic_tac_toe::TicTacToeState;
use crate::mcts::mcts::CachedMCTS;

unsafe fn play_single_game(
    random: &mut rand::rngs::SmallRng
) -> example::tic_tac_toe::TicTacToeFinishState {
    let mut state = TicTacToeState::new();

    let mut mcts = CachedMCTS::new(15000, 15000);

    while state.finish_state() == example::tic_tac_toe::TicTacToeFinishState::NotFinished {

        let result = mcts.monte_carlo_tree_search(state.clone(), 2.0, 1400, random);

        state.do_action(result.iter().max_by_key(|x| x.visits).unwrap().action);
    }

    return state.finish_state();
}

pub fn tic_tac_toe_player() {
    println!("Hello, world!");

    let mut rng = rand::rngs::SmallRng::seed_from_u64(999);

    let mut results: [usize; 3] = [0, 0, 0];

    unsafe {
        for i in 0..1000 {
            println!("Playing game {}", i);
            let result = play_single_game(&mut rng);

            match result {
                example::tic_tac_toe::TicTacToeFinishState::Xwon => results[0] += 1,
                example::tic_tac_toe::TicTacToeFinishState::Owon => results[1] += 1,
                example::tic_tac_toe::TicTacToeFinishState::Draw => results[2] += 1,
                _ => {}
            }
        }

        println!("Results after 1000 games:");
        println!("X won: {}", results[0]);
        println!("O won: {}", results[1]);
        println!("Draw: {}", results[2]);

    }


}
