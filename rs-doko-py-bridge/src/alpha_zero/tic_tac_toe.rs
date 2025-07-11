use std::sync::Arc;
use async_trait::async_trait;
use rand::prelude::{IndexedRandom, SmallRng};
use rand::SeedableRng;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluate::evaluate_single;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train::AzEvaluator;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
use rs_doko_alpha_zero::alpha_zero::train::glob::GlobalStats;
use rs_doko_alpha_zero::env::env_state::AzEnvState;
use rs_tictactoe::tictactoe::{TicTacToeFinishState, TicTacToeState};
use rs_unsafe_arena::unsafe_arena::UnsafeArena;

pub async fn policy(
    state: &TicTacToeState,

    batch_processor: &mut CachedNetworkBatchProcessorSender<
        TicTacToeState,
        usize,
        9,
        2,
        9
    >,

    rng: &mut SmallRng
) -> usize {

    let action = evaluate_single(
        state,
        AlphaZeroEvaluationOptions {
            iterations: 50,
            dirichlet_alpha: 1.0,
            dirichlet_epsilon: 0.25,
            puct_exploration_constant: 3.5,
            min_or_max_value: 1.0,
            par_iterations: 2,
            virtual_loss: 0.0,
        },
        batch_processor,
        &mut UnsafeArena::new(10000),
        &mut UnsafeArena::new(10000),
        rng
    ).await;

    return action.0 as usize;
}

pub async fn tic_tac_toe_continous_evaluation(
    mut batch_processor: BatchProcessorSender<[i64; 9], ([f32; 2], [f32; 9])>,

    num_epoch: usize,
    num_learning_steps: usize,

    tensorboard_sender: Arc<dyn TensorboardSender>
) {
    let mut handles = Vec::with_capacity(1000);

    for i in 0..1000 {
        let mut batch_processor = CachedNetworkBatchProcessorSender::<TicTacToeState, usize, 9, 2, 9>::new(
            batch_processor.clone(),
            512,
            Arc::new(GlobalStats::new())
        );

        let game = tokio::task::spawn(async move {
            let mut rng = SmallRng::from_os_rng();
            let mut state = TicTacToeState::new();

            loop {
                if state.is_terminal() {
                    break;
                }

                let current_player = state.current_player();

                if current_player == 0 {
                    let action = *state.allowed_actions().choose(&mut rng).unwrap();

                    state = state.take_action_by_action_index(action, false, 1000);
                } else {
                    let action = policy(&state, &mut batch_processor, &mut rng).await;

                    state = state.take_action_by_action_index(action, false, 1000);
                }
            }

            return state.finish_state();
        });

        handles.push(game);
    }

    let mut x_wins = 0;
    let mut o_wins = 0;
    let mut draws = 0;

    for handle in handles {
        let finish_state = handle.await.unwrap();

        match finish_state {
            TicTacToeFinishState::Xwon => {
                x_wins += 1;
            }
            TicTacToeFinishState::Owon => {
                o_wins += 1;
            }
            TicTacToeFinishState::Draw => {
                draws += 1;
            }
            _ => {
                panic!("Invalid finish state");
            }
        }
    }

    tensorboard_sender.scalar("epoch/x_win", x_wins as f32, num_epoch as i64);
    tensorboard_sender.scalar("epoch/o_win", o_wins as f32, num_epoch as i64);
    tensorboard_sender.scalar("epoch/draws", draws as f32, num_epoch as i64);
    tensorboard_sender.scalar("epoch/total", (x_wins as f32 / (o_wins as f32 + draws as f32)), num_epoch as i64);

    tensorboard_sender.scalar("training_steps/x_win", x_wins as f32, num_learning_steps as i64);
    tensorboard_sender.scalar("training_steps/o_win", o_wins as f32, num_learning_steps as i64);
    tensorboard_sender.scalar("training_steps/draws", draws as f32, num_learning_steps as i64);
    tensorboard_sender.scalar("training_steps/total", (x_wins as f32 / (o_wins as f32 + draws as f32)), num_learning_steps as i64);

    println!("{num_epoch} X wins: {}, O wins: {}, Draws: {}", x_wins, o_wins, draws);
}

pub struct AzTicTacToeEvaluator {}

#[async_trait]
impl AzEvaluator<9, 2, 9> for AzTicTacToeEvaluator {
    async fn evaluate(
        &self,
        batch_processor: BatchProcessorSender<[i64; 9], ([f32; 2], [f32; 9])>,
        num_epoch: usize,
        num_learning_steps: usize,
        alpha_zero_train_options: &AlphaZeroTrainOptions,
        tensorboard_sender: Arc<dyn TensorboardSender>
    ) {
        tic_tac_toe_continous_evaluation(batch_processor, num_epoch, num_learning_steps, tensorboard_sender).await;
    }
}
