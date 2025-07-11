// use std::cell::RefCell;
// use std::rc::Rc;
// use std::sync::Arc;
// use std::time::Duration;
// use async_trait::async_trait;
// use rand::prelude::{IndexedRandom, SmallRng};
// use rand::SeedableRng;
// use tch::{nn, Device};
// use tensorboard_rs::summary_writer::SummaryWriter;
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluate::evaluate_single;
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_train::{train_alpha_zero, EvaluationFn};
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
// use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
// use rs_doko_alpha_zero::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
// use rs_doko_alpha_zero::alpha_zero::mcts::mcts::tensor_apply_mask_vec;
// use rs_doko_alpha_zero::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
// use rs_doko_alpha_zero::env::env_state::AzEnvState;
// use rs_doko_alpha_zero::env::envs::tic_tac_toe::tic_tac_toe_net::TicTacToeNetwork;
// use rs_doko_evaluator::doko::policy::policy::EvDokoPolicy;
// use rs_doko_evaluator::doko_async::policy::policy::EvAsyncDokoPolicy;
// use rs_tictactoe::tictactoe::{TicTacToeFinishState, TicTacToeState};
// use rs_unsafe_arena::unsafe_arena::UnsafeArena;
//
//
// pub async fn policy(
//     state: &TicTacToeState,
//
//     batch_processor: &mut CachedNetworkBatchProcessorSender<
//         TicTacToeState,
//         usize,
//         9,
//         2,
//         9
//     >,
//
//     rng: &mut SmallRng
// ) -> usize {
//
//     let action = evaluate_single(
//         state,
//         AlphaZeroEvaluationOptions {
//             iterations: 50,
//             dirichlet_alpha: 1.0,
//             dirichlet_epsilon: 0.25,
//             puct_exploration_constant: 3.5,
//             min_or_max_value: 1.0,
//             par_iterations: 2,
//             virtual_loss: 0.0,
//         },
//         batch_processor,
//         &mut UnsafeArena::new(10000),
//         &mut UnsafeArena::new(10000),
//         rng
//     ).await;
//
//     return action as usize;
// }
//
// pub async fn async_evaluate_ttt(
//     mut batch_processor: BatchProcessorSender<[i64; 9], ([f32; 2], [f32; 9])>,
//
//     num_epoch: usize,
//     num_learning_steps: usize,
//
//     tensorboard_sender: Arc<dyn TensorboardSender>
// ) {
//
//     let mut handles = Vec::with_capacity(1000);
//
//     for i in 0..1000 {
//         let mut batch_processor = CachedNetworkBatchProcessorSender::<TicTacToeState, usize, 9, 2, 9>::new(
//             batch_processor.clone(),
//             512
//         );
//
//         let game = tokio::task::spawn(async move {
//             let mut rng = SmallRng::from_os_rng();
//             let mut state = TicTacToeState::new();
//
//             loop {
//                 if state.is_terminal() {
//                     break;
//                 }
//
//                 let current_player = state.current_player();
//
//                 if current_player == 0 {
//                     let action = *state.allowed_actions().choose(&mut rng).unwrap();
//
//                     state = state.take_action_by_action_index(action);
//                 } else {
//                     let action = policy(&state, &mut batch_processor, &mut rng).await;
//
//                     state = state.take_action_by_action_index(action);
//                 }
//             }
//
//             return state.finish_state();
//         });
//
//         handles.push(game);
//     }
//
//     let mut x_wins = 0;
//     let mut o_wins = 0;
//     let mut draws = 0;
//
//     for handle in handles {
//         let finish_state = handle.await.unwrap();
//
//         match finish_state {
//             TicTacToeFinishState::Xwon => {
//                 x_wins += 1;
//             }
//             TicTacToeFinishState::Owon => {
//                 o_wins += 1;
//             }
//             TicTacToeFinishState::Draw => {
//                 draws += 1;
//             }
//             _ => {
//                 panic!("Invalid finish state");
//             }
//         }
//     }
//
//     tensorboard_sender.add_scalar("epoch/x_win", x_wins as f32, num_epoch);
//     tensorboard_sender.add_scalar("epoch/o_win", o_wins as f32, num_epoch);
//     tensorboard_sender.add_scalar("epoch/draws", draws as f32, num_epoch);
//     tensorboard_sender.add_scalar("epoch/total", (x_wins as f32 / (o_wins as f32 + draws as f32)), num_epoch);
//
//     tensorboard_sender.add_scalar("training_steps/x_win", x_wins as f32, num_learning_steps);
//     tensorboard_sender.add_scalar("training_steps/o_win", o_wins as f32, num_learning_steps);
//     tensorboard_sender.add_scalar("training_steps/draws", draws as f32, num_learning_steps);
//     tensorboard_sender.add_scalar("training_steps/total", (x_wins as f32 / (o_wins as f32 + draws as f32)), num_learning_steps);
//
//     println!("{num_epoch} X wins: {}, O wins: {}, Draws: {}", x_wins, o_wins, draws);
// }
//
// struct AsyncTTTEvaluatorFn {
//
// }
// #[async_trait]
// impl EvaluationFn<9, 2, 9> for AsyncTTTEvaluatorFn {
//     async fn evaluate(
//         &self,
//         batch_processor: BatchProcessorSender<[i64; 9], ([f32; 2], [f32; 9])>,
//         num_epoch: usize,
//         num_learning_steps: usize,
//         alpha_zero_train_options: &AlphaZeroTrainOptions,
//         tensorboard_sender: Arc<dyn TensorboardSender>
//     ) {
//         async_evaluate_ttt(batch_processor, num_epoch, num_learning_steps, tensorboard_sender).await;
//     }
// }
//
//
// pub async fn experiment_3_ttt() {
//     let device = Device::Cuda(0);
//
//     let alpha_zero_train_options = AlphaZeroTrainOptions {
//         run_start: 0,
//         batch_nn_max_delay: Duration::from_millis(1),
//         batch_nn_buffer_size: 512,
//         games_per_epoch: 2000,
//         max_concurrent_games: 100,
//         // mcts_iterations: 100,
//         mcts_iterations_schedule: |epoch| {
//             100
//         },
//         mcts_iterations_schedule_name: "100",
//         dirichlet_alpha: 1.0,
//         dirichlet_epsilon: 0.25,
//         puct_exploration_constant: 3.5,
//         learning_rate: 0.001,
//         temperature: Some(1.25),
//         minibatch_size: 128,
//         checkpoint_every_n_epochs: 5,
//
//         min_or_max_value: 1.0,
//         avg_loss_for_tensorboard: 16,
//         device,
//         par_iterations: 1,
//         virtual_loss: 1.0,
//         probability_of_keeping_experience: 1.0,
//         number_of_batch_receivers: 1,
//         value_target: ValueTarget::Default,
//         node_arena_capacity: 1000,
//         state_arena_capacity: 1000,
//         cache_size_batch_processor: 300,
//         mcts_workers: 0,
//         max_concurrent_games_in_evaluation: 0,
//         load_from_checkpoint: None,
//         epoch_start: None,
//         evaluation_every_n_epochs: 1,
//         do_eval: true,
//     };
//
//     let network = TicTacToeNetwork::new(
//         device,
//         alpha_zero_train_options.learning_rate as f64
//     );
//
//     train_alpha_zero::<TicTacToeState, f32, usize, 9, 2, 9>(
//         |rng| TicTacToeState::new(),
//         Box::new(network),
//         alpha_zero_train_options,
//         Arc::new(AsyncTTTEvaluatorFn {})
//     )
//         .await;
// }