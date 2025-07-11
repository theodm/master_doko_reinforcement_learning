// use std::fmt::Debug;
// use rand_distr::num_traits::Zero;
// use tch::Device;
// use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
// use crate::alpha_zero::mcts::node::value_for_player_full;
// use crate::async_mcts::async_mcts::mcts_search;
// use crate::cached_batch_processor::BatchProcessorCached;
// use crate::env_state::AzEnvState;
// use rs_unsafe_arena::unsafe_arena::UnsafeArena;
// use crate::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
// use crate::env::env_state::AzEnvState;
//
// pub struct AlphaZeroEvaluationOptions {
//     pub iterations: usize,
//
//     pub dirichlet_alpha: f32,
//     pub dirichlet_epsilon: f32,
//     pub puct_exploration_constant: f32,
//
//     pub min_or_max_value: f32
// }
//
// pub async unsafe fn evaluate<
//     TGameState: AzEnvState<TActionType, TStateMemoryDataType, TNumberOfPlayers, TNumberOfActions>,
//     TStateMemoryDataType: Zero + Pod + Clone + Copy + tch::kind::Element + Serialize + DeserializeOwned  + Send + Debug,
//     TActionType: Copy + Send + Debug,
//     const TStateDim: usize,
//     const TNumberOfPlayers: usize,
//     const TNumberOfActions: usize
// >(
//     current_state: &TGameState,
//
//     alpha_zero_evaluation_options: AlphaZeroEvaluationOptions,
//
//     batch_processor: &mut CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateMemoryDataType, TStateDim, TNumberOfPlayers, TNumberOfActions>,
//     rng: &mut rand::rngs::SmallRng,
// ) -> usize {
//     let mut node_arena = UnsafeArena::new(50000);
//     let mut state_arena = UnsafeArena::new(50000);
//
//     let alpha_zero_options = AlphaZeroTrainOptions {
//         // doesn't matter
//         run_label: "",
//
//         batch_nn_max_delay: std::time::Duration::from_millis(0),
//         batch_nn_buffer_size: 0,
//         minibatch_size: 0,
//         checkpoint_every_n_epochs: 0,
//         games_per_epoch: 0,
//         max_concurrent_games: 0,
//         mcts_iterations_schedule: |_epoch| 0,
//         mcts_iterations_schedule_name: "",
//         learning_rate: 0.0,
//
//         // only these matter
//         dirichlet_alpha: alpha_zero_evaluation_options.dirichlet_alpha,
//         dirichlet_epsilon: alpha_zero_evaluation_options.dirichlet_epsilon,
//         puct_exploration_constant: alpha_zero_evaluation_options.puct_exploration_constant,
//         // Keine Temperatur!
//         temperature: None,
//         min_or_max_value: alpha_zero_evaluation_options.min_or_max_value,
//         avg_loss_for_tensorboard: 1024,
//         device: Device::Cpu,
//     };
//
//     let mcts_result = mcts_search(
//         // ToDo: Kann auch übernommen werden?
//         current_state.clone(),
//         batch_processor,
//
//         alpha_zero_evaluation_options.iterations,
//         &alpha_zero_options,
//         rng,
//
//         &mut node_arena,
//         &mut state_arena
//     )
//         .await;
//
//     mcts_result.1.print_graphviz(
//         2,
//         4f32,
//         7f32
//     );
//
//     println!("MCTS Result: {:?}", mcts_result.0);
//     println!("Value: {:?}", value_for_player_full(&mcts_result.1, 0));
//
//     return mcts_result.0;
// }