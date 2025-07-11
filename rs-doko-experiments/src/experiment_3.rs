// use std::cell::RefCell;
// use std::collections::HashMap;
// use crate::csv_writer_thread::CSVWriterThread;
// use async_trait::async_trait;
// use futures::{stream, StreamExt};
// use rand::prelude::{IndexedRandom, SliceRandom, SmallRng};
// use rand::SeedableRng;
// use rs_doko::action::action::DoAction;
// use rs_doko::observation::observation::DoObservation;
// use rs_doko::state::state::DoState;
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluate::{evaluate_single, evaluate_single_only_policy_head};
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_train::{train_alpha_zero, EvaluationFn};
// use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
// use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
// use rs_doko_alpha_zero::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
// use rs_doko_alpha_zero::alpha_zero::mcts::node::AzNode;
// use rs_doko_alpha_zero::env::envs::doko::doko_impl::{index2action, DokoState};
// use rs_doko_alpha_zero::env::envs::doko::doko_net2::{DokoNetSettings, DokoNetwork2};
// use rs_doko_evaluator::doko_async::evaluate_single_game::doko_async_evaluate_single_game;
// use rs_doko_evaluator::doko_async::policy::policy::EvAsyncDokoPolicy;
// use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use rs_unsafe_arena::unsafe_arena::UnsafeArena;
// use std::rc::Rc;
// use std::sync::Arc;
// use std::time::Duration;
// use async_channel::{RecvError, Sender};
// use tch::nn;
// use tensorboard_rs::summary_writer::SummaryWriter;
// use tokio::sync::oneshot;
// use tokio::sync::oneshot::Sender as OneshotSender;
// use tokio::task;
// use tokio::task::{JoinHandle, JoinSet};
// use tokio_metrics::TaskMonitor;
// use rs_doko::action::allowed_actions::{calculate_allowed_actions_in_normal_game, random_action};
// use rs_doko_alpha_zero::env::env_state::AzEnvState;
//
// type DoCachedNetworkBatchProcessorSender = CachedNetworkBatchProcessorSender<DokoState, DoAction, i64, 114, 4, 26>;
//
//
// #[derive(Default)]
// struct Aggregate {
//     sum: i32,
//     count: usize,
// }
//
// impl Aggregate {
//     fn add(&mut self, val: i32) {
//         self.sum += val;
//         self.count += 1;
//     }
//
//     fn average(&self) -> f32 {
//         if self.count == 0 {
//             0.0
//         } else {
//             self.sum as f32 / self.count as f32
//         }
//     }
// }
//
// fn calculate_average_points(
//     results: &[(i32, usize, f32, usize, f32)]
// ) -> HashMap<(usize, usize), f32> {
//     let mut aggregator: HashMap<(usize, usize), (i32, usize)> = HashMap::new();
//
//     for &(points, own_iters, _own_uct, enemy_iters, _enemy_uct) in results {
//         aggregator
//             .entry((own_iters, enemy_iters))
//             .and_modify(|(sum, count)| {
//                 *sum += points;
//                 *count += 1;
//             })
//             .or_insert((points, 1));
//     }
//
//     aggregator
//         .into_iter()
//         .map(|((own_iters, enemy_iters), (sum, count))| {
//             let avg = sum as f32 / count as f32;
//             ((own_iters, enemy_iters), avg)
//         })
//         .collect()
// }
//
// pub fn repeat_policy_remaining_2<'a>(
//     number_of_mcts_players: usize,
//
//     policy: &'a dyn EvAsyncDokoPolicy,
//     remaining_policy: &'a dyn EvAsyncDokoPolicy,
// ) -> [&'a dyn EvAsyncDokoPolicy; 4] {
//     match number_of_mcts_players {
//         1 => [policy, remaining_policy, remaining_policy, remaining_policy],
//         2 => [policy, policy, remaining_policy, remaining_policy],
//         3 => [policy, policy, policy, remaining_policy],
//         _ => panic!("Invalid number of MCTS players: must be between 1 and 4"),
//     }
// }
//
// #[derive(Clone)]
// struct EnemyStrategy {
//     mcts_iterations: usize,
//     mcts_uct_param: f32,
// }
//
//
// fn create_mcts_workers(num_workers: usize) -> (Sender<(DoState, f32, usize, tokio::sync::oneshot::Sender<DoAction>)>, Vec<JoinHandle<
//     Result<(), tokio::task::JoinError>
// >>) {
//
//     let (sender, receiver) = async_channel::unbounded();
//
//     let mut receiver_tasks = Vec::new();
//
//     for _ in 0..num_workers {
//         let receiver = receiver.clone();
//
//         let handle = tokio::task::spawn(async move {
//             let mut mcts: CachedMCTS<McDokoEnvState, DoAction, 4, 26>
//                 = CachedMCTS::new(12000, 12000);
//             let mut rng = SmallRng::from_os_rng();
//
//             loop {
//                 let result: (DoState, f32, usize, OneshotSender<DoAction>)
//                     = receiver.recv().await.unwrap();
//                 let state = result.0;
//                 let uct_exploration_constant = result.1;
//                 let iterations = result.2;
//
//                 mcts = tokio::task::spawn_blocking(move || {
//                     let moves = unsafe {
//                         mcts.monte_carlo_tree_search(
//                             McDokoEnvState::new(state, None),
//                             uct_exploration_constant as f64,
//                             iterations,
//                             &mut SmallRng::from_os_rng(),
//                         )
//                     };
//
//                     let best_move = moves.iter().max_by_key(|x| x.visits).unwrap();
//                     result.3.send(best_move.action).unwrap();
//
//                     mcts
//                 }).await.unwrap();
//
//             }});
//
//         receiver_tasks.push(handle);
//     }
//
//     (
//         sender,
//         receiver_tasks
//     )
// }
//
// #[async_trait]
// impl EvAsyncDokoPolicy for EnemyStrategy {
//     async fn policy(&self,
//                     state: &DoState,
//                     observation: &DoObservation,
//                     rng: &mut rand::rngs::SmallRng,
//                     mcts_sender: Sender<(DoState, f32, usize, OneshotSender<DoAction>)>,
//                     batch_processor: &mut CachedNetworkBatchProcessorSender<DokoState, DoAction, i64, 114, 4, 26>,
//                     nodes_arena: &mut UnsafeArena<AzNode<DokoState, DoAction, i64, 4, 26>>,
//                     states_arena: &mut UnsafeArena<DokoState>) -> DoAction {
//         if self.mcts_iterations == 0 {
//             let allowed_actions = observation.allowed_actions_current_player;
//
//             return random_action(allowed_actions, rng);
//         }
//
//         let mcts_iterations = self.mcts_iterations;
//         let mcts_uct_param = self.mcts_uct_param;
//
//         let (sender, receiver) = oneshot::channel();
//
//         mcts_sender.send((
//             state.clone(),
//             mcts_uct_param,
//             mcts_iterations,
//             sender,
//         )).await
//             .unwrap();
//
//         let action = receiver.await.unwrap();
//
//         action
//     }
// }
//
// #[derive(Clone)]
// struct OwnStrategy {
//     mcts_iterations: usize,
//     mcts_uct_param: f32,
//
//     alpha_zero_train_options: AlphaZeroTrainOptions,
// }
//
// #[async_trait]
// impl EvAsyncDokoPolicy for OwnStrategy {
//     async fn policy(&self, state: &DoState, observation: &DoObservation, rng: &mut rand::rngs::SmallRng, mcts_sender: Sender<(DoState, f32, usize, OneshotSender<DoAction>)>, batch_processor: &mut CachedNetworkBatchProcessorSender<DokoState, DoAction, i64, 114, 4, 26>, nodes_arena: &mut UnsafeArena<AzNode<DokoState, DoAction, i64, 4, 26>>, states_arena: &mut UnsafeArena<DokoState>) -> DoAction {
//         let iterations = self.mcts_iterations;
//
//         if iterations == 0 {
//             return index2action(evaluate_single_only_policy_head(
//                 &DokoState::new(state.clone(), None),
//                 batch_processor,
//             ).await);
//         }
//
//         index2action(evaluate_single(
//             &DokoState::new(state.clone(), None),
//             AlphaZeroEvaluationOptions {
//                 iterations,
//                 dirichlet_alpha: self.alpha_zero_train_options.dirichlet_alpha,
//                 dirichlet_epsilon: self.alpha_zero_train_options.dirichlet_epsilon,
//                 puct_exploration_constant: self.alpha_zero_train_options.puct_exploration_constant,
//                 min_or_max_value: self.alpha_zero_train_options.min_or_max_value,
//                 par_iterations: 3,
//                 virtual_loss: 0.3,
//             },
//             batch_processor,
//             nodes_arena,
//             states_arena,
//             rng,
//         ).await)
//     }
// }
//
// struct AsyncDokoEvaluatorFn;
//
// #[async_trait]
// impl EvaluationFn<i64, 114, 4, 26> for AsyncDokoEvaluatorFn {
//     async fn evaluate(
//         &self,
//         batch_processor: BatchProcessorSender<[i64; 114], ([f32; 4], [f32; 26])>,
//         num_epoch: usize,
//         num_learning_steps: usize,
//         tensorboard_log_dir: String,
//         alpha_zero_train_options: &AlphaZeroTrainOptions,
//     ) {
//         // Besser als Parameter übergeben, aber wie??
//
//         let mut summary_writer = SummaryWriter::new(alpha_zero_train_options.tensorboard_log_dir(DokoState::GAME_NAME) + "/evaluation");
//
//         let mut own_strategies = vec![
//             OwnStrategy {
//                 mcts_iterations: 0,
//                 mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
//                 alpha_zero_train_options: alpha_zero_train_options.clone(),
//             },
//             // OwnStrategy {
//             //     mcts_iterations: 10,
//             //     mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
//             //     alpha_zero_train_options: alpha_zero_train_options.clone(),
//             // },
//             // OwnStrategy {
//             //     mcts_iterations: 50,
//             //     mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
//             //     alpha_zero_train_options: alpha_zero_train_options.clone(),
//             // },
//             OwnStrategy {
//                 mcts_iterations: 300,
//                 mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
//                 alpha_zero_train_options: alpha_zero_train_options.clone(),
//             },
//         ];
//         let mut enemy_strategy = vec![
//             EnemyStrategy {
//                 mcts_iterations: 0,
//                 mcts_uct_param: 0f32
//             },
//             EnemyStrategy {
//                 mcts_iterations: 10,
//                 mcts_uct_param: 5f32.sqrt()
//             },
//             EnemyStrategy {
//                 mcts_iterations: 150,
//                 mcts_uct_param: 5f32.sqrt()
//             },
//             EnemyStrategy {
//                 mcts_iterations: 300,
//                 mcts_uct_param: 5f32.sqrt()
//             },
//             EnemyStrategy {
//                 mcts_iterations: 1000,
//                 mcts_uct_param: 5f32.sqrt()
//             },
//         ];
//         let number_of_az_player = [3, 2, 1];
//         let number_of_games = 1000;
//
//         let mut matchups: Vec<(i32, OwnStrategy, EnemyStrategy, usize, i32)> = vec![];
//
//         let mut game_id = 0;
//         // Liste aller Spiele vorbereiten
//         for own_strategy in own_strategies.iter() {
//             for enemy_strategy in enemy_strategy.iter() {
//                 for number_of_az_player in number_of_az_player.iter() {
//                     for i in 0..number_of_games {
//                         matchups.push((
//                             game_id,
//                             own_strategy.clone(),
//                             enemy_strategy.clone(),
//                             *number_of_az_player as usize,
//                             i
//                         ));
//                         game_id += 1;
//                     }
//                 }
//             }
//         }
//
//
//         let mut csv_writer = CSVWriterThread::new(
//             format!("{}.csv", alpha_zero_train_options.description("Doko")).into(),
//             &[
//                 "az_global_iteration",
//                 "az_train_options_minibatch_size",
//                 "az_train_options_games_per_epoch",
//                 "az_train_options_games_mcts_iterations_schedule",
//                 "az_train_options_games_mcts_dirichlet_alpha",
//                 "az_train_options_games_mcts_dirichlet_epsilon",
//                 "az_train_options_games_mcts_puct_exploration_constant",
//                 "az_train_options_games_mcts_puct_learning_rate",
//                 "az_train_options_games_mcts_puct_temperature",
//                 "az_train_options_games_mcts_min_or_max_value",
//                 "az_number_of_iterations",
//                 "az_uct_param",
//                 "mcts_uct_param",
//                 "mcts_iterations",
//                 "number_of_mcts_player",
//                 "game_number",
//                 "points_player_0",
//                 "points_player_1",
//                 "points_player_2",
//                 "points_player_3",
//                 "total_execution_time_player_0",
//                 "total_execution_time_player_1",
//                 "total_execution_time_player_2",
//                 "total_execution_time_player_3",
//                 "avg_execution_time_player_0",
//                 "avg_execution_time_player_1",
//                 "avg_execution_time_player_2",
//                 "avg_execution_time_player_3",
//                 "number_of_actions"
//             ],
//             Some(matchups.len() as u64),
//             if num_epoch == 0 {false } else {true}
//         );
//
//         matchups.shuffle(&mut SmallRng::from_os_rng());
//
//         let (mcts_sender, mcts_receivers) = create_mcts_workers(
//             alpha_zero_train_options.mcts_workers
//         );
//
//         let (matchup_sender, matchup_receiver) = async_channel::unbounded();
//
//         let mut join_set = JoinSet::new();
//
//         for i in 0..alpha_zero_train_options.max_concurrent_games_in_evaluation {
//             let alpha_zero_train_options = alpha_zero_train_options.clone();
//
//             let mut node_arena = UnsafeArena::new(alpha_zero_train_options.node_arena_capacity);
//             let mut state_arena = UnsafeArena::new(alpha_zero_train_options.state_arena_capacity);
//
//             let mut batch_processor = CachedNetworkBatchProcessorSender::new(
//                 batch_processor.clone(),
//                 alpha_zero_train_options.cache_size_batch_processor
//             );
//
//             let csv_writer = csv_writer.clone();
//             let mcts_sender = mcts_sender.clone();
//
//             let matchup_receiver = matchup_receiver
//                 .clone();
//
//             let task_handle = join_set.spawn(async move {
//                 let mut results = Vec::new();
//
//                 loop {
//                     let recv_result: Result<(i32, OwnStrategy, EnemyStrategy, usize, i32), RecvError> = matchup_receiver
//                         .recv()
//                         .await;
//
//                     if let Ok(matchup) = recv_result {
//                         let game_id = matchup.0;
//                         let alpha_zero_policy = matchup.1;
//                         let mcts_policy = matchup.2;
//
//                         let cloned_alpha_zero_policy = alpha_zero_policy.clone();
//                         let cloned_mcts_policy = mcts_policy.clone();
//
//                         let policies = repeat_policy_remaining_2(
//                             matchup.3,
//                             &(cloned_alpha_zero_policy),
//                             &(cloned_mcts_policy),
//                         );
//
//                         let mut rng = SmallRng::from_os_rng();
//                         let mut game_creation_rng = SmallRng::seed_from_u64(matchup.4 as u64 * 1000);
//
//                         let result = doko_async_evaluate_single_game(
//                             policies,
//                             &mut game_creation_rng,
//                             &mut rng,
//                             mcts_sender.clone(),
//                             &mut batch_processor,
//                             &mut node_arena,
//                             &mut state_arena,
//                         ).await;
//
//                         csv_writer.write_row(vec![
//                             // az_global_iteration
//                             num_epoch.to_string(),
//
//                             // az_train_options_minibatch_size
//                             alpha_zero_train_options.minibatch_size.to_string(),
//                             // az_train_options_games_per_epoch
//                             alpha_zero_train_options.games_per_epoch.to_string(),
//                             // az_train_options_games_mcts_iterations_schedule
//                             (alpha_zero_train_options.mcts_iterations_schedule)(num_epoch).to_string(),
//                             // az_train_options_games_mcts_dirichlet_alpha
//                             alpha_zero_train_options.dirichlet_alpha.to_string(),
//                             // az_train_options_games_mcts_dirichlet_epsilon
//                             alpha_zero_train_options.dirichlet_epsilon.to_string(),
//                             // az_train_options_games_mcts_puct_exploration_constant
//                             alpha_zero_train_options.puct_exploration_constant.to_string(),
//                             // az_train_options_games_mcts_learning_rate
//                             alpha_zero_train_options.learning_rate.to_string(),
//                             // az_train_options_games_mcts_temperature
//                             alpha_zero_train_options.temperature.unwrap_or(-1.0).to_string(),
//                             // az_train_options_games_mcts_min_or_max_value
//                             alpha_zero_train_options.min_or_max_value.to_string(),
//
//                             // az_number_of_iterations
//                             alpha_zero_policy.mcts_iterations.to_string(),
//                             // az_uct_param
//                             alpha_zero_policy.mcts_uct_param.to_string(),
//
//                             // mcts_uct_param
//                             mcts_policy.mcts_uct_param.to_string(),
//                             // mcts_iterations
//                             mcts_policy.mcts_iterations.to_string(),
//
//                             // number_of_mcts_player
//                             matchup.3.to_string(),
//
//                             // game_number
//                             game_id.to_string(),
//
//                             // points
//                             result.points[0].to_string(),
//                             result.points[1].to_string(),
//                             result.points[2].to_string(),
//                             result.points[3].to_string(),
//
//                             // total_execution_time
//                             result.total_execution_time[0].to_string(),
//                             result.total_execution_time[1].to_string(),
//                             result.total_execution_time[2].to_string(),
//                             result.total_execution_time[3].to_string(),
//
//                             // avg_execution_time
//                             result.avg_execution_time[0].to_string(),
//                             result.avg_execution_time[1].to_string(),
//                             result.avg_execution_time[2].to_string(),
//                             result.avg_execution_time[3].to_string(),
//
//                             // number_of_actions
//                             result.number_of_actions.to_string()
//                         ]);
//
//                         results.push((
//                             result.points[0],
//
//                             alpha_zero_policy.mcts_iterations,
//                             alpha_zero_policy.mcts_uct_param,
//
//                             mcts_policy.mcts_iterations,
//                             mcts_policy.mcts_uct_param,
//                         ));
//                     } else {
//                         break;
//                     }
//                 }
//
//                 return results;
//             });
//         }
//
//         for matchup in matchups {
//             matchup_sender
//                 .send(matchup)
//                 .await
//                 .unwrap();
//         }
//         matchup_sender.close();
//
//         let results = join_set
//             .join_all()
//             .await;
//
//         let results = results
//             .into_iter()
//             .flatten()
//             .collect::<Vec<_>>();
//
//         csv_writer.finish();
//
//         let averages = calculate_average_points(&results);
//
//         for ((own_iters, enemy_iters), avg_points) in averages {
//             let tag = format!("epoch/avg_own_{}_enemy_{}", own_iters, enemy_iters);
//             summary_writer.add_scalar(&tag, avg_points, num_epoch);
//         }
//
//         summary_writer.flush();
//
//         mcts_receivers.into_iter().for_each(|handle| {
//             handle.abort();
//         });
//     }
// }
//
//
// pub async fn experiment3_doko() {
//     tch::set_num_threads(1);
//     tch::set_num_interop_threads(1);
//
//     println!("{}", tch::Cuda::is_available());
//
//     let device = tch::Device::Cuda(0);
//
//     let alpha_zero_options = AlphaZeroTrainOptions {
//         run_start: std::time::SystemTime::now()
//             .duration_since(std::time::UNIX_EPOCH)
//             .unwrap()
//             .as_secs() as usize,
//
//         epoch_start: None,
//
//         batch_nn_max_delay: Duration::from_millis(1),
//         batch_nn_buffer_size: 350,
//         games_per_epoch: 10000,
//         max_concurrent_games: 700,
//
//         mcts_iterations_schedule: |epoch| {
//             5
//         },
//         mcts_iterations_schedule_name: "5",
//
//         dirichlet_alpha: 1.25,
//         dirichlet_epsilon: 0.25,
//         puct_exploration_constant: 4.0f32,
//         temperature: Some(1.25),
//
//         checkpoint_every_n_epochs: 1,
//
//         min_or_max_value: 15.0,
//         avg_loss_for_tensorboard: 64,
//         device,
//         probability_of_keeping_experience: 1.0,
//
//         virtual_loss: 1.0f32,
//         par_iterations: 3,
//         number_of_batch_receivers: 2,
//         value_target: ValueTarget::Avg,
//
//         node_arena_capacity: 4000,
//         state_arena_capacity: 4000,
//         cache_size_batch_processor: 2500,
//         mcts_workers: 14,
//         max_concurrent_games_in_evaluation: 7500,
//
//         evaluation_every_n_epochs: 100,
//         load_from_checkpoint: None,
//         do_eval: false
//     };
//
//     let mut settings = DokoNetSettings::default();
//
//     settings.shared_layer_sizes = vec![
//         4096,
//         4096,
//         4096,
//     ];
//
//     settings.value_head_sizes = vec![
//         3192,
//         3192
//     ];
//
//     settings.policy_head_sizes = vec![
//         3192,
//         3192
//     ];
//
//     let mut network = DokoNetwork2::new(
//         settings,
//         device,
//         alpha_zero_options.learning_rate as f64
//     );
//
//     // network.load("./checkpoints/doko_1739975504_128_40000_300_1.25_0.25_4_0.002_1.25_1_3_1_Avg/epoch_0.safetensors");
//
//     train_alpha_zero::<DokoState, i64, DoAction, 114, 4, 26>(
//         |rng| {
//             DokoState::new(DoState::new_game(rng), None)
//         },
//         Box::new(network),
//         alpha_zero_options,
//         Arc::new(AsyncDokoEvaluatorFn {}),
//     ).await;
// }