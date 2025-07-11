use std::cell::RefCell;
use std::collections::HashMap;
use async_trait::async_trait;
use rand::prelude::{IndexedRandom, SliceRandom, SmallRng};
use rand::SeedableRng;
use rs_doko::action::action::DoAction;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluate::{evaluate_single, evaluate_single_only_policy_head};
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train::{train_alpha_zero, AzEvaluator};
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::{AlphaZeroTrainOptions, ValueTarget};
use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::mcts::node::AzNode;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use async_channel::{RecvError, Sender};
use tokio::sync::oneshot;
use tokio::sync::oneshot::Sender as OneshotSender;
use tokio::task;
use tokio::task::{JoinHandle, JoinSet};
use rs_doko::action::allowed_actions::{calculate_allowed_actions_in_normal_game, random_action};
use rs_doko_alpha_zero::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
use rs_doko_alpha_zero::env::env_state::AzEnvState;
use rs_doko_evaluator::full_doko::evaluate_single_game::full_doko_evaluate_single_game;
use rs_doko_evaluator::full_doko::policy::az_policy::AZWorkers;
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSWorkers;
use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
use rs_doko_evaluator::full_doko::policy::random_policy::EvFullDokoRandomPolicy;
use rs_doko_impi::mcts_full_doko_policy::MCTSFullDokoPolicy;
use rs_full_doko::state::state::FdoState;
use crate::alpha_zero::csv_writer_thread::CSVWriterThread;

#[derive(Default)]
struct Aggregate {
    sum: i32,
    count: usize,
}

impl Aggregate {
    fn add(&mut self, val: i32) {
        self.sum += val;
        self.count += 1;
    }

    fn average(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f32 / self.count as f32
        }
    }
}


fn calculate_average_points(
    results: &[(i32, usize, f32, usize, f32)]
) -> HashMap<(usize, usize), f32> {
    let mut aggregator: HashMap<(usize, usize), (i32, usize)> = HashMap::new();

    for &(points, own_iters, _own_uct, enemy_iters, _enemy_uct) in results {
        aggregator
            .entry((own_iters, enemy_iters))
            .and_modify(|(sum, count)| {
                *sum += points;
                *count += 1;
            })
            .or_insert((points, 1));
    }

    aggregator
        .into_iter()
        .map(|((own_iters, enemy_iters), (sum, count))| {
            let avg = sum as f32 / count as f32;
            ((own_iters, enemy_iters), avg)
        })
        .collect()
}

pub fn repeat_policy_remaining_2<'a>(
    number_of_mcts_players: usize,

    policy: Arc<dyn EvFullDokoPolicy>,
    remaining_policy: Arc<dyn EvFullDokoPolicy>,
) -> [Arc<dyn EvFullDokoPolicy>; 4] {
    match number_of_mcts_players {
        1 => [policy.clone(), remaining_policy.clone(), remaining_policy.clone(), remaining_policy.clone()],
        2 => [policy.clone(), policy.clone(), remaining_policy.clone(), remaining_policy.clone()],
        3 => [policy.clone(), policy.clone(), policy.clone(), remaining_policy.clone()],
        _ => panic!("Invalid number of MCTS players: must be between 1 and 4"),
    }
}

#[derive(Clone)]
struct EnemyStrategy {
    mcts_iterations: usize,
    mcts_uct_param: f32,
}

#[derive(Clone)]
struct OwnStrategy {
    mcts_iterations: usize,
    mcts_uct_param: f32,
    alpha_zero_train_options: AlphaZeroTrainOptions,
}

pub struct AzFullDokoEvaluator {}

#[async_trait]
impl AzEvaluator<311, 4, 39> for AzFullDokoEvaluator {
    async fn evaluate(
        &self,
        batch_processor: BatchProcessorSender<[i64; 311], ([f32; 4], [f32; 39])>,
        num_epoch: usize,
        number_of_experiences: usize,
        alpha_zero_train_options: &AlphaZeroTrainOptions,
        tensorboard_sender: Arc<dyn TensorboardSender>
    ) {
        let (mcts_workers, mcts_factory) = MCTSWorkers::create_and_run(
            alpha_zero_train_options.mcts_workers,
            20000,
            20000
        );

        let (mut az_workers, az_factory) = AZWorkers::with_existing_batch_processor(
            alpha_zero_train_options.max_concurrent_games_in_evaluation,
            batch_processor,
            alpha_zero_train_options.mcts_iterations * 20,
            alpha_zero_train_options.mcts_iterations * 20,
            alpha_zero_train_options.max_concurrent_games_in_evaluation * 3000
        );

        let mut own_strategies = vec![
            OwnStrategy {
                mcts_iterations: 0,
                mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
                alpha_zero_train_options: alpha_zero_train_options.clone(),
            },
            // OwnStrategy {
            //     mcts_iterations: 10,
            //     mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
            //     alpha_zero_train_options: alpha_zero_train_options.clone(),
            // },
            // OwnStrategy {
            //     mcts_iterations: 50,
            //     mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
            //     alpha_zero_train_options: alpha_zero_train_options.clone(),
            // },
            OwnStrategy {
                mcts_iterations: alpha_zero_train_options.mcts_iterations,
                mcts_uct_param: alpha_zero_train_options.puct_exploration_constant,
                alpha_zero_train_options: alpha_zero_train_options.clone(),
            },
        ];
        let mut enemy_strategy = vec![
            EnemyStrategy {
                mcts_iterations: 0,
                mcts_uct_param: 0f32
            },
            // EnemyStrategy {
            //     mcts_iterations: 10,
            //     mcts_uct_param: 5f32.sqrt()
            // },
            // EnemyStrategy {
            //     mcts_iterations: 150,
            //     mcts_uct_param: 5f32.sqrt()
            // },
            // EnemyStrategy {
            //     mcts_iterations: 300,
            //     mcts_uct_param: 5f32.sqrt()
            // },
            EnemyStrategy {
                mcts_iterations: 1000,
                mcts_uct_param: 20f32 * 2f32.sqrt()
            },
        ];
        let number_of_az_player = [3, 2, 1];
        let number_of_games = 200;

        let mut matchups: Vec<(i32, OwnStrategy, EnemyStrategy, usize, i32)> = vec![];

        let mut game_id = 0;

        // Liste aller Spiele vorbereiten
        for own_strategy in own_strategies.iter() {
            for enemy_strategy in enemy_strategy.iter() {
                for number_of_az_player in number_of_az_player.iter() {
                    for i in 0..number_of_games {
                        matchups.push((
                            game_id,
                            own_strategy.clone(),
                            enemy_strategy.clone(),
                            *number_of_az_player as usize,
                            i
                        ));
                        game_id += 1;
                    }
                }
            }
        }

        let mut csv_writer = CSVWriterThread::new(
            format!("{}.csv", "doko_output9").into(),
            &[
                "az_global_iteration",
                "az_train_options_games_per_epoch",
                "az_train_options_games_mcts_iterations_schedule",
                "az_train_options_games_mcts_dirichlet_alpha",
                "az_train_options_games_mcts_dirichlet_epsilon",
                "az_train_options_games_mcts_puct_exploration_constant",
                "az_train_options_games_mcts_puct_temperature",
                "az_train_options_games_mcts_min_or_max_value",
                "az_number_of_iterations",
                "az_uct_param",
                "mcts_uct_param",
                "mcts_iterations",
                "number_of_mcts_player",
                "game_number",
                "points_player_0",
                "points_player_1",
                "points_player_2",
                "points_player_3",
                "total_execution_time_player_0",
                "total_execution_time_player_1",
                "total_execution_time_player_2",
                "total_execution_time_player_3",
                "avg_execution_time_player_0",
                "avg_execution_time_player_1",
                "avg_execution_time_player_2",
                "avg_execution_time_player_3",
                "number_of_actions",

                "played_game_mode",
                "lowest_announcement_re",
                "lowest_announcement_contra",

                "reservation_made_0",
                "reservation_made_1",
                "reservation_made_2",
                "reservation_made_3",

                "lowest_announcement_made_0",
                "lowest_announcement_made_1",
                "lowest_announcement_made_2",
                "lowest_announcement_made_3",

                "branching_factor"
            ],
            Some(matchups.len() as u64),
            if num_epoch == 0 { false } else { true }
        );

        matchups.shuffle(&mut SmallRng::from_os_rng());

        let (matchup_sender, matchup_receiver) = async_channel::unbounded();

        let mut join_set = JoinSet::new();

        println!("What happens?");

        for i in 0..alpha_zero_train_options.max_concurrent_games_in_evaluation {
            let alpha_zero_train_options = alpha_zero_train_options.clone();
            let csv_writer = csv_writer.clone();
            let matchup_receiver = matchup_receiver.clone();
            let mcts_factory = mcts_factory.clone();
            let mut az_factory = az_factory.clone();

            let task_handle = join_set.spawn(async move {
                let mut results = Vec::new();

                loop {
                    let recv_result: Result<(i32, OwnStrategy, EnemyStrategy, usize, i32), RecvError> = matchup_receiver
                        .recv()
                        .await;

                    if let Ok(matchup) = recv_result {
                        let game_id = matchup.0;
                        let alpha_zero_policy = matchup.1;
                        let mcts_policy = matchup.2;

                        let cloned_alpha_zero_policy = az_factory.policy(AlphaZeroEvaluationOptions {
                            iterations: alpha_zero_policy.mcts_iterations,

                            dirichlet_alpha: 1.0,
                            dirichlet_epsilon: 0.0,

                            puct_exploration_constant: alpha_zero_train_options.puct_exploration_constant,
                            min_or_max_value: alpha_zero_train_options.min_or_max_value,

                            par_iterations: 0,
                            virtual_loss: 0.0,
                        });
                        let cloned_mcts_policy: Arc<dyn EvFullDokoPolicy> = if mcts_policy.mcts_iterations != 0 {
                            Arc::new(mcts_factory.policy(
                                mcts_policy.mcts_uct_param,
                                mcts_policy.mcts_iterations,
                            ))
                        } else {
                            Arc::new(EvFullDokoRandomPolicy {})
                        };

                        let policies = repeat_policy_remaining_2(
                            matchup.3,
                            Arc::new(cloned_alpha_zero_policy),
                            cloned_mcts_policy,
                        );

                        // Rng für das einzelne Spiel
                        let mut rng = SmallRng::from_os_rng();
                        let mut game_creation_rng = SmallRng::seed_from_u64(matchup.4 as u64 * 1000);

                        // println!("Game {} start", game_id);

                        // Das Spiel tatsächlich ausführen
                        let result = full_doko_evaluate_single_game(
                            policies,
                            &mut game_creation_rng,
                            &mut rng
                        ).await;

                        csv_writer.write_row(vec![
                            // az_global_iteration
                            num_epoch.to_string(),

                            // az_train_options_games_per_epoch
                            alpha_zero_train_options.games_per_epoch.to_string(),
                            // az_train_options_games_mcts_iterations_schedule
                            alpha_zero_train_options.mcts_iterations.to_string(),
                            // az_train_options_games_mcts_dirichlet_alpha
                            alpha_zero_train_options.dirichlet_alpha.to_string(),
                            // az_train_options_games_mcts_dirichlet_epsilon
                            alpha_zero_train_options.dirichlet_epsilon.to_string(),
                            // az_train_options_games_mcts_puct_exploration_constant
                            alpha_zero_train_options.puct_exploration_constant.to_string(),
                            // az_train_options_games_mcts_temperature
                            alpha_zero_train_options.temperature.unwrap_or(-1.0).to_string(),
                            // az_train_options_games_mcts_min_or_max_value
                            alpha_zero_train_options.min_or_max_value.to_string(),

                            // az_number_of_iterations
                            alpha_zero_policy.mcts_iterations.to_string(),
                            // az_uct_param
                            alpha_zero_policy.mcts_uct_param.to_string(),

                            // mcts_uct_param
                            mcts_policy.mcts_uct_param.to_string(),
                            // mcts_iterations
                            mcts_policy.mcts_iterations.to_string(),

                            // number_of_mcts_player
                            matchup.3.to_string(),

                            // game_number
                            game_id.to_string(),

                            // points
                            result.points[0].to_string(),
                            result.points[1].to_string(),
                            result.points[2].to_string(),
                            result.points[3].to_string(),

                            // total_execution_time
                            result.total_execution_time[0].to_string(),
                            result.total_execution_time[1].to_string(),
                            result.total_execution_time[2].to_string(),
                            result.total_execution_time[3].to_string(),

                            // avg_execution_time
                            result.avg_execution_time[0].to_string(),
                            result.avg_execution_time[1].to_string(),
                            result.avg_execution_time[2].to_string(),
                            result.avg_execution_time[3].to_string(),

                            // number_of_actions
                            result.number_of_actions.to_string(),

                            result.played_game_mode.to_string(),
                            result.lowest_announcement_re.map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_contra.map(|x| x.to_string()).unwrap_or("".to_string()),

                            result.reservation_made[0].to_string(),
                            result.reservation_made[1].to_string(),
                            result.reservation_made[2].to_string(),
                            result.reservation_made[3].to_string(),

                            result.lowest_announcement_made[0].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[1].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[2].map(|x| x.to_string()).unwrap_or("".to_string()),
                            result.lowest_announcement_made[3].map(|x| x.to_string()).unwrap_or("".to_string()),

                            result.branching_factor.to_string()
                        ]);

                        results.push((
                            result.points[0],

                            alpha_zero_policy.mcts_iterations,
                            alpha_zero_policy.mcts_uct_param,

                            mcts_policy.mcts_iterations,
                            mcts_policy.mcts_uct_param,
                        ));
                    } else {
                        break;
                    }
                }

                return results;
            });
        }

        for matchup in matchups {
            matchup_sender
                .send(matchup)
                .await
                .unwrap();
        }
        matchup_sender.close();

        let results = join_set
            .join_all()
            .await;

        let results = results
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        csv_writer.finish();

        let averages = calculate_average_points(&results);

        for ((own_iters, enemy_iters), avg_points) in averages {
            let tag = format!("epoch/avg_own_{}_enemy_{}", own_iters, enemy_iters);
            tensorboard_sender.scalar(tag.as_str(), avg_points, num_epoch as i64);

            let tag = format!("experiences/avg_own_{}_enemy_{}", enemy_iters, own_iters);
            tensorboard_sender.scalar(tag.as_str(), avg_points, number_of_experiences as i64);
        }

        az_workers.abort();
        mcts_workers.abort();
    }
}

