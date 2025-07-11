use std::cmp::min;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::prelude::SmallRng;
use rand::SeedableRng;
use tokio::task::JoinSet;
use rs_unsafe_arena::unsafe_arena::UnsafeArena;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand_distr::num_traits::ops::mul_add;
use crate::alpha_zero::alpha_zero_train::AzEvaluator;
use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::batch_processor::batch_processor::create_batch_processor;
use crate::alpha_zero::batch_processor::cached_network_batch_processor::CachedNetworkBatchProcessorSender;
use crate::alpha_zero::batch_processor::network_batch_processor::NetworkBatchProcessor;
use crate::alpha_zero::mcts::node::AzNode;
use crate::alpha_zero::net::experience_replay_buffer3::ExperienceReplayBuffer3;
use crate::alpha_zero::net::network::Network;
use crate::alpha_zero::net_trainer::net_trainer::NetworkTrainer;
use crate::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
use crate::alpha_zero::train::glob::GlobalStats;
use crate::alpha_zero::train::self_play::self_play;
use crate::env::env_state::AzEnvState;

struct ArenaContainer<A1, A2> {
    pub arenas: Vec<(A1, A2)>,
}

impl <A1, A2> ArenaContainer<A1, A2> {

}

unsafe impl <A1, A2> Sync for ArenaContainer<A1, A2> {}

pub async unsafe fn alpha_zero_main_train_loop<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions> + 'static,
    TActionType: Copy + Send + Debug + 'static,

    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>(
    initial_network: Box<dyn Network>,

    create_root_state_fn: fn(&mut SmallRng) -> TGameState,
    alpha_zero_options: AlphaZeroTrainOptions,
    evaluation_fn_clone: Arc<dyn AzEvaluator<{ TStateDim }, { TNumberOfPlayers }, { TNumberOfActions }>>,
    tensorboard_sender: Arc<dyn TensorboardSender>,
) {
    let multi_progress = Arc::new(MultiProgress::new());

    let run_start_time = std::time::Instant::now();

    let erb = Arc::new(
        ExperienceReplayBuffer3::new(
            TStateDim,
            TNumberOfPlayers,
            TNumberOfActions
        )
    );

    let mut network_trainer: NetworkTrainer<TActionType, TGameState, TStateDim, TNumberOfPlayers, TNumberOfActions> = NetworkTrainer::new(
        erb.clone()
    );

    let mut epoch_i = alpha_zero_options.epoch_start.unwrap_or(0);
    let mut games_overall_played = 0;


    // Das Netzwerk, auf dem kontinuierlich trainiert wird
    let mut current_network: Arc<dyn Network> = Arc::from(initial_network);

    let mut glob_stats = Arc::new(GlobalStats {
        number_of_experiences_added_in_epoch: AtomicUsize::new(0),

        number_of_cache_misses: AtomicUsize::new(0),
        number_of_cache_hits: AtomicUsize::new(0),
        time_of_batch_processor: AtomicUsize::new(0),
        number_of_batch_processor_hits: AtomicUsize::new(0),
        number_of_turns_in_epoch: AtomicUsize::new(0),
        number_of_batch_processed_entries: AtomicUsize::new(0),
    });

    loop {
        let tensorboard_sender = tensorboard_sender.clone();
        let epoch_start_time = std::time::Instant::now();

        let glob_stats = glob_stats.clone();
        // Batch-Prozessoren für diese Epoche erstellen
        let (batch_processor, mut batch_processor_tasks) = create_batch_processor(
            alpha_zero_options.batch_nn_buffer_size,
            alpha_zero_options.batch_nn_max_delay,
            (0..alpha_zero_options.number_of_batch_receivers)
                .map(|i| NetworkBatchProcessor::new(
                    Arc::from(current_network.clone_network(i)),
                    alpha_zero_options.batch_nn_buffer_size,
                ))
                .collect(),
            alpha_zero_options.number_of_batch_receivers,
            glob_stats.clone()
        );

        let iterations_this_epoch = alpha_zero_options.mcts_iterations;

        println!("Starting epoch {} with {} iterations", epoch_i, iterations_this_epoch);

        // Concurrent Tasks limitieren, wg. Speicher
        let games_in_epoch_played = Arc::new(AtomicUsize::new(0));
        let games_in_epoch_started = Arc::new(AtomicUsize::new(0));

        let mut currently_running_games = JoinSet::new();

        for i in 0..alpha_zero_options.max_concurrent_games {
            let mut node_arena: UnsafeArena<
                AzNode<TGameState, TActionType, TNumberOfPlayers, TNumberOfActions>
            > = UnsafeArena::new(
                alpha_zero_options.node_arena_capacity
            );

            let mut state_arena: UnsafeArena<
                TGameState
            > = UnsafeArena::new(
                alpha_zero_options.state_arena_capacity
            );

            let mut batch_processor = CachedNetworkBatchProcessorSender::<
                TGameState,
                TActionType,
                TStateDim,
                TNumberOfPlayers,
                TNumberOfActions
            >::new(
                batch_processor.clone(),
                alpha_zero_options.cache_size_batch_processor,
                glob_stats.clone()
            );
            let games_in_epoch_played = games_in_epoch_played.clone();
            let mut games_in_epoch_started = games_in_epoch_started.clone();

            let mut erb = erb.clone();
            let glob_stats = glob_stats.clone();

            currently_running_games.spawn(
                async move {
                    let mut rng = SmallRng::from_os_rng();

                    const BUFFER: usize = 400;

                    let mut states_buffer: Vec<[i64; TStateDim]> = Vec::with_capacity(BUFFER);
                    let mut value_targets_buffer: Vec<[f32; TNumberOfPlayers]> = Vec::with_capacity(BUFFER);
                    let mut policy_targets_buffer: Vec<[f32; TNumberOfActions]> = Vec::with_capacity(BUFFER);

                    let mut games_played_in_concurrent_worker = 0;

                    loop {
                         games_in_epoch_started
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                        self_play(
                            create_root_state_fn,
                            &mut batch_processor,
                            iterations_this_epoch,
                            alpha_zero_options,
                            &mut node_arena,
                            &mut state_arena,
                            epoch_i,
                            &mut states_buffer,
                            &mut value_targets_buffer,
                            &mut policy_targets_buffer,
                            glob_stats.clone()
                        ).await;

                        games_played_in_concurrent_worker += 1;

                        erb
                            .append_slice(
                                states_buffer.as_slice(),
                                value_targets_buffer.as_slice(),
                                policy_targets_buffer.as_slice(),
                            );

                        states_buffer.clear();
                        value_targets_buffer.clear();
                        policy_targets_buffer.clear();

                        let games_in_epoch_started = games_in_epoch_started
                            .load(std::sync::atomic::Ordering::Relaxed);

                        if (games_in_epoch_started + 1) >= alpha_zero_options.games_per_epoch {
                            return games_played_in_concurrent_worker;
                        }
                    }
                }
            );
        }

        let time_keeper_task = {
            let games_in_epoch_played = games_in_epoch_played.clone();
            let tensorboard_sender = tensorboard_sender.clone();
            let multi_progress = multi_progress.clone();

            tokio::task::spawn(async move {
                let pb = multi_progress
                    .add(ProgressBar::new(alpha_zero_options.games_per_epoch as u64));

                let pb_info = multi_progress
                    .add(ProgressBar::new_spinner());

                pb
                    .set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));

                let mut task_start_time = std::time::Instant::now();
                let mut games_played_in_epoch_at_start_time = games_in_epoch_played
                    .load(Ordering::Relaxed);
                let number_of_processed_entries_at_start_time = glob_stats.number_of_batch_processed_entries.load(Ordering::Relaxed);

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                    let games_in_epoch_played = games_in_epoch_played
                        .load(Ordering::Relaxed);

                    let new = min(games_in_epoch_played, alpha_zero_options.games_per_epoch);

                    pb.set_position(new as u64);

                    let duration_to_tast_start_time = task_start_time
                        .elapsed()
                        .as_secs_f32();

                    let games_played_since_task_start_time = games_in_epoch_played
                        - games_played_in_epoch_at_start_time;

                    let games_per_second = games_played_since_task_start_time as f32
                        / duration_to_tast_start_time;

                    tensorboard_sender.scalar(
                        "time/games_played_in_epoch",
                        games_in_epoch_played as f32,
                        run_start_time.elapsed().as_secs() as i64,
                    );

                    tensorboard_sender.scalar(
                        "time/games_per_second",
                        games_per_second,
                        run_start_time.elapsed().as_secs() as i64,
                    );

                    let avg_batch_processor_time = glob_stats.time_of_batch_processor.load(Ordering::Relaxed) as f32
                        / glob_stats.number_of_batch_processor_hits.load(Ordering::Relaxed) as f32;

                    glob_stats.time_of_batch_processor.store(0, Ordering::Relaxed);
                    glob_stats.number_of_batch_processor_hits.store(0, Ordering::Relaxed);

                    let number_of_processed_entries_per_second = (glob_stats.number_of_batch_processed_entries.load(Ordering::Relaxed) - number_of_processed_entries_at_start_time) as f32
                        / duration_to_tast_start_time;

                    let number_of_processed_entries = glob_stats.number_of_batch_processed_entries.load(Ordering::Relaxed);

                    let turns_in_epoch = glob_stats.number_of_turns_in_epoch.load(Ordering::Relaxed);

                    let avg_hits = glob_stats.number_of_cache_hits.load(Ordering::Relaxed) as f32
                        / (glob_stats.number_of_cache_hits.load(Ordering::Relaxed) + glob_stats.number_of_cache_misses.load(Ordering::Relaxed)) as f32;

                    glob_stats.number_of_cache_hits.store(0, Ordering::Relaxed);
                    glob_stats.number_of_cache_misses.store(0, Ordering::Relaxed);

                    pb_info.set_message(format!(
                        "Turns: {}, Avg. BP Time: {:.2}ms, Avg. Cache Hits: {:.2}, Entries/s: {:.2}, Entries: {}",
                        turns_in_epoch,
                        avg_batch_processor_time,
                        avg_hits,
                        number_of_processed_entries_per_second,
                        number_of_processed_entries
                    ));
                }
            })
        };

        let mut games_played_in_epoch = 0;
        loop {
            let handle = currently_running_games
                .join_next()
                .await;

            if handle.is_none() {
                // Alle Worker sind fertig
                break;
            }

            let (games_played) = handle
                .unwrap()
                .unwrap();

            games_played_in_epoch += games_played;
            games_overall_played += games_played;
        }

        time_keeper_task.abort();

        let cloned_network = current_network.clone_network(0);

        let evaluation_fn_clone = evaluation_fn_clone.clone();

        let _tensorboard_sender = tensorboard_sender.clone();
        let evaluation_task = tokio::task::spawn(async move {
            if epoch_i % alpha_zero_options.evaluation_every_n_epochs == 0 && !alpha_zero_options.skip_evaluation {
                evaluation_fn_clone
                    .evaluate(
                        batch_processor,
                        epoch_i - 1,
                        0,
                        &alpha_zero_options,
                        _tensorboard_sender.clone()
                    )
                    .await;
            } else {
                println!("Skipping eval");
            }
        });

        let train_task = tokio::task::spawn_blocking(move || {
            let result = Arc::from(
                network_trainer
                    .train_network(
                        cloned_network,
                        &alpha_zero_options,
                        epoch_i
                    )
            );

            (result, network_trainer)
        });

        // Dann trainieren.
        (current_network, network_trainer) = train_task.await.unwrap();
        evaluation_task.await.unwrap();

        // Alle Batch-Prozessoren beenden, damit diese
        // im nächsten Durchlauf wieder neu erstellt werden.
        // (mit neuem Netzwerk)
        batch_processor_tasks.abort_all();

        let epoch_duration = epoch_start_time
            .elapsed()
            .as_secs_f32();

        let tensorboard_sender = tensorboard_sender.clone();

        tensorboard_sender.scalar(
            "epoch/epoch_duration_seconds",
            epoch_duration,
            epoch_i as i64,
        );

        tensorboard_sender.scalar(
            "epoch/games_per_seconds_in_epoch",
            games_played_in_epoch as f32 / epoch_duration,
            epoch_i as i64,
        );

        epoch_i += 1;
    }
}