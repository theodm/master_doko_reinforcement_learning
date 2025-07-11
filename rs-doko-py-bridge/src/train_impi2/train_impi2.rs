use std::fmt::format;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::thread::available_parallelism;
use std::time::Duration;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pyo3::PyObject;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::ParallelSlice;
use tokio::task;
use rs_doko_impi::csv_writer_thread::CSVWriterThread;
use rs_doko_impi::forward::async_tests;
use rs_doko_impi::mcts_full_doko_policy::MCTSFullDokoPolicy;
use rs_doko_impi::network::ImpiNetwork;
use rs_doko_impi::tensorboard::TensorboardSender;
use rs_doko_impi::train_impi::{reverse_pred_process, reverse_pred_process_unsync, FullDokoPolicy, MiniBufferSender};
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::gather_games::db::SledStateDb;
use crate::tensorboard::TensorboardController;
use rayon::iter::ParallelIterator;
use tokio::task::JoinHandle;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_alpha_zero::alpha_zero::net::network::Network;
use rs_doko_evaluator::full_doko::policy::az_policy::AZWorkers;
use rs_doko_impi::training::NetworkTrainer;
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};
use crate::alpha_zero::network::PyAlphaZeroNetwork;

pub async fn test_task(
    network: Box<dyn ImpiNetwork>,

    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,

    multi_progress_bar: MultiProgress,

    tensorboard_controller: Arc<dyn TensorboardSender>,
    csv_writer_thread: CSVWriterThread
) {
    let number_of_trained_examples = network
        .number_of_trained_examples();

    async_tests(
        network,
        policies.clone(),
        300,
        1,
        300,
        1,
        Duration::from_millis(1),
        2,
        multi_progress_bar.clone(),
        number_of_trained_examples as usize,

        "eval".to_string(),
        tensorboard_controller,

        csv_writer_thread
    ).await;
}

pub async fn train_impi2_c(
    chance_of_keeping_experience: f64,
    number_of_experiences_per_training_step: usize,

    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,

    tensorboard_controller: Arc<dyn TensorboardSender>,

    mut network: Box<dyn ImpiNetwork>
) {
    // let multi_progress_bar = MultiProgress::new();
    let db = Arc::new(SledStateDb::new("played_games_db").unwrap());
    //
    // let experience_bar = multi_progress_bar.add(ProgressBar::new(number_of_experiences_per_training_step as u64));
    // experience_bar.set_style(
    //     ProgressStyle::default_bar()
    //         .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})")
    //         .unwrap()
    //         .progress_chars("#>-"),
    // );
    //
    //
    // let csv_writer_thread = CSVWriterThread::new(
    //     "test.csv".into(),
    //     &[
    //         "number_of_experiences",
    //         "type",
    //         "temp",
    //         "is_consistent",
    //         "not_consistent_reason",
    //         "game_type",
    //     ],
    //     None,
    //     false
    // );
    //
    // let mut eval_task = {
    //     let network = network.clone_network();
    //     let policies = policies.clone();
    //     let multi_progress_bar = multi_progress_bar.clone();
    //     let tensorboard_controller = tensorboard_controller.clone();
    //     let csv_writer_thread = csv_writer_thread.clone();
    //
    //     task::spawn(async move {
    //         test_task(
    //             network,
    //             policies,
    //             multi_progress_bar,
    //             tensorboard_controller,
    //             csv_writer_thread
    //         ).await
    //     })
    // };
    //
    // multi_progress_bar.println("Starting training...");
    // multi_progress_bar.println(format!("Training with {} experiences per step", number_of_experiences_per_training_step));
    //
    // loop {
    //     let mut rng = SmallRng::from_os_rng();
    //     let mut experiences: Vec<([i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE], [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE])> = vec![];
    //
    //     experience_bar.reset();
    //     experience_bar.set_position(0);
    //
    //     loop {
    //         let batch_of_games = db
    //             .get_random_states(40000)
    //             .unwrap();
    //
    //         multi_progress_bar.println(format!("Got {} games", batch_of_games.len()));
    //
    //         let exp: Vec<([i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE], [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE])> = batch_of_games
    //             .par_chunks(1024)
    //             .map(|chunk| {
    //                 let mut rng = SmallRng::from_os_rng();
    //                 let mut experiences_chunk = vec![];
    //
    //                 for state in chunk {
    //                     reverse_pred_process_unsync(
    //                         state,
    //                         chance_of_keeping_experience,
    //                         &mut experiences_chunk,
    //                         &mut rng,
    //                     );
    //                 }
    //
    //                 experiences_chunk
    //             })
    //             .flatten()
    //             .collect();
    //
    //         experiences.extend(exp);
    //         experience_bar.set_position(experiences.len() as u64);
    //
    //         if experiences.len() >= number_of_experiences_per_training_step {
    //             break;
    //         }
    //     }
    //
    //
    //     let fit_task = {
    //         let multi_progress_bar = multi_progress_bar.clone();
    //         task::spawn(async move {
    //             // Spinner for the fit process
    //             let fit_spinner = multi_progress_bar.add(ProgressBar::new_spinner());
    //             fit_spinner.set_style(
    //                 ProgressStyle::default_spinner()
    //                     .template("{spinner:.green} [{elapsed_precise}] Fitting network...")
    //                     .unwrap(),
    //             );
    //             fit_spinner.enable_steady_tick(Duration::from_millis(100));
    //
    //             fit_spinner.set_message("Fitting network...");
    //
    //             let state_memory = experiences
    //                 .iter()
    //                 .map(|it| it.0)
    //                 .flatten()
    //                 .collect::<Vec<i64>>();
    //
    //             let target_memory = experiences
    //                 .iter()
    //                 .map(|it| it.1)
    //                 .flatten()
    //                 .collect::<Vec<f32>>();
    //
    //             network.fit(
    //                 state_memory,
    //                 target_memory
    //             );
    //
    //             fit_spinner.finish_with_message("Fit complete");
    //
    //             network
    //         })
    //     };
    //
    //     network = fit_task.await.unwrap();
    //     eval_task.await.unwrap();
    //
    //     {
    //         let network = network.clone_network();
    //         let policies = policies.clone();
    //         let multi_progress_bar = multi_progress_bar.clone();
    //         let tensorboard_controller = tensorboard_controller.clone();
    //         let csv_writer_thread = csv_writer_thread.clone();
    //
    //         eval_task = task::spawn(async move {
    //             test_task(
    //                 network,
    //                 policies,
    //                 multi_progress_bar,
    //                 tensorboard_controller,
    //                 csv_writer_thread
    //             ).await
    //         });
    //     }
    // }

    let experiences_buffer_length = number_of_experiences_per_training_step;
    let multi_progress_bar = indicatif::MultiProgress::new();

    let number_of_played_games = Arc::new(AtomicU32::new(0));
    let number_of_played_games_since_learning_started = Arc::new(AtomicU32::new(0));
    let number_of_experiences_generated = Arc::new(AtomicUsize::new(0));
    let number_of_experiences_learned = Arc::new(AtomicUsize::new(0));

    let (mut sender, mut receiver) = tokio::sync::mpsc::channel::<(
        [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],
        [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]
    )>(experiences_buffer_length);

    let mini_buffer = MiniBufferSender {
        sender
    };

    let simultaneous_games = available_parallelism()
        .unwrap()
        .get() as usize;

    for i in 0..simultaneous_games {
        let number_of_played_games = number_of_played_games
            .clone();
        let number_of_played_games_since_learning_started = number_of_played_games_since_learning_started
            .clone();
        let number_of_experiences_generated = number_of_experiences_generated
            .clone();
        let number_of_experiences_learned = number_of_experiences_learned
            .clone();

        let policies = policies.clone();

        let mini_buffer = mini_buffer.clone();
        let multi_progress = multi_progress_bar.clone();
        let db = db.clone();
        tokio::spawn(async move {
            let mut rng = SmallRng::from_os_rng();

            loop {
                let states = db.get_random_states(10000).unwrap();


                for state in states {
                    let mut number_of_experiences = 0;

                    reverse_pred_process(
                        &state,
                        chance_of_keeping_experience,
                        &mut number_of_experiences,
                        mini_buffer.clone(),
                        &mut rng
                    ).await;

                    number_of_experiences_generated
                        .fetch_add(number_of_experiences, std::sync::atomic::Ordering::Relaxed);
                    number_of_played_games
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    tokio::task::yield_now().await;
                }

            }
        });
    }
    {
        let multi_progress_bar = multi_progress_bar
            .clone();

        let number_of_played_games = number_of_played_games
            .clone();
        let number_of_played_games_since_learning_started = number_of_played_games_since_learning_started
            .clone();
        let number_of_experiences_generated = number_of_experiences_generated
            .clone();
        let number_of_experiences_learned = number_of_experiences_learned
            .clone();

        tokio::spawn(async move {
            let number_of_games_pb =
                multi_progress_bar.clone().add(ProgressBar::new_spinner());

            loop {
                let games_played = number_of_played_games
                    .load(Ordering::Relaxed);
                let games_played_since_learning_started = number_of_played_games_since_learning_started
                    .load(Ordering::Relaxed);
                let number_of_experiences_generated = number_of_experiences_generated
                    .load(Ordering::Relaxed);
                let number_of_experiences_learned = number_of_experiences_learned
                    .load(Ordering::Relaxed);

                number_of_games_pb.set_message(
                    format!("Games played: {} Games in buffer: {} Experiences generated: {} Experiences learned: {} Experiences per game: {:.2}",
                            games_played,
                            games_played_since_learning_started,
                            number_of_experiences_generated,
                            number_of_experiences_learned,
                            number_of_experiences_generated as f64 / games_played as f64
                    )
                );
                number_of_games_pb.tick();

                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
    }

    {
        let multi_progress_bar = multi_progress_bar
            .clone();

        tokio::spawn(async move {
            let number_of_games_pb =
                multi_progress_bar.clone().add(ProgressBar::new_spinner());

            let mut network_trainer = NetworkTrainer::new();

            let csv_writer_thread = CSVWriterThread::new(
                "test.csv".into(),
                &[
                    "number_of_experiences",
                    "type",
                    "temp",
                    "is_consistent",
                    "not_consistent_reason",
                    "game_type",
                ],
                None,
                false
            );

            const EVAL_NUMBER_OF_GAMES: usize = 150;
            const EVAL_NUMBER_OF_GAMES_PER_STEP: usize = 1;
            const EVAL_MAC_CONCURRENT_GAMES: usize = 150;
            const EVAL_BUFFER_SIZE_PROCESSORS: usize = 1024;
            const EVAL_NUM_PROCESSORS: usize = 2;
            let mut eval_task: JoinHandle<()> = {

                let policies = policies.clone();
                let csv_writer_thread = csv_writer_thread.clone();
                let tensorboard = tensorboard_controller.clone();
                let multi_progress_bar = multi_progress_bar.clone();
                let network = network.clone_network();

                task::spawn(async move {
                    async_tests(
                        network,
                        policies.clone(),
                        EVAL_NUMBER_OF_GAMES,
                        EVAL_NUMBER_OF_GAMES_PER_STEP,
                        EVAL_MAC_CONCURRENT_GAMES,
                        EVAL_BUFFER_SIZE_PROCESSORS,
                        Duration::from_millis(1),
                        EVAL_NUM_PROCESSORS,
                        multi_progress_bar.clone(),
                        0,
                        "eval".to_string(),
                        tensorboard,
                        csv_writer_thread
                    ).await;
                })
            };

            let mut epoch = 0;


            let mut buffer = Vec::with_capacity(experiences_buffer_length);

            let mut pb = multi_progress_bar.add(
                ProgressBar::new(experiences_buffer_length as u64)
            );

            pb
                .set_style(
                    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
                        .unwrap()
                        .progress_chars("#>-")
                );
            pb.set_message("Waiting for enough experiences".to_string());

            loop {
                let received = receiver
                    .recv()
                    .await
                    .unwrap();

                buffer.push(received);
                pb.set_position(buffer.len() as u64);

                if buffer.len() < experiences_buffer_length {
                    continue
                }

                pb.finish_and_clear();

                let multi_progress_bar = multi_progress_bar.clone();

                network_trainer.train_network(
                    &buffer,
                    epoch,
                    &mut network,
                    multi_progress_bar.clone(),
                );

                buffer.clear();
                pb = multi_progress_bar.add(
                    ProgressBar::new(experiences_buffer_length as u64)
                );
                pb
                    .set_style(
                        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
                            .unwrap()
                            .progress_chars("#>-")
                    );
                pb.set_message("Waiting for enough experiences".to_string());


                number_of_experiences_learned
                    .store(network.number_of_trained_examples() as usize, Ordering::Relaxed);

                let network = network.
                    clone_network();
                let number_of_trained_example =
                    network.number_of_trained_examples();
                let policies = policies.clone();

                if eval_task.is_finished() {
                    let tensorboard = tensorboard_controller.clone();
                    let csv_writer_thread = csv_writer_thread.clone();

                    eval_task = task::spawn(async move {
                        async_tests(
                            network,
                            policies.clone(),
                            EVAL_NUMBER_OF_GAMES,
                            EVAL_NUMBER_OF_GAMES_PER_STEP,
                            EVAL_MAC_CONCURRENT_GAMES,
                            EVAL_BUFFER_SIZE_PROCESSORS,
                            Duration::from_millis(1),
                            EVAL_NUM_PROCESSORS,
                            multi_progress_bar.clone(),
                            number_of_trained_example as usize,

                            "eval".to_string(),
                            tensorboard,

                            csv_writer_thread

                        ).await;
                    });

                }

                epoch += 1;
            }
        }).await.unwrap();
    }

    tokio::time::sleep(std::time::Duration::from_secs(1000000000000)).await;
}

pub fn train_impi2(
    chance_of_keeping_experience: f64,

    number_of_experiences_per_training_step: usize,

    mut network: Box<dyn ImpiNetwork>,
    mut az_network: PyObject,
    tensorboard_controller: Arc<dyn TensorboardSender>,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .enable_time()
        .build()
        .unwrap();

    let available_parallelism = available_parallelism().unwrap().get();

    rt.block_on(async {

        // let az_network = PyAlphaZeroNetwork::new(az_network);
        //
        // let (az_workers, mut az_factory, joins) = AZWorkers::with_new_batch_processors(
        //     4096,
        //     200*15,
        //     200*15,
        //     300*400,
        //     1024,
        //     Duration::from_millis(5),
        //     2,
        //     Box::new(az_network)
        // );
        //
        // let az_policy = az_factory.policy(AlphaZeroEvaluationOptions {
        //     iterations: 200,
        //     dirichlet_alpha: 0.5,
        //     dirichlet_epsilon: 0.0,
        //     puct_exploration_constant: 4.0,
        //     min_or_max_value: 0.0,
        //     par_iterations: 0,
        //     virtual_loss: 0.0,
        // });

        let iterations = 100000;
        let mcts_policy = MCTSFullDokoPolicy::new(
            available_parallelism,
            iterations + 1,
            iterations + 1,
            iterations,
            2.5f64
        );

        train_impi2_c(
            chance_of_keeping_experience,
            number_of_experiences_per_training_step,
            PlayerZeroOrientedArr::from_full([
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
                Arc::new(mcts_policy.clone()),
            ]),
            tensorboard_controller,
            network
        ).await
    });

}