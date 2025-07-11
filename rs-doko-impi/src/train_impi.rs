
// struct ExecutePolicyMessage {
//     state: FdoState,
//
//     return_channel: tokio::sync::oneshot::Sender<FdoAction>
// }

use crate::csv_writer_thread::CSVWriterThread;
use crate::forward::async_tests;
use crate::network::ImpiNetwork;
use crate::save_log_maybe;
use crate::tensorboard::TensorboardSender;
use crate::training::NetworkTrainer;
use async_trait::async_trait;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::{IndexedRandom, SmallRng};
use rand::{Rng, SeedableRng};
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};
// use rs_doko_networks::full_doko::ipi_network::{FullDokoImperfectInformationNetwork, FullDokoImperfectInformationNetworkConfiguration};
use rs_doko_networks::full_doko::var1::encode_ipi::encode_state_ipi;
use rs_doko_networks::full_doko::var1::ipi_output::ImperfectInformationOutput;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::display::display::display_game;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::reservation::reservation::FdoVisibleReservation;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::mem::transmute;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task;
use tokio::task::JoinHandle;

macro_rules! debug_println {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            println!($($arg)*);
        }
    }
}

pub async fn reverse_pred_process(
    state: &FdoState,
    chance_of_keeping_experience: f64,

    number_of_experiences: &mut usize,

    mut erb: MiniBufferSender,
    rng: &mut SmallRng
) {
    let with_log = rng.random_bool(1.0f64 / 10000f64);

    let mut log = if with_log {
        Some(String::new())
    } else {
        None
    };

    let obs = state
        .observation_for_current_player();
    let current_player = obs
        .current_player
        .unwrap();

    let mut hands = obs
        .phi_real_hands
        .clone();

    let mut reservations = obs
        .phi_real_reservations
        .reservations
        .to_zero_array_remaining_option()
        .clone();

    log = log
        .map(|mut log| {
            log.push_str(display_game(obs.clone()).as_str());
            log.push_str("\n");
            log
        });


    for i in 0..3 {
        let i = 3 - i;

        let player_to_guess = current_player + i;

        log = log
            .map(|mut log| {
                log.push_str(format!("\n===\nVorheriger Zustand: \n").as_str());
                log.push_str(format!("Hände: {}\n", hands).as_str());
                log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                log.push_str("\n");
                log.push_str(format!("Nun an der Reihe: {:#?}\n", player_to_guess).as_str());

                log
            });

        // Das Target, welches das neuronale Netzwerk ausgeben  muss. Also
        // der tatsächlich zu erratende Vorbehalt.
        let target = reservations[player_to_guess];
        // Damit es was zu erraten gibt, setzen wir den Wert auf None :)
        reservations[player_to_guess] = None;

        // Uns intressieren die Targets aber auch nur, wenn sie überhaupt durch uns
        // erraten werden müssen.
        let is_interesting = match obs.visible_reservations[player_to_guess] {
            FdoVisibleReservation::NotRevealed => true,
            _ => false
        };

        log = log.map(|mut log| {
            log.push_str(format!("Target: {:#?}\n", target).as_str());
            log.push_str(format!("Interessant: {:#?}\n", is_interesting).as_str());
            log
        });

        // Dann fügen wir es zu unseren Trainingsdaten hinzu.
        if is_interesting && rng.random_bool(chance_of_keeping_experience) {
            // if obs.visible_reservations[player_to_guess] == FdoVisibleReservation::NotRevealed && target.is_none() {
            //     debug_println!("{:#?}", state);
            //     debug_println!("{:#?}", obs);
            //     debug_println!("{:#?}", reservations);
            // }

            let target = ImperfectInformationOutput::from_reservation(target.unwrap());

            let e_state = encode_state_ipi(
                state,
                &obs,
                hands,
                reservations,
                player_to_guess
            );

            log = log.map(|mut log| {
                log.push_str("Eingefügt wird: \n");
                log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                log.push_str(format!("Hände: {}\n", hands).as_str());
                log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                log
            });

            erb.append(
                e_state,
                target.to_arr()
            ).await;
            *number_of_experiences += 1;
        }
    }

    for hand_card_index in 0..12 {
        for i in 0..3 {
            let i = 3 - i;

            let player_to_guess = current_player + i;

            log = log
                .map(|mut log| {
                    log.push_str(format!("Vorheriger Zustand: \n").as_str());
                    log.push_str(format!("Hände: {:#?}\n", hands).as_str());
                    log.push_str(format!("Vorbehalte: {:#?}\n", reservations).as_str());
                    log.push_str("\n");
                    log.push_str(format!("Nun an der Reihe: {:#?}\n", player_to_guess).as_str());

                    log
                });

            if hands[player_to_guess].len() == 0 {
                log = log
                    .map(|mut log| {
                        log.push_str(format!("Keine Karten mehr in der Hand. Überspringen.\n").as_str());
                        log
                    });
                continue;
            }

            // Das Target, welches das neuronale Netzwerk ausgeben  muss. Also
            // der tatsächlich zu erratende Vorbehalt.
            let target: Vec<_> = hands[player_to_guess]
                .iter()
                .collect();

            let target = target
                .choose(rng)
                .unwrap();

            hands[player_to_guess].remove(*target);

            log = log
                .map(|mut log| {
                    log.push_str(format!("Target: {:#?}\n", target).as_str());
                    log
                });

            if rng.random_bool(chance_of_keeping_experience) {
                let single_pred = false;

                if (single_pred) {
                    let target = ImperfectInformationOutput::from_card(target);

                    let e_state = encode_state_ipi(
                        state,
                        &obs,
                        hands,
                        reservations,
                        player_to_guess
                    );

                    log = log.map(|mut log| {
                        log.push_str("Eingefügt wird: \n");
                        log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                        log.push_str(format!("Player: {}\n", player_to_guess).as_str());
                        log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                        log.push_str(format!("Hände: {}\n", hands).as_str());
                        log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                        log.push_str(format!("Remaining cards: {}\n", hands[player_to_guess]).as_str());
                        log.push_str(format!("Remaining cards (real): {:?}\n", target.to_arr()).as_str());
                        log
                    });

                    erb.append(
                        e_state,
                        target.to_arr()
                    ).await;
                    *number_of_experiences += 1;
                } else {
                    let remaining_cards = obs
                        .phi_real_hands[player_to_guess]
                        .minus_hand(hands[player_to_guess]);

                    let e_state = encode_state_ipi(
                        state,
                        &obs,
                        hands,
                        reservations,
                        player_to_guess
                    );

                    log = log.map(|mut log| {
                        log.push_str("Eingefügt wird: \n");
                        log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                        log.push_str(format!("Player: {}\n", player_to_guess).as_str());
                        log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                        log.push_str(format!("Hände: {}\n", hands).as_str());
                        log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                        log.push_str(format!("Remaining cards: {}\n", remaining_cards).as_str());
                        log.push_str(format!("Remaining cards (real): {:?}\n", ImperfectInformationOutput::arr_from_hand(remaining_cards)).as_str());
                        log
                    });

                    erb.append(
                        e_state,
                        ImperfectInformationOutput::arr_from_hand(remaining_cards)
                    ).await;
                    *number_of_experiences += 1;
                }
            }
        }
    }

    save_log_maybe::save_log_maybe(log, "game_pred_logs");
}


pub fn reverse_pred_process_unsync(
    state: &FdoState,
    chance_of_keeping_experience: f64,

    mut erb: &mut Vec<([i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE], [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE])>,
    rng: &mut SmallRng
) {
    let with_log = rng.random_bool(1.0f64 / 10000f64);

    let mut log = if with_log {
        Some(String::new())
    } else {
        None
    };

    let obs = state
        .observation_for_current_player();
    let current_player = obs
        .current_player
        .unwrap();

    let mut hands = obs
        .phi_real_hands
        .clone();

    let mut reservations = obs
        .phi_real_reservations
        .reservations
        .to_zero_array_remaining_option()
        .clone();

    log = log
        .map(|mut log| {
            log.push_str(display_game(obs.clone()).as_str());
            log.push_str("\n");
            log
        });


    for i in 0..3 {
        let i = 3 - i;

        let player_to_guess = current_player + i;

        log = log
            .map(|mut log| {
                log.push_str(format!("\n===\nVorheriger Zustand: \n").as_str());
                log.push_str(format!("Hände: {}\n", hands).as_str());
                log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                log.push_str("\n");
                log.push_str(format!("Nun an der Reihe: {:#?}\n", player_to_guess).as_str());

                log
            });

        // Das Target, welches das neuronale Netzwerk ausgeben  muss. Also
        // der tatsächlich zu erratende Vorbehalt.
        let target = reservations[player_to_guess];
        // Damit es was zu erraten gibt, setzen wir den Wert auf None :)
        reservations[player_to_guess] = None;

        // Uns intressieren die Targets aber auch nur, wenn sie überhaupt durch uns
        // erraten werden müssen.
        let is_interesting = match obs.visible_reservations[player_to_guess] {
            FdoVisibleReservation::NotRevealed => true,
            _ => false
        };

        log = log.map(|mut log| {
            log.push_str(format!("Target: {:#?}\n", target).as_str());
            log.push_str(format!("Interessant: {:#?}\n", is_interesting).as_str());
            log
        });

        // Dann fügen wir es zu unseren Trainingsdaten hinzu.
        if is_interesting && rng.random_bool(chance_of_keeping_experience) {
            // if obs.visible_reservations[player_to_guess] == FdoVisibleReservation::NotRevealed && target.is_none() {
            //     debug_println!("{:#?}", state);
            //     debug_println!("{:#?}", obs);
            //     debug_println!("{:#?}", reservations);
            // }

            let target = ImperfectInformationOutput::from_reservation(target.unwrap());

            let e_state = encode_state_ipi(
                state,
                &obs,
                hands,
                reservations,
                player_to_guess
            );

            log = log.map(|mut log| {
                log.push_str("Eingefügt wird: \n");
                log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                log.push_str(format!("Hände: {}\n", hands).as_str());
                log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                log
            });

            erb.push((e_state, target.to_arr()));
        }
    }

    for hand_card_index in 0..12 {
        for i in 0..3 {
            let i = 3 - i;

            let player_to_guess = current_player + i;

            log = log
                .map(|mut log| {
                    log.push_str(format!("Vorheriger Zustand: \n").as_str());
                    log.push_str(format!("Hände: {:#?}\n", hands).as_str());
                    log.push_str(format!("Vorbehalte: {:#?}\n", reservations).as_str());
                    log.push_str("\n");
                    log.push_str(format!("Nun an der Reihe: {:#?}\n", player_to_guess).as_str());

                    log
                });

            if hands[player_to_guess].len() == 0 {
                log = log
                    .map(|mut log| {
                        log.push_str(format!("Keine Karten mehr in der Hand. Überspringen.\n").as_str());
                        log
                    });
                continue;
            }

            // Das Target, welches das neuronale Netzwerk ausgeben  muss. Also
            // der tatsächlich zu erratende Vorbehalt.
            let target: Vec<_> = hands[player_to_guess]
                .iter()
                .collect();

            let target = target
                .choose(rng)
                .unwrap();

            hands[player_to_guess].remove(*target);

            log = log
                .map(|mut log| {
                    log.push_str(format!("Target: {:#?}\n", target).as_str());
                    log
                });

            if rng.random_bool(chance_of_keeping_experience) {
                let single_pred = false;

                if (single_pred) {
                    let target = ImperfectInformationOutput::from_card(target);

                    let e_state = encode_state_ipi(
                        state,
                        &obs,
                        hands,
                        reservations,
                        player_to_guess
                    );

                    log = log.map(|mut log| {
                        log.push_str("Eingefügt wird: \n");
                        log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                        log.push_str(format!("Player: {}\n", player_to_guess).as_str());
                        log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                        log.push_str(format!("Hände: {}\n", hands).as_str());
                        log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                        log.push_str(format!("Remaining cards: {}\n", hands[player_to_guess]).as_str());
                        log.push_str(format!("Remaining cards (real): {:?}\n", target.to_arr()).as_str());
                        log
                    });

                    erb.push((
                        e_state,
                        target.to_arr()
                    ));
                } else {
                    let remaining_cards = obs
                        .phi_real_hands[player_to_guess]
                        .minus_hand(hands[player_to_guess]);

                    let e_state = encode_state_ipi(
                        state,
                        &obs,
                        hands,
                        reservations,
                        player_to_guess
                    );

                    log = log.map(|mut log| {
                        log.push_str("Eingefügt wird: \n");
                        log.push_str(format!("Eingefügt: {:?}\n", e_state).as_str());
                        log.push_str(format!("Player: {}\n", player_to_guess).as_str());
                        log.push_str(format!("Target (real): {:#?}\n", target).as_str());
                        log.push_str(format!("Hände: {}\n", hands).as_str());
                        log.push_str(format!("Vorbehalte: {:?}\n", reservations).as_str());
                        log.push_str(format!("Remaining cards: {}\n", remaining_cards).as_str());
                        log.push_str(format!("Remaining cards (real): {:?}\n", ImperfectInformationOutput::arr_from_hand(remaining_cards)).as_str());
                        log
                    });

                    erb.push((
                        e_state,
                        ImperfectInformationOutput::arr_from_hand(remaining_cards)
                    ));
                }
            }
        }
    }

    save_log_maybe::save_log_maybe(log, "game_pred_logs");
}

async fn play_game(
    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,
    rng: &mut SmallRng,

    chance_of_keeping_experience: f64,
    erb: MiniBufferSender,

    multi_progress: &MultiProgress
) -> usize {
    let mut state = FdoState::new_game(rng);

    let mut number_of_experiences = 0;

    loop {
        let obs = state.observation_for_current_player();

        if obs.finished_stats.is_some() {
            return number_of_experiences;
        }

        reverse_pred_process(
            &state,
            chance_of_keeping_experience,
            &mut number_of_experiences,
            erb.clone(),
            rng
        ).await;

        let current_player = obs
            .current_player
            .unwrap();

        if obs.allowed_actions_current_player.len() == 1 {
            unsafe {
                state.play_action(transmute(obs.allowed_actions_current_player.0.0));
            }

            continue;
        }

        let action = policies[current_player]
            .execute_policy(&state, &obs, rng)
            .await;

        state.play_action(action);
    }
}

#[async_trait]
pub trait FullDokoPolicy : Sync + Send + Debug {
    async fn execute_policy(
        &self,
        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> FdoAction;
}

#[derive(Debug)]
pub struct _ModifiedRandomFullDokoPolicy {
    pub sender: async_channel::Sender<()>
}

impl _ModifiedRandomFullDokoPolicy {
    pub fn new() -> _ModifiedRandomFullDokoPolicy {
        _ModifiedRandomFullDokoPolicy {
            sender: todo!()
        }
    }
}

#[async_trait]
impl FullDokoPolicy for _ModifiedRandomFullDokoPolicy {
    async fn execute_policy(
        &self,
        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> FdoAction {
        self.sender
            .send(())
            .await;

        todo!()
    }
}

struct MiniBuffer {

}


#[derive(Serialize, Deserialize)]
struct DBRecord {
    state: heapless::Vec<i64, { FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE }>,
    result: heapless::Vec<f32, { FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE }>,
}

impl DBRecord {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self)
            .expect("Failed to serialize DBRecord")
    }

    pub fn from_bytes(bytes: &[u8]) -> DBRecord {
        bincode::deserialize(bytes)
            .expect("Failed to deserialize DBRecord")
    }
}

#[derive(Clone)]
pub struct MiniBufferSender {
    pub sender: mpsc::Sender<([i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE], [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE])>
}

impl MiniBufferSender {
    pub async fn append(
        &mut self,
        state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],
        result: [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]
    ) {
        self.sender
            .send((state, result))
            .await
            .unwrap();
    }
}



pub async fn train_impi(
    simultaneous_games: usize,

    adam_learning_rate: f64,
    minibatch_size: usize,

    mut network: Box<dyn ImpiNetwork>,

    chance_of_keeping_experience: f64,

    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,

    run_name: String,

    learn_after_x_games: u32,
    eval_every_x_epoch: u32,

    tensorboard: Arc<dyn TensorboardSender>
) {
    let experiences_buffer_length = 1_000_000;
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

    // Start x tasks gaining data
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
        tokio::spawn(async move {
            let mut rng = SmallRng::from_os_rng();

            loop {
                let noe = play_game(
                    policies.clone(),
                    &mut rng,
                    chance_of_keeping_experience,
                    mini_buffer.clone(),
                    &multi_progress
                ).await;

                number_of_experiences_generated
                    .fetch_add(noe, std::sync::atomic::Ordering::Relaxed);
                number_of_played_games
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                tokio::task::yield_now().await;
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

        let run_name = run_name.clone();
        tokio::spawn(async move {
            let number_of_games_pb =
                multi_progress_bar.clone().add(ProgressBar::new_spinner());

            let mut network_trainer = NetworkTrainer::new();

            let csv_writer_thread = CSVWriterThread::new(
                "test2.csv".into(),
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

            const EVAL_NUMBER_OF_GAMES: usize = 300;
            const EVAL_NUMBER_OF_GAMES_PER_STEP: usize = 1;
            const EVAL_MAC_CONCURRENT_GAMES: usize = 300;
            const EVAL_BUFFER_SIZE_PROCESSORS: usize = 2048;
            const EVAL_NUM_PROCESSORS: usize = 2;
            let mut eval_task: JoinHandle<()> = {

            let policies = policies.clone();
            let csv_writer_thread = csv_writer_thread.clone();
            let tensorboard = tensorboard.clone();
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
            let run_name = run_name.clone();


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

                if epoch % eval_every_x_epoch as usize == 0 && eval_task.is_finished() {
                    let tensorboard = tensorboard.clone();
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
