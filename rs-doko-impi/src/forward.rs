use crate::eval::batch_eval_net_fn::BatchNetworkEvaluateNetFn;
use crate::eval::eval_net_fn::EvaluateNetFn;
use crate::eval::single_eval_net_fn::SingleNetworkEvaluateNetFn;
use crate::network::ImpiNetwork;
use crate::train_impi::FullDokoPolicy;
use futures::task::SpawnExt;
use futures::{stream, SinkExt, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::{IndexedRandom, SmallRng};
use rand::{Rng, SeedableRng};
use rand_distr::num_traits::Signed;
use rs_doko_networks::full_doko::ipi_network::FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE;
use rs_doko_networks::full_doko::var1::encode_ipi::encode_state_ipi;
use rs_doko_networks::full_doko::var1::ipi_output::ImperfectInformationOutput;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::display::display::display_game;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::matching::is_consistent::{NotConsistentReason, _is_consistent};
use rs_full_doko::reservation::reservation::{FdoReservation, FdoVisibleReservation};
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use std::fmt::Debug;
use std::intrinsics::transmute;
use std::sync::Arc;
use strum::{EnumCount, IntoEnumIterator};
use tokio::task::JoinSet;
use rs_full_doko::player::player_set::FdoPlayerSet;
use crate::csv_writer_thread::CSVWriterThread;
use crate::next_consistent::{next_possible_card, next_possible_reservation};
use crate::save_log_maybe;
use crate::tensorboard::TensorboardSender;

pub fn softmax_inplace<const N: usize>(data: &mut [f32; N], temperature: f32) {
    let temp = temperature.max(1e-6);

    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    for i in 0..data.len() {
        data[i] = ((data[i] - max_val) / temp).exp();
    }

    let sum: f32 = data.iter().sum();

    if sum.is_finite() && sum > 0.0 {
        for i in 0..data.len() {
            data[i] /= sum;
        }
    } else {
        let val = 1.0 / data.len() as f32;
        for i in 0..data.len() {
            data[i] = val;
        }
    }
}

pub fn apply_mask_inplace<const N: usize>(data: &mut [f32; N], mask: &[bool; N]) {
    for i in 0..data.len() {
        if !mask[i] {
            data[i] = f32::MIN;
        }
    }
}

pub fn normalize_float(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }

    if value.is_infinite() {
        return 0.0;
    }

    if value.is_negative() {
        return 0.0;
    }

    value
}

async fn stochastic_next(
    impi_state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],

    temperature: f32,
    mask: [bool; ImperfectInformationOutput::COUNT],

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    rng: &mut SmallRng,
) -> (ImperfectInformationOutput, Vec<(usize, f32)>) {
    let mut next_card_logits = evaluate_net_fn.evaluate(impi_state).await;

    apply_mask_inplace(&mut next_card_logits, &mask);
    softmax_inplace(&mut next_card_logits, temperature);

    // println!("next_card_logits: {:?}", next_card_logits);
    // println!("next_card_logits: {}", ImperfectInformationOutput::stringify_logits(next_card_logits));

    let next_card_logits = next_card_logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .map(|(i, x)| (i, normalize_float(x)))
        .collect::<Vec<_>>();

    // println!("next_card_logits: {:?}", next_card_logits);

    // let next_card = next_card
    //     .iter()
    //     .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    //     .map(|(i, _)| i)
    //     .unwrap();

    let next_card = next_card_logits
        .choose_weighted(rng, |item| normalize_float(item.1))
        .map(|(i, _)| i)
        .unwrap();

    (
        ImperfectInformationOutput::from_index(*next_card),
        next_card_logits,
    )
}

async fn stochastic_next_card_mask(
    impi_state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],

    mask: [bool; ImperfectInformationOutput::COUNT],

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    temperature: f32,
    rng: &mut SmallRng,
) -> (FdoCard, Vec<(usize, f32)>) {
    let (next_card, logits) =
        stochastic_next(impi_state, temperature, mask, evaluate_net_fn, rng).await;

    (next_card.to_card().unwrap(), logits)
}

async fn stochastic_next_reservation_mask(
    impi_state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],

    mask: [bool; ImperfectInformationOutput::COUNT],

    temperature: f32,

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    rng: &mut SmallRng,
) -> (FdoReservation, Vec<(usize, f32)>) {
    let (next_reservation, logits) =
        stochastic_next(impi_state, temperature, mask, evaluate_net_fn, rng).await;

    (next_reservation.to_reservation().unwrap(), logits)
}

async fn stochastic_next_card(
    impi_state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    temperature: f32,
    rng: &mut SmallRng,
) -> (FdoCard, Vec<(usize, f32)>) {
    let mask = ImperfectInformationOutput::card_only_mask();

    let (next_card, logits) =
        stochastic_next(impi_state, temperature, mask, evaluate_net_fn, rng).await;

    (next_card.to_card().unwrap(), logits)
}

async fn stochastic_next_reservation(
    impi_state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],

    temperature: f32,

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    rng: &mut SmallRng,
) -> (FdoReservation, Vec<(usize, f32)>) {
    let mask = ImperfectInformationOutput::reservation_only_mask();

    let (next_reservation, logits) =
        stochastic_next(impi_state, temperature, mask, evaluate_net_fn, rng).await;

    (next_reservation.to_reservation().unwrap(), logits)
}

#[derive(Debug, Clone)]
pub enum ForwardPredResult {
    Consistent(FdoState),
    NotConsistent(NotConsistentReason),
}

pub async fn forward_pred_process(
    state: &FdoState,

    temperature: f32,

    log_chance: f32,

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    rng: &mut SmallRng,

    multi_progress: &MultiProgress,
) -> ForwardPredResult {
    let with_log = true;

    let mut log = if with_log { Some(String::new()) } else { None };

    let obs = state.observation_for_current_player();
    let current_player = obs.current_player.unwrap();

    let mut remaining_slots = obs.phi_real_hands.map(|hand| hand.len());

    let mut hands = PlayerZeroOrientedArr::from_full([FdoHand::empty(); 4]);
    let mut reservations: PlayerZeroOrientedArr<Option<FdoReservation>> =
        PlayerZeroOrientedArr::from_full([None; 4]);

    hands[current_player] = obs.phi_real_hands[current_player];

    reservations[current_player] = obs
        .phi_real_reservations
        .reservations
        .get(current_player)
        .copied();

    log = log.map(|mut log| {
        log.push_str(display_game(obs.clone()).as_str());
        log
    });

    // println!("hands: {:#?}", obs.phi_real_hands);
    // println!("current_player: {:#?}", obs.current_player);

    for hand_card_index in 0..12 {
        for i in 0..3 {
            let i = i + 1;
            let player_to_guess = current_player + i;

            if remaining_slots[player_to_guess] == 0 {
                continue;
            }

            let encoded_state =
                encode_state_ipi(&state, &obs, hands, reservations, player_to_guess);

            let next_possible_cards = next_possible_card(
                &state,
                &obs,
                hands,
                reservations.to_oriented_arr().clone(),
                player_to_guess
            );

            if (next_possible_cards.len() == 0) {
                return ForwardPredResult::NotConsistent(NotConsistentReason::RemainingCardsLeft);
            }

            let mask = ImperfectInformationOutput::mask_from_hand(
                next_possible_cards
            );

            let (next_card, logits) =
                stochastic_next_card_mask(encoded_state, mask, &evaluate_net_fn, temperature, rng).await;
            let mut next_card = next_card;

            log = log.map(|mut log| {
                log.push_str(
                    format!(
                        "Guessed {} for player {}. (prob: {:.2}) ({})\n ",
                        next_card,
                        player_to_guess,
                        logits[ImperfectInformationOutput::from_card(&next_card) as usize].1
                            * 100f32,
                        ImperfectInformationOutput::stringify_logits(
                            logits
                                .iter()
                                .map(|x| x.1)
                                .collect::<Vec<_>>()
                                .try_into()
                                .unwrap()
                        )
                    )
                    .as_str(),
                );

                log
            });

            if hands[player_to_guess].contains_both(next_card) {
                log = log.map(|mut log| {
                    log.push_str(
                        format!(
                            "ERROR: {} already in hand of player {}.\n",
                            next_card, player_to_guess
                        )
                        .as_str(),
                    );
                    log
                });

                save_log_maybe::save_log_maybe(log, "game_logs");

                return ForwardPredResult::NotConsistent(NotConsistentReason::NotInRemainingCards);
            }

            hands[player_to_guess].add(next_card);
            remaining_slots[player_to_guess] -= 1;
        }
    }

    for i in 0..3 {
        let i = i + 1;
        let player_to_guess = current_player + i;

        if obs.visible_reservations[player_to_guess] != FdoVisibleReservation::NotRevealed {
            reservations[player_to_guess] = obs
                .phi_real_reservations
                .reservations
                .get(player_to_guess)
                .copied();
            continue;
        }

        let encoded_state = encode_state_ipi(&state, &obs, hands, reservations, player_to_guess);

        let mask = ImperfectInformationOutput::mask_for_reservation(
            next_possible_reservation(
                &state,
                &obs,
                hands,
                player_to_guess
            )
        );

        let (next_reservation, logits) =
            stochastic_next_reservation_mask(encoded_state, mask, temperature, &evaluate_net_fn, rng).await;
        let mut next_reservation = next_reservation;

        log = log.map(|mut log| {
            log.push_str(
                format!(
                    "Guessed {:?} for player {}. (prob: {:.2}) ({})\n ",
                    next_reservation,
                    player_to_guess,
                    logits[ImperfectInformationOutput::from_reservation(next_reservation) as usize]
                        .1
                        * 100f32,
                    ImperfectInformationOutput::stringify_logits(
                        logits
                            .iter()
                            .map(|x| x.1)
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap()
                    )
                )
                .as_str(),
            );

            log
        });

        reservations[player_to_guess] = Some(next_reservation);
    }

    let cons_reason = _is_consistent(&state, &obs, hands, reservations.to_oriented_arr());

    if cons_reason.is_none() {
        log = log.map(|mut log| {
            log.push_str("Consistent :-).\n");
            log
        });

        if rng.random_bool(log_chance as f64) {
            save_log_maybe::save_log_maybe(log, "game_logs");
        }

        return ForwardPredResult::Consistent(state.clone());
    }

    log = log.map(|mut log| {
        log.push_str(format!("Not consistent: {:?}.\n", cons_reason).as_str());
        log
    });

    if rng.random_bool(log_chance as f64 * 4f64) {
        let _ = save_log_maybe::save_log_maybe(log, "game_logs");
    }

    ForwardPredResult::NotConsistent(cons_reason.unwrap())
}


pub async fn forward_pred_process_with_mask(
    state: &FdoState,

    temperature: f32,

    log_chance: f32,

    evaluate_net_fn: &Arc<dyn EvaluateNetFn>,
    rng: &mut SmallRng
) -> ForwardPredResult {
    let with_log = true;

    let mut log = if with_log { Some(String::new()) } else { None };

    let obs = state.observation_for_current_player();
    let current_player = obs.current_player.unwrap();

    let mut remaining_slots = obs.phi_real_hands.map(|hand| hand.len());

    let mut hands = PlayerZeroOrientedArr::from_full([FdoHand::empty(); 4]);
    let mut reservations: PlayerZeroOrientedArr<Option<FdoReservation>> =
        PlayerZeroOrientedArr::from_full([None; 4]);

    hands[current_player] = obs.phi_real_hands[current_player];

    // for (p, h) in obs.phi_real_hands.iter_with_player() {
    //     if p == current_player {
    //         continue;
    //     }
    //     let np = next_possible_card(
    //         &state,
    //         &obs,
    //         hands,
    //         reservations.to_oriented_arr().clone(),
    //         p
    //     );
    //
    //     for x in h.iter() {
    //         if !np.contains(x) {
    //             println!("=============================");
    //             println!("=============================");
    //             println!("=============================");
    //             println!("{} not in next_possible_cards", x);
    //             println!("np: {:?}", np);
    //             println!("hands: {:?}", hands);
    //             println!("reservations: {:?}", reservations);
    //             println!("current_player: {:?}", current_player);
    //             println!("p: {:?}", p);
    //             println!("h: {:?}", h);
    //             println!("obs: {:?}", obs);
    //             println!("state: {:?}", state);
    //             println!("obs.phi_real_hands: {:?}", obs.phi_real_hands);
    //             println!("obs.phi_real_reservations: {:?}", obs.phi_real_reservations);
    //         }
    //     }
    // }

    reservations[current_player] = obs
        .phi_real_reservations
        .reservations
        .get(current_player)
        .copied();
    //
    // // ToDo: Neu
    // for p in FdoPlayerSet::all().iter() {
    //     reservations[p] = match obs.visible_reservations[p] {
    //         FdoVisibleReservation::Wedding => Some(FdoReservation::Wedding),
    //         FdoVisibleReservation::Healthy => Some(FdoReservation::Healthy),
    //         FdoVisibleReservation::NotRevealed => None,
    //         FdoVisibleReservation::DiamondsSolo => Some(FdoReservation::DiamondsSolo),
    //         FdoVisibleReservation::HeartsSolo => Some(FdoReservation::HeartsSolo),
    //         FdoVisibleReservation::SpadesSolo => Some(FdoReservation::SpadesSolo),
    //         FdoVisibleReservation::ClubsSolo => Some(FdoReservation::ClubsSolo),
    //         FdoVisibleReservation::QueensSolo => Some(FdoReservation::QueensSolo),
    //         FdoVisibleReservation::JacksSolo => Some(FdoReservation::JacksSolo),
    //         FdoVisibleReservation::TrumplessSolo => Some(FdoReservation::TrumplessSolo),
    //         FdoVisibleReservation::NoneYet => None
    //     };
    // }


    log = log.map(|mut log| {
        log.push_str(display_game(obs.clone()).as_str());
        log
    });

    // println!("hands: {:#?}", obs.phi_real_hands);
    // println!("current_player: {:#?}", obs.current_player);

    for hand_card_index in 0..12 {
        for i in 0..3 {
            let i = i + 1;
            let player_to_guess = current_player + i;

            if remaining_slots[player_to_guess] == 0 {
                continue;
            }

            let encoded_state =
                encode_state_ipi(&state, &obs, hands, reservations, player_to_guess);

            let next_possible_cards = next_possible_card(
                &state,
                &obs,
                hands,
                reservations.to_oriented_arr().clone(),
                player_to_guess
            );

            if (next_possible_cards.len() == 0) {
                log = log.map(|mut log| {
                    log.push_str(&format!("next_possible_cards: {:?}\n", next_possible_cards));
                    log.push_str(&format!("remaining_slots: {:?}\n", remaining_slots));
                    log.push_str(&format!("hands: {:?}\n", hands));
                    log.push_str(&format!("reservations: {:?}\n", reservations));
                    log.push_str(&format!("player_to_guess: {:?}\n", player_to_guess));
                    log.push_str(&format!("current_player: {:?}\n", current_player));
                    log.push_str(&format!(
                        "mask: {:?}\n",
                        ImperfectInformationOutput::mask_from_hand(next_possible_cards)
                    ));

                    log.push_str(format!("Not consistent: {:?}.\n", NotConsistentReason::RemainingCardsLeft).as_str());
                    log
                });

                let _ = save_log_maybe::save_log_maybe(log, "game_logs");

                return ForwardPredResult::NotConsistent(NotConsistentReason::RemainingCardsLeft);
            }

            let mask = ImperfectInformationOutput::mask_from_hand(
                next_possible_cards
            );

            let (next_card, logits) =
                stochastic_next_card_mask(encoded_state, mask, &evaluate_net_fn, temperature, rng).await;
            let mut next_card = next_card;

            log = log.map(|mut log| {
                log.push_str(
                    format!(
                        "Guessed {} for player {}. (prob: {:.2}) ({})\n ",
                        next_card,
                        player_to_guess,
                        logits[ImperfectInformationOutput::from_card(&next_card) as usize].1
                            * 100f32,
                        ImperfectInformationOutput::stringify_logits(
                            logits
                                .iter()
                                .map(|x| x.1)
                                .collect::<Vec<_>>()
                                .try_into()
                                .unwrap()
                        )
                    )
                        .as_str(),
                );

                log
            });

            if hands[player_to_guess].contains_both(next_card) {
                log = log.map(|mut log| {
                    log.push_str(
                        format!(
                            "ERROR: {} already in hand of player {}.\n",
                            next_card, player_to_guess
                        )
                            .as_str(),
                    );
                    log
                });

                save_log_maybe::save_log_maybe(log, "game_logs");

                return ForwardPredResult::NotConsistent(NotConsistentReason::NotInRemainingCards);
            }

            hands[player_to_guess].add(next_card);
            remaining_slots[player_to_guess] -= 1;
        }
    }

    for i in 0..3 {
        let i = i + 1;
        let player_to_guess = current_player + i;

        if obs.visible_reservations[player_to_guess] != FdoVisibleReservation::NotRevealed {
            reservations[player_to_guess] = obs
                .phi_real_reservations
                .reservations
                .get(player_to_guess)
                .copied();
            continue;
        }

        let encoded_state = encode_state_ipi(&state, &obs, hands, reservations, player_to_guess);

        let mask = ImperfectInformationOutput::mask_for_reservation(
            next_possible_reservation(
                &state,
                &obs,
                hands,
                player_to_guess
            )
        );

        let (next_reservation, logits) =
            stochastic_next_reservation_mask(encoded_state, mask, temperature, &evaluate_net_fn, rng).await;
        let mut next_reservation = next_reservation;

        log = log.map(|mut log| {
            log.push_str(
                format!(
                    "Guessed {:?} for player {}. (prob: {:.2}) ({})\n ",
                    next_reservation,
                    player_to_guess,
                    logits[ImperfectInformationOutput::from_reservation(next_reservation) as usize]
                        .1
                        * 100f32,
                    ImperfectInformationOutput::stringify_logits(
                        logits
                            .iter()
                            .map(|x| x.1)
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap()
                    )
                )
                    .as_str(),
            );

            log
        });

        reservations[player_to_guess] = Some(next_reservation);
    }

    let cons_reason = _is_consistent(&state, &obs, hands, reservations.to_oriented_arr());

    if cons_reason.is_none() {
        log = log.map(|mut log| {
            log.push_str("Consistent :-).\n");
            log
        });

        if rng.random_bool(log_chance as f64) {
            save_log_maybe::save_log_maybe(log, "game_logs");
        }

        let result = state.clone_with_different_hands_and_reservations(
            hands,
            reservations.clone(),
        );

        return ForwardPredResult::Consistent(result);
    }

    log = log.map(|mut log| {
        log.push_str(format!("Not consistent: {:?}.\n", cons_reason).as_str());
        log
    });

    if rng.random_bool(log_chance as f64 * 4f64) {
        let _ = save_log_maybe::save_log_maybe(log, "game_logs");
    }

    ForwardPredResult::NotConsistent(cons_reason.unwrap())
}



pub async fn async_tests(
    current_network: Box<dyn ImpiNetwork>,

    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,

    number_of_games: usize,
    number_of_states_per_step: usize,

    max_concurrent_games: usize,

    buffer_size_processors: usize,
    batch_timeout_processors: std::time::Duration,
    num_processors: usize,
    multi_progress_bar: MultiProgress,
    number_of_experiences: usize,

    tensorboard_prefix: String,
    tensorboard: Arc<dyn TensorboardSender>,
    csv_writer_thread: CSVWriterThread,
) {
    let (batch_network_evaluate_fn, mut batch_network_processors_join_set) =
        BatchNetworkEvaluateNetFn::new(
            current_network,
            buffer_size_processors,
            batch_timeout_processors,
            num_processors,
        );
    let batch_network_evaluate_fn: Arc<dyn EvaluateNetFn> = Arc::new(batch_network_evaluate_fn);

    let game_control_channel = async_channel::unbounded::<
        tokio::sync::oneshot::Sender<(Vec<ForwardPredResult>, Vec<ForwardPredResult>)>,
    >();

    let mut worker_tasks_join_set = JoinSet::new();

    for i in 0..max_concurrent_games {
        let batch_network_evaluate_fn = batch_network_evaluate_fn.clone();
        let game_control_channel = game_control_channel.clone();
        let policies = policies.clone();
        let csv_writer_thread = csv_writer_thread.clone();

        let multi_progress_bar = multi_progress_bar.clone();
        let tensorboard_prefix = tensorboard_prefix.clone();

        worker_tasks_join_set.spawn(async move {
            let mut rng = SmallRng::from_os_rng();

            loop {
                let callback = game_control_channel.1.recv().await;

                let callback = match callback {
                    Ok(callback) => callback,
                    Err(err) => {
                        println!("Worker task finished. {:?}", err);
                        break;
                    }
                };

                let mut results_temp_1 = JoinSet::new();
                let mut results_temp_0 = JoinSet::new();

                let mut state = FdoState::new_game(&mut rng);

                loop {
                    let obs = state.observation_for_current_player();

                    if obs.finished_stats.is_some() {
                        break;
                    }

                    for _ in 0..number_of_states_per_step {
                        {
                            let state = state.clone();
                            let batch_network_evaluate_fn = batch_network_evaluate_fn.clone();
                            let tensorboard_prefix = tensorboard_prefix.clone();
                            let number_of_experiences = number_of_experiences;
                            let multi_progress_bar = multi_progress_bar.clone();
                            let csv_writer_thread = csv_writer_thread.clone();

                            results_temp_1.spawn(async move {
                                let mut rng = SmallRng::from_os_rng();

                                let result = forward_pred_process(
                                    &state,
                                    1.0,
                                    1.0f32 / 10000f32,
                                    &batch_network_evaluate_fn,
                                    &mut rng,
                                    &multi_progress_bar,
                                )
                                .await;

                                csv_writer_thread.write_row(vec![
                                    format!("{}", number_of_experiences),
                                    format!("{}", tensorboard_prefix),
                                    format!("{}", 1.0),
                                    format!(
                                        "{}",
                                        matches!(result, ForwardPredResult::Consistent(_))
                                    ),
                                    format!(
                                        "{}",
                                        match result.clone() {
                                            ForwardPredResult::Consistent(c) => "".to_string(),
                                            ForwardPredResult::NotConsistent(nc) =>
                                                format!("{:?}", nc),
                                        }
                                    ),
                                    format!(
                                        "{}",
                                        state
                                            .game_type
                                            .map(|x| x.to_string())
                                            .unwrap_or("".to_string())
                                    ),
                                ]);

                                result
                            });
                        }

                        {
                            let state = state.clone();
                            let batch_network_evaluate_fn = batch_network_evaluate_fn.clone();
                            let tensorboard_prefix = tensorboard_prefix.clone();
                            let number_of_experiences = number_of_experiences;
                            let multi_progress_bar = multi_progress_bar.clone();
                            let csv_writer_thread = csv_writer_thread.clone();
                            results_temp_0.spawn(async move {
                                let mut rng = SmallRng::from_os_rng();
                                let result = forward_pred_process(
                                    &state,
                                    0f32,
                                    1.0f32 / 10000f32,
                                    &batch_network_evaluate_fn,
                                    &mut rng,
                                    &multi_progress_bar,
                                )
                                .await;

                                csv_writer_thread.write_row(vec![
                                    format!("{}", number_of_experiences),
                                    format!("{}", tensorboard_prefix),
                                    format!("{}", 1.0),
                                    format!(
                                        "{}",
                                        matches!(result, ForwardPredResult::Consistent(_))
                                    ),
                                    format!(
                                        "{}",
                                        match result.clone() {
                                            ForwardPredResult::Consistent(c) => "".to_string(),
                                            ForwardPredResult::NotConsistent(nc) =>
                                                format!("{:?}", nc),
                                        }
                                    ),
                                    format!(
                                        "{}",
                                        state
                                            .game_type
                                            .map(|x| x.to_string())
                                            .unwrap_or("".to_string())
                                    ),
                                ]);

                                result
                            });
                        }
                    }

                    let current_player = obs.current_player.unwrap();

                    if obs.allowed_actions_current_player.len() == 1 {
                        unsafe {
                            state.play_action(transmute(obs.allowed_actions_current_player.0.0));
                        }
                        continue;
                    }

                    state.play_action(
                        policies[current_player]
                            .execute_policy(&state, &obs, &mut rng)
                            .await,
                    );
                }

                let results_temp_1 = results_temp_1.join_all().await;
                let results_temp_0 = results_temp_0.join_all().await;

                callback.send((results_temp_1, results_temp_0)).unwrap();
            }
        });
    }

    let mut pb = multi_progress_bar.add(ProgressBar::new(number_of_games as u64));

    pb
        .set_style(
            ProgressStyle::with_template("{spinner:.green} {msg} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
                .unwrap()
                .progress_chars("#>-")
        );
    pb.set_message(format!("Evaluating (at {})", number_of_experiences).to_string());

    // Send jobs to the workers
    let result = stream::iter(0..number_of_games)
        .map(|_| async {
            let (sender, receiver) = tokio::sync::oneshot::channel();

            game_control_channel.0.send(sender).await.unwrap();

            let result = receiver.await.unwrap();

            pb.inc(1);

            result
        })
        .buffer_unordered(max_concurrent_games)
        .collect::<Vec<_>>()
        .await;

    pb.finish();

    // Wir sind fertig.

    // Alle Batch-Prozessoren abbrechen.
    batch_network_processors_join_set.abort_all();

    // Alle Worker-Tasks abbrechen.
    worker_tasks_join_set.abort_all();

    let temp_1_flattened = result
        .iter()
        .map(|(temp_1, temp_0)| temp_1)
        .flatten()
        .collect::<Vec<_>>();

    let temp_0_flattened = result
        .iter()
        .map(|(temp_1, temp_0)| temp_0)
        .flatten()
        .collect::<Vec<_>>();

    let not_consistent_temp1 = temp_1_flattened
        .iter()
        .filter(|temp_1| {
            if let ForwardPredResult::NotConsistent(temp_1) = temp_1 {
                true
            } else {
                false
            }
        })
        .count();

    let not_consistent_temp0 = temp_0_flattened
        .iter()
        .filter(|temp_0| {
            if let ForwardPredResult::NotConsistent(temp_0) = temp_0 {
                true
            } else {
                false
            }
        })
        .count();

    tensorboard.scalar(
        format!("{}/stochastic_not_consistent_temp_1", tensorboard_prefix.clone()).as_str(),
        not_consistent_temp1 as f32 / temp_1_flattened.len() as f32,
        number_of_experiences as i64,
    );

    tensorboard.scalar(
        format!("{}/stochastic_not_consistent_temp_0", tensorboard_prefix.clone()).as_str(),
        not_consistent_temp0 as f32 / temp_0_flattened.len() as f32,
        number_of_experiences as i64,
    );

    for reason in NotConsistentReason::iter() {
        let temp1_count = temp_1_flattened
            .iter()
            .filter(|temp_1| {
                if let ForwardPredResult::NotConsistent(temp_1) = temp_1 {
                    if *temp_1 == reason {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .count();

        let temp0_count = temp_0_flattened
            .iter()
            .filter(|temp_0| {
                if let ForwardPredResult::NotConsistent(temp_0) = temp_0 {
                    if *temp_0 == reason {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .count();

        tensorboard.scalar(
            format!(
                "{}_ext/stochastic_not_consistent_temp_1_{:?}",
                tensorboard_prefix.clone(), reason
            )
            .as_str(),
            temp1_count as f32 / temp_1_flattened.len() as f32,
            number_of_experiences as i64,
        );

        tensorboard.scalar(
            format!(
                "{}_ext/stochastic_not_consistent_temp_0_{:?}",
                tensorboard_prefix.clone(), reason
            )
            .as_str(),
            temp0_count as f32 / temp_0_flattened.len() as f32,
            number_of_experiences as i64,
        );
    }

    pb.finish_with_message(
        format!(
            "Finished evaluation {} (at {})",
            tensorboard_prefix.clone(), number_of_experiences
        )
        .to_string(),
    );
}

pub async fn test_some(
    policies: PlayerZeroOrientedArr<Arc<dyn FullDokoPolicy>>,

    mut network: Box<dyn ImpiNetwork>,
) {
    let mut rng = SmallRng::from_os_rng();

    let mut state = FdoState::new_game(&mut rng);

    //
    // {
    //     network
    //         .load("./checkpoints/train_impi5/epoch_43.safetensors");
    // }

    let evaluate_net_fn: Arc<dyn EvaluateNetFn> =
        Arc::new(SingleNetworkEvaluateNetFn::new(network));

    loop {
        let obs = state.observation_for_current_player();

        if obs.finished_stats.is_some() {
            break;
        }

        let result = forward_pred_process(
            &state,
            0.5f32,
            1.0f32,
            &evaluate_net_fn.clone(),
            &mut rng,
            &MultiProgress::new(),
        )
        .await;

        let action = policies[obs.current_player.unwrap()]
            .execute_policy(&state, &obs, &mut rng)
            .await;

        state.play_action(action);
    }
}
