use std::fmt::DebugList;
use rs_doko::card::cards::DoCard;
use rs_doko::hand::hand::{hand_contains, hand_contains_both, hand_to_vec_sorted_by_rank};
use rs_doko::observation::observation::DoObservation;
use rs_doko::player::player::{player_wraparound, PLAYER_BOTTOM};
use rs_doko::reservation::reservation::DoVisibleReservation;
use rs_doko::state::state::DoState;
use rs_doko::util::RotArr::RotArr;
use crate::doko::var1::encoder::{encode_card_or_none, encode_state_with_reservations};

macro_rules! encode_card {
    ($hand:expr, $card:expr, $index:expr, $encoded_hand:expr, $pos:expr) => {
        if hand_contains_both($hand, $card) {
            $encoded_hand[($index - 1) * 48 + $pos] = 1;
            $encoded_hand[($index - 1) * 48 + $pos + 1] = 1;
        } else if hand_contains($hand, $card) {
            $encoded_hand[($index - 1) * 48 + $pos] = 1;
        }
    };
}

fn encode_hands(state: &DoState) -> [i64; 144] {
    let obs = state.observation_for_current_player();

    let current_player = obs.current_player.unwrap();

    let mut hot_one_encoded_hand: [i64; 48 * 3] = [0; 48 * 3];

    let rot_hands = RotArr::new_from_0(0, state.hands)
        .rotate_to_perspective(current_player)
        .extract();

    for i in 1..4 {
        let hand = rot_hands[i];

        encode_card!(hand, DoCard::HeartTen, i, hot_one_encoded_hand, 0);
        encode_card!(hand, DoCard::ClubQueen, i, hot_one_encoded_hand, 2);
        encode_card!(hand, DoCard::SpadeQueen, i, hot_one_encoded_hand, 4);
        encode_card!(hand, DoCard::HeartQueen, i, hot_one_encoded_hand, 6);
        encode_card!(hand, DoCard::DiamondQueen, i, hot_one_encoded_hand, 8);
        encode_card!(hand, DoCard::ClubJack, i, hot_one_encoded_hand, 10);
        encode_card!(hand, DoCard::SpadeJack, i, hot_one_encoded_hand, 12);
        encode_card!(hand, DoCard::HeartJack, i, hot_one_encoded_hand, 14);
        encode_card!(hand, DoCard::DiamondJack, i, hot_one_encoded_hand, 16);
        encode_card!(hand, DoCard::DiamondAce, i, hot_one_encoded_hand, 18);
        encode_card!(hand, DoCard::DiamondTen, i, hot_one_encoded_hand, 20);
        encode_card!(hand, DoCard::DiamondKing, i, hot_one_encoded_hand, 22);
        encode_card!(hand, DoCard::DiamondNine, i, hot_one_encoded_hand, 24);
        encode_card!(hand, DoCard::ClubAce, i, hot_one_encoded_hand, 26);
        encode_card!(hand, DoCard::ClubTen, i, hot_one_encoded_hand, 28);
        encode_card!(hand, DoCard::ClubKing, i, hot_one_encoded_hand, 30);
        encode_card!(hand, DoCard::ClubNine, i, hot_one_encoded_hand, 32);
        encode_card!(hand, DoCard::SpadeAce, i, hot_one_encoded_hand, 34);
        encode_card!(hand, DoCard::SpadeTen, i, hot_one_encoded_hand, 36);
        encode_card!(hand, DoCard::SpadeKing, i, hot_one_encoded_hand, 38);
        encode_card!(hand, DoCard::SpadeNine, i, hot_one_encoded_hand, 40);
        encode_card!(hand, DoCard::HeartAce, i, hot_one_encoded_hand, 42);
        encode_card!(hand, DoCard::HeartKing, i, hot_one_encoded_hand, 44);
        encode_card!(hand, DoCard::HeartNine, i, hot_one_encoded_hand, 46);
    }

    return hot_one_encoded_hand;
}

fn encode_information_set(
    obs: &DoObservation,
    memory: &mut [i64]
) {
    let phase_num = obs.phase as usize;

    let current_player = obs.current_player.unwrap_or(PLAYER_BOTTOM);

    let reservation_start_player =
        crate::doko::var1::encoder::get_relative_player_index(current_player, obs.game_starting_player) + 1;

    let mut trick_starting_players: [Option<usize>; 12] = [None; 12];

    for (i, trick) in obs.tricks.iter().enumerate() {
        trick_starting_players[i] =
            trick.map(|t| crate::doko::var1::encoder::get_relative_player_index(current_player, t.start_player));
    }

    let mut trick_cards: [Option<DoCard>; 12 * 4] = [None; 12 * 4];

    let mut card_index = 0;

    for trick in obs.tricks.iter() {
        match trick {
            None => break,
            Some(trick) => {
                for card in trick.cards.iter() {
                    match card {
                        None => {
                            break;
                        }
                        Some(card) => {
                            trick_cards[card_index] = Some(*card);
                            card_index += 1;
                        }
                    }
                }
            }
        }
    }

    let mut own_cards_on_hand: [Option<DoCard>; 12] = [None; 12];

    let current_hand = obs.observing_player_hand;

    for j in 0..12 {
        let mut hand_as_vec = hand_to_vec_sorted_by_rank(current_hand);

        own_cards_on_hand[j] = hand_as_vec.get(j).copied();
    }

    let mut reservations: [Option<DoVisibleReservation>; 4] = [None; 4];

    for i in 0..4 {
        reservations[i] = obs.visible_reservations[i];
    }

    let encoded_trick_starting_players: Vec<i64> = trick_starting_players
        .iter()
        .map(|&player| crate::doko::var1::encoder::encode_relative_player(player))
        .collect();

    let encoded_trick_cards: Vec<i64> = trick_cards
        .iter()
        .map(|&card| encode_card_or_none(card))
        .collect();

    let encoded_hand_cards: Vec<i64> = own_cards_on_hand
        .iter()
        .map(|&card| encode_card_or_none(card))
        .collect();

    let encoded_reservations: Vec<i64> = reservations
        .iter()
        .map(|&reservation| crate::doko::var1::encoder::encode_visible_reservation(reservation))
        .collect();

    memory[0] = phase_num as i64;
    memory[1] = reservation_start_player as i64;
    memory[2..14].copy_from_slice(&encoded_trick_starting_players);
    memory[14..62].copy_from_slice(&encoded_trick_cards);
    memory[62..74].copy_from_slice(&encoded_hand_cards);
    memory[74..78].copy_from_slice(&encoded_reservations);
}

pub fn encode_information_set_m(
    doko: &DoState
) -> [i64; 78] {
    let mut memory: [i64; 78] = [0; 78];

    let obs = doko.observation_for_current_player();

    encode_information_set(&obs, &mut memory);

    return memory;
}


/// Diese Methoden kodiert einen Spielzustand für das GAN, welches aus einem Spielzustand
/// die möglichen
pub fn encode_for_impi_gan(
    doko: &DoState,
) -> ([i64; 78], [i64; 144]) {
    let information_set = encode_information_set_m(&doko);
    let remaining_state = encode_hands(&doko);

    (information_set, remaining_state)
}
