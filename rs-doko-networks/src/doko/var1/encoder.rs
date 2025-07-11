use rs_doko::card::cards::DoCard;
use rs_doko::hand::hand::{hand_to_vec, hand_to_vec_sorted_by_rank};
use rs_doko::observation::observation::DoObservation;
use rs_doko::player::player::{DoPlayer, PLAYER_BOTTOM};
use rs_doko::reservation::reservation::DoVisibleReservation;
use rs_doko::state::state::DoState;

/// Berechnet den relativen Index eines Zielspielers ausgehend vom aktuellen Spieler.
pub(crate) fn get_relative_player_index(current_player: DoPlayer, target_player: DoPlayer) -> usize {
    (target_player as usize + 4 - current_player as usize) % 4
}

fn card_to_rank_in_normal_game(card: DoCard) -> i32 {
    match card {
        DoCard::HeartNine => 0,
        DoCard::HeartKing => 1,
        DoCard::HeartAce => 2,

        DoCard::SpadeNine => 3,
        DoCard::SpadeKing => 4,
        DoCard::SpadeTen => 5,
        DoCard::SpadeAce => 6,

        DoCard::ClubNine => 7,
        DoCard::ClubKing => 8,
        DoCard::ClubTen => 9,
        DoCard::ClubAce => 10,

        DoCard::DiamondNine => 11,
        DoCard::DiamondKing => 12,
        DoCard::DiamondTen => 13,
        DoCard::DiamondAce => 14,

        DoCard::DiamondJack => 15,
        DoCard::HeartJack => 16,
        DoCard::SpadeJack => 17,
        DoCard::ClubJack => 18,

        DoCard::DiamondQueen => 19,
        DoCard::HeartQueen => 20,
        DoCard::SpadeQueen => 21,
        DoCard::ClubQueen => 22,

        DoCard::HeartTen => 23,
    }
}

pub fn encode_card_or_none(card: Option<DoCard>) -> i64 {
    match card {
        None => 0,
        Some(card) => {
            (match card {
                DoCard::DiamondNine => 1,
                DoCard::DiamondTen => 2,
                DoCard::DiamondJack => 3,
                DoCard::DiamondQueen => 4,
                DoCard::DiamondKing => 5,
                DoCard::DiamondAce => 6,
                DoCard::HeartNine => 7,
                DoCard::HeartTen => 8,
                DoCard::HeartJack => 9,
                DoCard::HeartQueen => 10,
                DoCard::HeartKing => 11,
                DoCard::HeartAce => 12,
                DoCard::ClubNine => 13,
                DoCard::ClubTen => 14,
                DoCard::ClubJack => 15,
                DoCard::ClubQueen => 16,
                DoCard::ClubKing => 17,
                DoCard::ClubAce => 18,
                DoCard::SpadeNine => 19,
                DoCard::SpadeTen => 20,
                DoCard::SpadeJack => 21,
                DoCard::SpadeQueen => 22,
                DoCard::SpadeKing => 23,
                DoCard::SpadeAce => 24,
            })
        }
    }
}

pub(crate) fn encode_relative_player(player: Option<usize>) -> i64 {
    match player {
        None => 0,
        Some(player) => player as i64 + 1,
    }
}

pub(crate) fn encode_visible_reservation(reservation: Option<DoVisibleReservation>) -> i64 {
    match reservation {
        None => 0,
        Some(reservation) => match reservation {
            DoVisibleReservation::NotRevealed => 1,
            DoVisibleReservation::Healthy => 2,
            DoVisibleReservation::Wedding => 3,
        },
    }
}
pub fn encode_state_with_reservations_copy(obs: &DoObservation, memory: &mut [i64]) {
    let phase_num = obs.phase as usize;

    let current_player = obs.current_player.unwrap_or(PLAYER_BOTTOM);

    let reservation_start_player =
        get_relative_player_index(current_player, obs.game_starting_player) + 1;

    let mut trick_starting_players: [Option<usize>; 12] = [None; 12];

    for (i, trick) in obs.tricks.iter().enumerate() {
        trick_starting_players[i] =
            trick.map(|t| get_relative_player_index(current_player, t.start_player));
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

    let mut cards_on_hands: [Option<DoCard>; 4 * 12] = [None; 4 * 12];

    for i in 0..4 {
        let current_hand = obs.phi_real_hands[i];

        let i = get_relative_player_index(current_player, i);

        for j in 0..12 {
            let mut hand_as_vec = hand_to_vec_sorted_by_rank(current_hand);

            cards_on_hands[i * 12 + j] = hand_as_vec.get(j).copied();
        }
    }

    let mut reservations: [Option<DoVisibleReservation>; 4] = [None; 4];

    for i in 0..4 {
        reservations[i] = obs.visible_reservations[i];
    }

    let encoded_trick_starting_players: Vec<i64> = trick_starting_players
        .iter()
        .map(|&player| encode_relative_player(player))
        .collect();

    let encoded_trick_cards: Vec<i64> = trick_cards
        .iter()
        .map(|&card| encode_card_or_none(card))
        .collect();

    let encoded_hand_cards: Vec<i64> = cards_on_hands
        .iter()
        .map(|&card| encode_card_or_none(card))
        .collect();

    let encoded_reservations: Vec<i64> = reservations
        .iter()
        .map(|&reservation| encode_visible_reservation(reservation))
        .collect();

    memory[0] = phase_num as i64;
    memory[1] = reservation_start_player as i64;
    memory[2..14].copy_from_slice(&encoded_trick_starting_players);
    memory[14..62].copy_from_slice(&encoded_trick_cards);
    memory[62..110].copy_from_slice(&encoded_hand_cards);
    memory[110..114].copy_from_slice(&encoded_reservations);
}

pub fn encode_state_with_reservations(state: &DoState) -> Vec<i64> {
    let obs = state.observation_for_current_player();

    let mut memory = vec![0; 114];

    encode_state_with_reservations_copy(&obs, &mut memory);

    memory
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;
    use rs_doko::basic::phase::DoPhase;
    use rs_doko::hand::hand::DoHand;
    use rs_doko::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};

    #[test]
    fn test_rotate_player() {
        assert_eq!(get_relative_player_index(PLAYER_BOTTOM, PLAYER_BOTTOM), 0);
        assert_eq!(get_relative_player_index(PLAYER_BOTTOM, PLAYER_LEFT), 1);
        assert_eq!(get_relative_player_index(PLAYER_BOTTOM, PLAYER_TOP), 2);
        assert_eq!(get_relative_player_index(PLAYER_BOTTOM, PLAYER_RIGHT), 3);

        assert_eq!(get_relative_player_index(PLAYER_LEFT, PLAYER_BOTTOM), 3);
        assert_eq!(get_relative_player_index(PLAYER_LEFT, PLAYER_LEFT), 0);
        assert_eq!(get_relative_player_index(PLAYER_LEFT, PLAYER_TOP), 1);
        assert_eq!(get_relative_player_index(PLAYER_LEFT, PLAYER_RIGHT), 2);

        assert_eq!(get_relative_player_index(PLAYER_TOP, PLAYER_BOTTOM), 2);
        assert_eq!(get_relative_player_index(PLAYER_TOP, PLAYER_LEFT), 3);
        assert_eq!(get_relative_player_index(PLAYER_TOP, PLAYER_TOP), 0);
        assert_eq!(get_relative_player_index(PLAYER_TOP, PLAYER_RIGHT), 1);

        assert_eq!(get_relative_player_index(PLAYER_RIGHT, PLAYER_BOTTOM), 1);
        assert_eq!(get_relative_player_index(PLAYER_RIGHT, PLAYER_LEFT), 2);
        assert_eq!(get_relative_player_index(PLAYER_RIGHT, PLAYER_TOP), 3);
        assert_eq!(get_relative_player_index(PLAYER_RIGHT, PLAYER_RIGHT), 0);
    }

    #[test]
    fn test_encode_random_state() {
        let mut random = SmallRng::seed_from_u64(0);

        let mut doko = DoState::new_game(&mut random);

        for i in 0..15 {
            doko.random_action_for_current_player(&mut random);
        }

        let encoded = encode_state_with_reservations(&doko);

        fn print_hand(do_hand: DoHand) {
            let mut hand = hand_to_vec(do_hand);

            hand.sort_by_key(|&card| -card_to_rank_in_normal_game(card));

            println!("{:?}", hand);
        }

        assert_eq!(
            encoded,
            vec![
                // 0 : Phase (PlayCard)
                DoPhase::PlayCard as i64,
                // 1 : Startspieler des Spiels (relativ zu aktuellem Spieler PLAYER_TOP)
                (get_relative_player_index(PLAYER_TOP, PLAYER_LEFT) + 1) as i64,
                // 2 : Startspieler des ersten Stichs (relativ zu aktuellem Spieler PLAYER_TOP)
                (get_relative_player_index(PLAYER_TOP, PLAYER_LEFT) + 1) as i64,
                // 3 : Startspieler des zweiten Stichs (relativ zu aktuellem Spieler PLAYER_TOP)
                (get_relative_player_index(PLAYER_TOP, PLAYER_TOP) + 1) as i64,
                // 4 : Startspieler des dritten Stichs (hier: niemand)
                (get_relative_player_index(PLAYER_TOP, PLAYER_RIGHT) + 1) as i64,
                // 5 : Startspieler des vierten Stichs (hier: niemand)
                0,
                // 6 : Startspieler des fünften Stichs (hier: niemand)
                0,
                // 7 : Startspieler des sechsten Stichs (hier: niemand)
                0,
                // 8 : Startspieler des siebten Stichs (hier: niemand)
                0,
                // 9 : Startspieler des achten Stichs (hier: niemand)
                0,
                // 10 : Startspieler des neunten Stichs (hier: niemand)
                0,
                // 11 : Startspieler des zehnten Stichs (hier: niemand)
                0,
                // 12 : Startspieler des elften Stichs (hier: niemand)
                0,
                // 13 : Startspieler des zwölften Stichs (hier: niemand)
                0,
                // 14: 1. gelegte Karte
                encode_card_or_none(Some(DoCard::SpadeKing)),
                // 15: 2. gelegete Karte
                encode_card_or_none(Some(DoCard::SpadeAce)),
                // 16: 3. gelegte Karte
                encode_card_or_none(Some(DoCard::SpadeAce)),
                // 17: 4. gelegte Karte
                encode_card_or_none(Some(DoCard::SpadeNine)),
                // 18: 5. gelegte Karte
                encode_card_or_none(Some(DoCard::HeartNine)),
                // 19: 6. gelegte Karte
                encode_card_or_none(Some(DoCard::SpadeJack)),
                // 20: 7. gelegte Karte
                encode_card_or_none(Some(DoCard::HeartAce)),
                // 21: 8. gelegte Karte
                encode_card_or_none(Some(DoCard::HeartAce)),
                // 22: 9. gelegte Karte
                encode_card_or_none(Some(DoCard::ClubTen)),
                // 23: 10. gelegte Karte
                encode_card_or_none(Some(DoCard::SpadeJack)),
                // 24: 11. gelegte Karte
                encode_card_or_none(Some(DoCard::ClubKing)),
                // 25: 12. gelegte Karte
                0,
                // 26: 13. gelegte Karte
                0,
                // 27: 14. gelegte Karte
                0,
                // 28: 15. gelegte Karte
                0,
                // 29: 16. gelegte Karte
                0,
                // 30: 17. gelegte Karte
                0,
                // 31: 18. gelegte Karte
                0,
                // 32: 19. gelegte Karte
                0,
                // 33: 20. gelegte Karte
                0,
                // 34: 21. gelegte Karte
                0,
                // 35: 22. gelegte Karte
                0,
                // 36: 23. gelegte Karte
                0,
                // 37: 24. gelegte Karte
                0,
                // 38: 25. gelegte Karte
                0,
                // 39: 26. gelegte Karte
                0,
                // 40: 27. gelegte Karte
                0,
                // 41: 28. gelegte Karte
                0,
                // 42: 29. gelegte Karte
                0,
                // 43: 30. gelegte Karte
                0,
                // 44: 31. gelegte Karte
                0,
                // 45: 32. gelegte Karte
                0,
                // 46: 33. gelegte Karte
                0,
                // 47: 34. gelegte Karte
                0,
                // 48: 35. gelegte Karte
                0,
                // 49: 36. gelegte Karte
                0,
                // 50: 37. gelegte Karte
                0,
                // 51: 38. gelegte Karte
                0,
                // 52: 39. gelegte Karte
                0,
                // 53: 40. gelegte Karte
                0,
                // 54: 41. gelegte Karte
                0,
                // 55: 42. gelegte Karte
                0,
                // 56: 43. gelegte Karte
                0,
                // 57: 44. gelegte Karte
                0,
                // 58: 45. gelegte Karte
                0,
                // 59: 46. gelegte Karte
                0,
                // 60: 47. gelegte Karte
                0,
                // 61: 48. gelegte Karte
                0,
                // 62: 1. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::HeartTen)),
                // 63: 2. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::ClubQueen)),
                // 64: 3. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::DiamondQueen)),
                // 65: 4. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::DiamondJack)),
                // 66: 5. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::ClubAce)),
                // 67: 6. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::ClubAce)),
                // 68: 7. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::SpadeTen)),
                // 69: 8. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::SpadeTen)),
                // 70: 9. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::SpadeNine)),
                // 71: 10. Handkarte des aktuellen Spielers
                encode_card_or_none(Some(DoCard::HeartNine)),
                // 72: 11. Handkarte des aktuellen Spielers
                0,
                // 73: 12. Handkarte des aktuellen Spielers
                0,
                // 74: 1. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::SpadeQueen)),
                // 75: 2. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::SpadeQueen)),
                // 76: 3. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::HeartQueen)),
                // 77: 4. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::DiamondQueen)),
                // 78: 5. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::DiamondJack)),
                // 79: 6. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::DiamondKing)),
                // 80: 7. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::ClubKing)),
                // 81: 8. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::ClubNine)),
                // 82: 9. Handkarte des linken Spielers
                encode_card_or_none(Some(DoCard::ClubNine)),
                // 83: 10. Handkarte des linken Spielers
                0,
                // 84: 11. Handkarte des linken Spielers
                0,
                // 85: 12. Handkarte des linken Spielers
                0,
                // 86: 1. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::ClubJack)),
                // 87: 2. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::ClubJack)),
                // 88: 3. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::DiamondTen)),
                // 89: 4. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::DiamondTen)),
                // 90: 5. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::DiamondKing)),
                // 91: 6. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::DiamondNine)),
                // 92: 7. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::DiamondNine)),
                // 93: 8. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::HeartKing)),
                // 94: 9. Handkarte des oberen Spielers
                encode_card_or_none(Some(DoCard::HeartKing)),
                // 95: 10. Handkarte des oberen Spielers
                0,
                // 96: 11. Handkarte des oberen Spielers
                0,
                // 97: 12. Handkarte des oberen Spielers
                0,
                // 98: 1. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::HeartTen)),
                // 99: 2. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::ClubQueen)),
                // 100: 3. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::HeartQueen)),
                // 101: 4. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::HeartJack)),
                // 102: 5. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::HeartJack)),
                // 103: 6. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::DiamondAce)),
                // 104: 7. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::DiamondAce)),
                // 105: 8. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::ClubTen)),
                // 106: 9. Handkarte des rechten Spielers
                encode_card_or_none(Some(DoCard::SpadeKing)),
                // 107: 10. Handkarte des rechten Spielers
                0,
                // 108: 11. Handkarte des rechten Spielers
                0,
                // 109: 12. Handkarte des rechten Spielers
                0,
                // 110: Vorbehalt 1
                encode_visible_reservation(Some(DoVisibleReservation::Healthy)),
                // 111: Vorbehalt 2
                encode_visible_reservation(Some(DoVisibleReservation::Healthy)),
                // 112: Vorbehalt 3
                encode_visible_reservation(Some(DoVisibleReservation::Healthy)),
                // 113: Vorbehalt 4
                encode_visible_reservation(Some(DoVisibleReservation::Healthy)),
            ]
        )
    }
}
