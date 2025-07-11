use array_concat::concat_arrays;
use rs_full_doko::basic::team::FdoTeam;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::game_type::game_type::FdoGameType;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::player::player_set::FdoPlayerSet;
use rs_full_doko::reservation::reservation::{FdoReservation, FdoVisibleReservation};
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::full_doko::var1::announcement::{encode_full_announcements, encode_lowest_announcements};
use crate::full_doko::var1::bool::encode_bool;
use crate::full_doko::var1::card::encode_card_or_none;
use crate::full_doko::var1::game_type::encode_game_type_or_none;
use crate::full_doko::var1::phase::encode_phase;
use crate::full_doko::var1::player::encode_player_or_none;
use crate::full_doko::var1::reservation::encode_reservation_or_none;
use crate::full_doko::var1::trick_method2::encode_tricks_method_2;
use crate::full_doko::var1::visible_reservation::encode_visible_reservation;
use crate::full_doko::var2::encode_annoucnement_team::encode_announcement_team;
use crate::full_doko::var2::encode_position_or_unknown::{encode_position_or_unknown_hand, encode_position_or_unknown_int};
use crate::full_doko::var2::encode_reservation_or_card_or_none::{encode_reservation_or_card_or_none_card, encode_reservation_or_card_or_pi_announcement, encode_reservation_or_card_or_reservation};
use crate::full_doko::var2::encode_subposition::{encode_subposition_card, encode_subposition_pos};

// Soviele Karten kann ein Spieler maximal auf der Hand haben
pub const NUM_MAX_CARDS_ON_HAND: i64 = 12;

fn reservation_to_visible_reservation(reservation: Option<FdoReservation>) -> FdoVisibleReservation {
    match reservation {
        None => FdoVisibleReservation::NoneYet,
        Some(reservation) => {
            match reservation {
                FdoReservation::Healthy => FdoVisibleReservation::Healthy,
                FdoReservation::Wedding => FdoVisibleReservation::Wedding,
                FdoReservation::DiamondsSolo => FdoVisibleReservation::DiamondsSolo,
                FdoReservation::HeartsSolo => FdoVisibleReservation::HeartsSolo,
                FdoReservation::SpadesSolo => FdoVisibleReservation::SpadesSolo,
                FdoReservation::ClubsSolo => FdoVisibleReservation::ClubsSolo,
                FdoReservation::QueensSolo => FdoVisibleReservation::QueensSolo,
                FdoReservation::JacksSolo => FdoVisibleReservation::JacksSolo,
                FdoReservation::TrumplessSolo => FdoVisibleReservation::TrumplessSolo
            }
        }
    }
}

pub fn encode_state_ipi(
    state: &FdoState,
    obs: &FdoObservation,

    assumed_hands: PlayerZeroOrientedArr<FdoHand>,
    assumed_reservations: PlayerZeroOrientedArr<Option<FdoReservation>>,
    next_player_to_chose_card_for: FdoPlayer
) -> [i64; 311] {
    let current_player = obs
        .current_player
        .unwrap_or(FdoPlayer::BOTTOM);

    let mut index = 0;

    let mut encoded_positions: heapless::Vec<i64, 62> = heapless::Vec::new();
    let mut encoded_reservations_cards: heapless::Vec<i64, 62> = heapless::Vec::new();
    let mut encoded_players_played: heapless::Vec<i64, 62> = heapless::Vec::new();
    let mut encoded_subpositions: heapless::Vec<i64, 62> = heapless::Vec::new();
    let mut encoded_teams: heapless::Vec<i64, 62> = heapless::Vec::new();

    for (player, reservation) in obs.visible_reservations.iter_with_player() {
        // Wenn bereits eine Vermutung besteht, was es sein konnte, dann wird diese
        // statt dem echten Wert genommen.
        let assumed_reservation = assumed_reservations[player];

        let reservation = if *reservation == FdoVisibleReservation::NotRevealed && assumed_reservation.is_some()  {
            reservation_to_visible_reservation(assumed_reservation)
        } else {
            *reservation
        };

        let encoded_position = encode_position_or_unknown_int(index);
        let encoded_reservation = encode_reservation_or_card_or_reservation(Some(reservation));
        let encoded_player_played = encode_player_or_none(Some(player), current_player);
        let encoded_subposition = encode_subposition_pos(None);
        let encoded_team = encode_announcement_team(None);

        encoded_positions.push(encoded_position[0]);
        encoded_reservations_cards.push(encoded_reservation[0]);
        encoded_players_played.push(encoded_player_played[0]);
        encoded_subpositions.push(encoded_subposition[0]);
        encoded_teams.push(encoded_team[0]);

        index += 1;
    }

    for trick in obs.tricks.iter() {
        for (player, card) in trick.cards.iter_with_player() {
            let encoded_position = encode_position_or_unknown_int(index);
            let encoded_card = encode_reservation_or_card_or_none_card(Some(*card));
            let encoded_player_played = encode_player_or_none(Some(player), current_player);
            let encoded_subposition = encode_subposition_pos(None);
            let encoded_team = encode_announcement_team(None);

            encoded_positions.push(encoded_position[0]);
            encoded_reservations_cards.push(encoded_card[0]);
            encoded_players_played.push(encoded_player_played[0]);
            encoded_subpositions.push(encoded_subposition[0]);
            encoded_teams.push(encoded_team[0]);

            index += 1;
        }
    }

    // Für den aktuellen Spieler alle Karten auf die Hand schreiben
    let hand = obs.phi_real_hands[current_player];

    let mut already_encoded_cards = FdoHand::empty();
    for card in hand.iter() {
        let encoded_position = encode_position_or_unknown_hand(Some(current_player), current_player);
        let encoded_card = encode_reservation_or_card_or_none_card(Some(card));
        let encoded_player_played = encode_player_or_none(None, current_player);
        let encoded_subposition = encode_subposition_card(if already_encoded_cards.contains(card) { Some(1) } else { Some(0) });
        let encoded_team = encode_announcement_team(None);

        encoded_positions.push(encoded_position[0]);
        encoded_reservations_cards.push(encoded_card[0]);
        encoded_players_played.push(encoded_player_played[0]);
        encoded_subpositions.push(encoded_subposition[0]);
        encoded_teams.push(encoded_team[0]);

        already_encoded_cards.add(card);
    }

    for (player, hand) in obs.phi_real_hands.rotate_to(current_player).iter_with_player() {
        if player == current_player {
            // Das Blatt vom aktuellen Spieler haben wir bereits
            continue;
        }

        let mut already_encoded_cards = FdoHand::empty();
        for card in &assumed_hands[player] {
            let encoded_position = encode_position_or_unknown_hand(Some(player), current_player);
            let encoded_card = encode_reservation_or_card_or_none_card(Some(card));
            let encoded_player_played = encode_player_or_none(None, current_player);
            let encoded_subposition = encode_subposition_card(if already_encoded_cards.contains(card) { Some(1) } else { Some(0) });
            let encoded_team = encode_announcement_team(None);

            encoded_positions.push(encoded_position[0]);
            encoded_reservations_cards.push(encoded_card[0]);
            encoded_players_played.push(encoded_player_played[0]);
            encoded_subpositions.push(encoded_subposition[0]);
            encoded_teams.push(encoded_team[0]);

            already_encoded_cards.add(card);
        }

        // Die Differenz zwischen der tatsächlichen Anzahl an Karten und der bisher erratenen Karten
        // als Unknown ausgeben
        let diff = hand.len() - assumed_hands[player].len();

        for _ in 0..diff {
            let encoded_position = encode_position_or_unknown_hand(Some(player), current_player);
            let encoded_card = encode_reservation_or_card_or_none_card(None);
            let encoded_player_played = encode_player_or_none(None, current_player);
            let encoded_subposition = encode_subposition_pos(None);
            let encoded_team = encode_announcement_team(None);

            encoded_positions.push(encoded_position[0]);
            encoded_reservations_cards.push(encoded_card[0]);
            encoded_players_played.push(encoded_player_played[0]);
            encoded_subpositions.push(encoded_subposition[0]);
            encoded_teams.push(encoded_team[0]);
        }
    }

    fn get_team_for_announcement(
        announcement_player: FdoPlayer,
        re_players: Option<FdoPlayerSet>
    ) -> FdoTeam {
        match re_players {
            None => panic!("not possible"),
            Some(re_players) => {
                return if re_players.contains(announcement_player) {
                    FdoTeam::Re
                } else {
                    FdoTeam::Kontra
                }
            }
        }
    }

    let mut subposition_index = 0;
    let mut last_position: i64 = -1;
    for announcement in obs.announcements.clone() {
        if announcement.card_index as i64 != last_position {
            last_position = announcement.card_index as i64;
            subposition_index = 0;
        }

        let encoded_position = encode_position_or_unknown_int(announcement.card_index);
        let encoded_announcement = encode_reservation_or_card_or_pi_announcement(Some(announcement.announcement));
        let encoded_player_played = encode_player_or_none(Some(announcement.player), current_player);
        let encoded_subposition = encode_subposition_pos(Some(subposition_index));
        let encoded_team = encode_announcement_team(
            Some(get_team_for_announcement(announcement.player, obs.phi_re_players))
        );

        encoded_positions.push(encoded_position[0]).unwrap();
        encoded_reservations_cards.push(encoded_announcement[0]).unwrap();
        encoded_players_played.push(encoded_player_played[0]).unwrap();
        encoded_subpositions.push(encoded_subposition[0]).unwrap();
        encoded_teams.push(encoded_team[0]).unwrap();

        subposition_index += 1;
    }

    for i in 0..(10 - obs.announcements.len()) {
        let encoded_position = encode_position_or_unknown_hand(None, current_player);
        let encoded_announcement = encode_reservation_or_card_or_pi_announcement(None);
        let encoded_player_played = encode_player_or_none(None, current_player);
        let encoded_subposition = encode_subposition_pos(None);
        let encoded_team = encode_announcement_team(None);

        encoded_positions.push(encoded_position[0]).unwrap();
        encoded_reservations_cards.push(encoded_announcement[0]).unwrap();
        encoded_players_played.push(encoded_player_played[0]).unwrap();
        encoded_subpositions.push(encoded_subposition[0]).unwrap();
        encoded_teams.push(encoded_team[0]).unwrap();
    }

    let encoded_player_to_guess_for = encode_player_or_none(Some(next_player_to_chose_card_for), current_player);

    // println!("encoded_positions: {:?}", encoded_positions.len());
    // println!("encoded_reservations_cards: {:?}", encoded_reservations_cards.len());
    // println!("encoded_player_played: {:?}", encoded_players_played.len());

    let encoded_positions: [i64; 62] = encoded_positions.as_slice().try_into().unwrap();
    let encoded_reservations_cards: [i64; 62] = encoded_reservations_cards.as_slice().try_into().unwrap();
    let encoded_player_played: [i64; 62] = encoded_players_played.as_slice().try_into().unwrap();
    let encoded_subpositions: [i64; 62] = encoded_subpositions.as_slice().try_into().unwrap();
    let encoded_teams: [i64; 62] = encoded_teams.as_slice().try_into().unwrap();
    let encoded_player_to_guess_for: [i64; 1] = encoded_player_to_guess_for;

    if cfg!(debug_assertions) {
        let mut counter = 0;

        macro_rules! debug_println {
            ($($arg:tt)*) => {
                println!($($arg)*);
            };
        }

        macro_rules! trace_indices {
            ($var:ident, $counter:expr) => {
                debug_println!(
                    "{}: [{}..{}] len: {}",
                    stringify!($var),
                    $counter,
                    $counter + $var.len(),
                    $var.len()
                );
                $counter += $var.len(); // Modify the passed counter
            };
        }

        trace_indices!(encoded_reservations_cards, counter);
        trace_indices!(encoded_positions, counter);
        trace_indices!(encoded_player_played, counter);
        trace_indices!(encoded_subpositions, counter);
        trace_indices!(encoded_teams, counter);
        trace_indices!(encoded_player_to_guess_for, counter);


    }

    return concat_arrays!(
        encoded_reservations_cards,
        encoded_positions,
        encoded_player_played,
        encoded_subpositions,
        encoded_teams,
        encoded_player_to_guess_for
    );
}


#[cfg(test)]
mod tests {
    use rs_full_doko::announcement::announcement::{FdoAnnouncementOccurrence, FdoAnnouncements};
    use rs_full_doko::announcement::announcement::FdoAnnouncement::{No30, No60, No90, ReContra};
    use rs_full_doko::announcement::announcement_set::FdoAnnouncementSet;
    use rs_full_doko::basic::phase::FdoPhase;
    use super::*;
    use rs_full_doko::card::cards::FdoCard::{ClubAce, ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondJack, DiamondKing, DiamondNine, DiamondQueen, DiamondTen, HeartAce, HeartJack, HeartKing, HeartNine, HeartQueen, HeartTen, SpadeAce, SpadeJack, SpadeKing, SpadeNine, SpadeQueen, SpadeTen};
    use rs_full_doko::display::display::display_game;
    use rs_full_doko::game_type::game_type::FdoGameType::ClubsSolo;
    use rs_full_doko::hand::hand::FdoHand;
    use rs_full_doko::reservation::reservation::{FdoReservation, FdoVisibleReservation};
    use rs_full_doko::reservation::reservation_round::FdoReservationRound;
    use rs_full_doko::reservation::reservation_winning_logic::FdoReservationResult::Solo;
    use rs_full_doko::team::team_logic::FdoTeamState::NoWedding;
    use rs_full_doko::trick::trick::FdoTrick;
    use rs_full_doko::util::po_vec::PlayerOrientedVec;
    use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
    use crate::full_doko::var1::announcement::encode_announcement;
    use crate::full_doko::var2::encode_reservation_or_card_or_none::encode_reservation_or_card_or_none_card;
    use crate::full_doko::var2::encode_subposition::encode_subposition_card;

    #[test]
    fn test_encode_state() {
        let state = FdoState {
            reservations_round: FdoReservationRound {
                reservations: PlayerOrientedVec::from_full(FdoPlayer::BOTTOM, vec![
                    FdoReservation::Healthy,
                    FdoReservation::ClubsSolo,
                    FdoReservation::DiamondsSolo,
                    FdoReservation::Healthy,
                ])
            },
            tricks: heapless::Vec::from_slice(&[
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::BOTTOM, vec![
                        SpadeAce,
                        ClubKing,
                        SpadeTen,
                        SpadeNine,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(ClubKing),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        SpadeQueen,
                        HeartJack,
                        ClubAce,
                        ClubKing,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(SpadeQueen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        ClubQueen,
                        SpadeJack,
                        ClubNine,
                        ClubTen,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(ClubQueen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        ClubQueen,
                        ClubJack,
                        DiamondQueen,
                        DiamondJack,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(ClubQueen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        HeartTen,
                        ClubTen,
                        HeartNine,
                        SpadeQueen,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(HeartTen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        HeartQueen,
                        ClubJack,
                        HeartNine,
                        DiamondNine,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(HeartQueen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        HeartTen,
                        ClubAce,
                        DiamondKing,
                        DiamondNine,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(HeartTen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        DiamondQueen,
                        HeartQueen,
                        DiamondTen,
                        DiamondTen,
                    ]),
                    winning_player: Some(FdoPlayer::TOP),
                    winning_card: Some(HeartQueen),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::TOP, vec![
                        DiamondAce,
                        SpadeKing,
                        DiamondKing,
                        ClubNine,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(ClubNine),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        SpadeJack,
                        SpadeTen,
                        SpadeKing,
                        SpadeNine,
                    ]),
                    winning_player: Some(FdoPlayer::LEFT),
                    winning_card: Some(SpadeJack),
                },
                FdoTrick {
                    cards: PlayerOrientedVec::from_full(FdoPlayer::LEFT, vec![
                        HeartJack,
                        SpadeAce,
                        HeartKing
                    ]),
                    winning_player: None,
                    winning_card: None,
                },
            ]).unwrap(),
            hands: PlayerZeroOrientedArr::from_full([
                FdoHand::from_vec(vec![HeartAce, HeartKing]),
                FdoHand::from_vec(vec![DiamondJack]),
                FdoHand::from_vec(vec![DiamondAce]),
                FdoHand::from_vec(vec![HeartAce]),
            ]),
            announcements: FdoAnnouncements {
                announcements: heapless::Vec::from_slice(&[
                    FdoAnnouncementOccurrence {
                        card_index: 4,
                        player: FdoPlayer::LEFT,
                        announcement: ReContra,
                    },
                    FdoAnnouncementOccurrence {
                        card_index: 8,
                        player: FdoPlayer::LEFT,
                        announcement: No90,
                    },
                    FdoAnnouncementOccurrence {
                        card_index: 12,
                        player: FdoPlayer::LEFT,
                        announcement: No60,
                    },
                    FdoAnnouncementOccurrence {
                        card_index: 16,
                        player: FdoPlayer::LEFT,
                        announcement: No30,
                    },
                    FdoAnnouncementOccurrence {
                        card_index: 17,
                        player: FdoPlayer::TOP,
                        announcement: ReContra,
                    },
                ]).unwrap(),
                re_lowest_announcement: Some(No30),
                contra_lowest_announcement: Some(ReContra),
                number_of_turns_without_announcement: 4,
                starting_player: FdoPlayer::BOTTOM,
                current_player_allowed_announcements: FdoAnnouncementSet::new(),
            },
            card_index: 43,
            current_player: Some(FdoPlayer::BOTTOM),
            current_phase: FdoPhase::PlayCard,
            reservation_result: Some(Solo(FdoPlayer::LEFT, FdoReservation::ClubsSolo)),
            game_type: Some(ClubsSolo),
            player_eyes: PlayerZeroOrientedArr::from_full([0, 158, 26, 0]),
            player_num_tricks: PlayerZeroOrientedArr::from_full([0, 9, 1, 0]),
            team_state: NoWedding {
                re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::LEFT]),
            },
            end_of_game_stats: None,
        };

        println!("{}", display_game(state.observation_for_current_player()));

        let result = encode_state_ipi(
            &state,
            &state.observation_for_current_player().clone(),
            PlayerZeroOrientedArr::from_full([
                FdoHand::from_vec(vec![HeartAce, HeartKing]),
                FdoHand::from_vec(vec![DiamondJack]),
                FdoHand::from_vec(vec![DiamondAce]),
                FdoHand::from_vec(vec![]),
            ]),
            PlayerZeroOrientedArr::from_full(
                [
                    Some(FdoReservation::Healthy),
                    Some(FdoReservation::ClubsSolo),
                    Some(FdoReservation::DiamondsSolo),
                    None,
                ]
            ),
            FdoPlayer::RIGHT,
        );

        println!("{:?}", result);


        let mut i = 0;
        macro_rules! assert_eq_inc {
            ($a:expr, $b:expr) => {
                println!("expected: {:?} at {:?} (was: {:?})", $b, i, &$a);
                assert_eq!($a, $b);
                i += 1;
            };
        }

        assert_eq!(encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM), [1]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM), [2]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM), [3]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM), [4]);


        // Karten der Kartenphase
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::Healthy)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::ClubsSolo)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::DiamondsSolo)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::Healthy)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeAce)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeNine)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubAce)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubKing)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubNine)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubTen)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondJack)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartNine)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeQueen)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartNine)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondNine)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubAce)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondNine)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartQueen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondTen)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondAce)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::ClubNine)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeTen)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeNine)));

        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartJack)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeAce)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartKing)));

        // Eigene Hand
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartKing)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::HeartAce)));

        // P1
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondJack)));

        // P2
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondAce)));

        // P3
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_none_card(None));

        // Announcements
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(Some(ReContra)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(Some(No90)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(Some(No60)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(Some(No30)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(Some(ReContra)));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(None));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(None));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(None));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(None));
        assert_eq_inc!(result[i..i+1], encode_reservation_or_card_or_pi_announcement(None));

        // Positionszuordnungen zu den Karten
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(0));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(1));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(2));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(3));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(4));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(5));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(6));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(7));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(8));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(9));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(10));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(11));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(12));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(13));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(14));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(15));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(16));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(17));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(18));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(19));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(20));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(21));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(22));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(23));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(24));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(25));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(26));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(27));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(28));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(29));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(30));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(31));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(32));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(33));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(34));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(35));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(36));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(37));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(38));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(39));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(40));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(41));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(42));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(43));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(44));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(45));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(46));

        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(4));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(8));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(12));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(16));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_int(17));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_position_or_unknown_hand(None, FdoPlayer::BOTTOM));

        // Spielerzuordungen zu den Karten
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));

        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));
        assert_eq_inc!(result[i..i+1], encode_player_or_none(None, FdoPlayer::BOTTOM));


        // Subpositionen
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));

        assert_eq_inc!(result[i..i+1], encode_subposition_card(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_card(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_card(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_card(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_card(None));

        assert_eq_inc!(result[i..i+1], encode_subposition_pos(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(Some(0)));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));
        assert_eq_inc!(result[i..i+1], encode_subposition_pos(None));

        // encode teams
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));

        assert_eq_inc!(result[i..i+1], encode_announcement_team(Some(FdoTeam::Re)));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(Some(FdoTeam::Re)));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(Some(FdoTeam::Re)));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(Some(FdoTeam::Re)));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(Some(FdoTeam::Kontra)));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));
        assert_eq_inc!(result[i..i+1], encode_announcement_team(None));

        // to guess for
        assert_eq_inc!(result[i..i+1], encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM));

        assert_eq!(i, 311);
    }


}