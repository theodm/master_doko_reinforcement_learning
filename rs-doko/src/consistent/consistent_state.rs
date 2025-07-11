use enumset::EnumSet;
use crate::basic::color::DoColor;
use crate::card::card_to_color::card_to_color_in_normal_game;
use crate::card::cards::DoCard;
use crate::hand::hand::{hand_contains, hand_len, hand_to_vec, hand_to_vec_sorted_by_rank, DoHand};
use crate::player::player::{player_wraparound, DoPlayer};
use crate::player::player_set::player_set_contains;
use crate::teams::team_logic::DoTeamState;
use crate::trick::trick::DoTrick;

impl DoTrick {
    pub fn cards_with_players(&self) -> impl Iterator<Item = (DoPlayer, DoCard)> + '_ {
        self.cards
            .iter()
            .enumerate()
            .filter_map(move |(i, &maybe_card)| {
                maybe_card.map(|card| {
                    let player = player_wraparound(self.start_player + i);

                    (player, card)
                })
            })
    }
}

fn gather_possible_colors(
    previous_tricks: heapless::Vec<DoTrick, 12>,
) -> [EnumSet<DoColor>; 4] {
    // Prinzipiell gehen wir davon aus, dass wir kein Wissen darüber haben, wer welche Farben hat. Anders
    // gesagt gehen wir davon aus, dass jeder Spieler jede Farbe haben kann.
    let all_colors = DoColor::Trump | DoColor::Spade | DoColor::Heart | DoColor::Club;

    let mut possible_colors = [all_colors; 4];

    for trick in previous_tricks.iter() {
        let trick_color = trick.color();

        let trick_color = match trick_color {
            None => { continue }
            Some(trick_color) => {trick_color}
        };

        for (player, card) in trick.cards_with_players() {
            if card_to_color_in_normal_game(card) != trick_color {
                possible_colors[player].remove(trick_color);
            }
        }

    }

    return possible_colors;
}

pub fn are_hands_consistent(
    team_state: DoTeamState,
    previous_tricks: heapless::Vec<DoTrick, 12>,

    real_hands: [DoHand; 4],

    hands: [DoHand; 4]
) -> bool {
    // Folgende Dinge können wir immer ableiten:
    //
    // Ein Spieler hat Trumpf nicht bekannt. -> Er hat (von den verbleibenden Karten) keinen Trumpf.
    // Ein Spieler hat Herz nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Herz.
    // Ein Spieler hat Pik nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Pik.
    // Ein Spieler hat Kreuz nicht bekannt. -> Er hat (von den verbleibenden Karten) keine Kreuz.
    // Ein Spieler hat eine Hochzeit angesagt. -> Er hat beide Kreuz-Damen, alle anderen haben keine Kreuz-Dame.


    // Wenn die Anzahl der Karten in den Händen nicht übereinstimmt, sind die Hände nicht konsistent.
    for i in 0..4 {
        if hand_len(hands[i]) != hand_len(real_hands[i]) {
            return false;
        }
    }

    // Wenn jemand eine Karte auf der Hand hat, die es nicht mehr gibt, ist das Spiel nicht konsistent.
    let mut remaining_cards = hand_to_vec(real_hands[0]);
    remaining_cards.append(&mut hand_to_vec(real_hands[1]));
    remaining_cards.append(&mut hand_to_vec(real_hands[2]));
    remaining_cards.append(&mut hand_to_vec(real_hands[3]));

    for i in 0..4 {
        let hand = hand_to_vec(hands[i]);

        for card in hand {
            if !remaining_cards.contains(&card) {
                return false;
            }

            remaining_cards.remove(remaining_cards.iter().position(|&x| x == card).unwrap());
        }
    }

    if !remaining_cards.is_empty() {
        return false;
    }

    // Wenn jemand eine Farbe hat, auf die er bereits abgeworfen hat, ist das Spiel nicht konsistent.
    let possible_colors = gather_possible_colors(previous_tricks);

    for i in 0..4 {
        let hand = hand_to_vec(hands[i]);

        for card in hand {
            let color = card_to_color_in_normal_game(card);

            if !possible_colors[i].contains(color) {
                return false;
            }
        }
    }

    // Wenn jemand eine Hochzeit angesagt hat, muss er beide Kreuz-Damen haben.
    let wedding_player = match team_state {
        DoTeamState::WeddingUnsolved { wedding_player } => Some(wedding_player),
        DoTeamState::WeddingSolved { wedding_player, .. } => Some(wedding_player),
        _ => None
    };

    // Andere Spieler dürfen dann keine Kreuz-Dame haben
    if let Some(wedding_player) = wedding_player {
        for i in 0..4 {
            if i == wedding_player {
                continue
            }

            if hand_contains(hands[i], DoCard::ClubQueen) {
                return false;
            }
        }
    }

    return true;
}

#[cfg(test)]
mod tests {
    use crate::card::cards::DoCard::{ClubAce, ClubJack, DiamondAce, DiamondQueen, DiamondTen, HeartJack, HeartQueen};
    use crate::hand::hand::hand_from_vec;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use crate::player::player_set::{player_set_add, player_set_create};
    use super::*;

    #[test]
    fn test_gather_possible_colors() {
        let previous_tricks = heapless::Vec::from_slice(&[
            // Left hat Kreuz nicht bedient.
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::ClubAce,
                DoCard::DiamondNine,
                DoCard::ClubNine,
                DoCard::ClubKing
            ]),
            // Top hat Herz nicht bedient
            DoTrick::existing(PLAYER_LEFT, vec![
                DoCard::HeartAce,
                DoCard::DiamondNine,
                DoCard::HeartNine,
                DoCard::HeartKing
            ]),
            // Right hat Pik nicht bedient
            DoTrick::existing(PLAYER_TOP, vec![
                DoCard::SpadeAce,
                DoCard::DiamondNine,
                DoCard::SpadeNine,
                DoCard::SpadeKing
            ]),
            // Left hat Trumpf nicht bedient
            DoTrick::existing(PLAYER_RIGHT, vec![
                DoCard::DiamondAce,
                DoCard::ClubJack,
                DoCard::SpadeNine,
                DoCard::DiamondQueen
            ]),
            // Offener Stich, muss ignoriert werden
            DoTrick::empty(PLAYER_RIGHT)
        ]).unwrap();

        let possible_colors = gather_possible_colors(previous_tricks);

        assert_eq!(possible_colors[PLAYER_BOTTOM], DoColor::Heart | DoColor::Spade | DoColor::Trump | DoColor::Club);
        assert_eq!(possible_colors[PLAYER_LEFT], DoColor::Spade | DoColor::Heart);
        assert_eq!(possible_colors[PLAYER_TOP], DoColor::Club | DoColor::Trump | DoColor::Spade);
        assert_eq!(possible_colors[PLAYER_RIGHT], DoColor::Club | DoColor::Heart | DoColor::Trump);
    }

    #[test]
    fn test_correct() {
        // https://www.online-doppelkopf.com/spiele/99.642.551
        let team_state = DoTeamState::NoWedding {
            re_players: player_set_create(vec![PLAYER_BOTTOM, PLAYER_TOP]),
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeKing,
                DoCard::SpadeTen,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine
            ]),
            // Bottom -> 10 Karten
            // Left -> 10 Karten
            // Top -> 11 Karten
            // Right -> 11 Karten
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        let hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), true);
    }


    #[test]
    fn test_possible() {
        // https://www.online-doppelkopf.com/spiele/99.642.551
        let team_state = DoTeamState::NoWedding {
            re_players: player_set_create(vec![PLAYER_BOTTOM, PLAYER_TOP]),
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeKing,
                DoCard::SpadeTen,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine
            ]),
            // Bottom -> 10 Karten
            // Left -> 10 Karten
            // Top -> 11 Karten
            // Right -> 11 Karten
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        // Statt Spieler RIGHT könnte auch Spieler TOP beide Kreuz-Dame haben.
        let hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubQueen, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubJack, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), true);
    }

    #[test]
    fn test_fail_wrong_number_of_cards() {
        // https://www.online-doppelkopf.com/spiele/99.642.551 (mit einem Fehler)
        let team_state = DoTeamState::NoWedding {
            re_players: player_set_create(vec![PLAYER_BOTTOM, PLAYER_TOP]),
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeKing,
                DoCard::SpadeTen,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine
            ]),
            // Bottom -> 10 Karten
            // Left -> 10 Karten
            // Top -> 11 Karten
            // Right -> 11 Karten
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        // Right fehlt eine Karte: DoCard::DiamondAce
        let hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), false);
    }

    #[test]
    fn test_fail_not_available_card_played_three_times_in_game() {
        // https://www.online-doppelkopf.com/spiele/99.642.551 (mit einem Fehler)
        let team_state = DoTeamState::NoWedding {
            re_players: player_set_create(vec![PLAYER_BOTTOM, PLAYER_TOP]),
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeKing,
                DoCard::SpadeTen,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine
            ]),
            // Bottom -> 10 Karten
            // Left -> 10 Karten
            // Top -> 11 Karten
            // Right -> 11 Karten
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::ClubTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        // Bei Bottom wurde DoCard::ClubTen durch DoCard::SpadeTen ersetzt. (Damit existiert DoCard::SpadeTen dreimal im Spiel und DoCard::ClubTen nur einmal)
        let hands = [
            hand_from_vec(vec![DoCard::SpadeTen, DoCard::DiamondNine, DoCard::SpadeJack, DoCard::ClubKing, DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::ClubAce, DoCard::DiamondNine, DoCard::ClubJack, DoCard::ClubNine, DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::HeartNine, DoCard::ClubNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::ClubKing, DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::HeartKing, DoCard::ClubAce, DoCard::ClubQueen, DoCard::HeartJack, DoCard::DiamondKing, DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), false);
    }


    #[test]
    fn test_fail_already_abgeworfen() {
        // https://www.online-doppelkopf.com/spiele/99.642.551 (mit einem Fehler)
        let team_state = DoTeamState::NoWedding {
            re_players: player_set_create(vec![PLAYER_BOTTOM, PLAYER_TOP]),
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeKing,
                DoCard::SpadeTen,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine,
                DoCard::HeartNine,
                DoCard::HeartKing
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::ClubTen,
                DoCard::ClubAce,
                DoCard::ClubNine,
                DoCard::ClubAce
            ]),
            DoTrick::existing(PLAYER_LEFT, vec![
                DoCard::DiamondNine,
                DoCard::DiamondJack,
                DoCard::ClubQueen,
                DoCard::DiamondNine
            ]),
            DoTrick::existing(PLAYER_RIGHT, vec![
                DoCard::HeartJack,
                DoCard::SpadeJack,
                DoCard::ClubJack,
                DoCard::DiamondQueen
            ]),

            DoTrick::existing(PLAYER_TOP, vec![
                DoCard::ClubKing,
                DoCard::DiamondKing,
                DoCard::ClubKing,
                DoCard::ClubNine
            ]),
            // Bottom -> 6 Karten
            // Left -> 6 Karten
            // Top -> 6 Karten
            // Right -> 6 Karten (hat kein Kreuz mehr)
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::ClubTen, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::SpadeKing, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        // Right hat noch die Kreuz-10 von TOP (Tausch mit Pik König)
        let hands = [
            hand_from_vec(vec![DoCard::DiamondAce, DoCard::DiamondJack, DoCard::DiamondTen, DoCard::DiamondQueen, DoCard::HeartKing]),
            hand_from_vec(vec![DoCard::SpadeAce, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubJack, DoCard::ClubQueen, DoCard::HeartAce]),
            hand_from_vec(vec![DoCard::SpadeKing, DoCard::HeartQueen, DoCard::SpadeQueen, DoCard::DiamondKing, DoCard::HeartTen, DoCard::DiamondTen]),
            hand_from_vec(vec![DoCard::ClubTen, DoCard::SpadeQueen, DoCard::SpadeNine, DoCard::HeartTen, DoCard::SpadeTen, DoCard::DiamondAce]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), false);
    }

    #[test]
    fn test_fail_wedding_not_consistent() {
        // https://www.online-doppelkopf.com/spiele/99.643.778 (mit einem Fehler)
        let team_state = DoTeamState::WeddingUnsolved {
            wedding_player: PLAYER_BOTTOM
        };
        let previous_tricks = heapless::Vec::from_slice(&[
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::ClubAce,
                DoCard::ClubKing,
                DoCard::ClubNine,
                DoCard::ClubTen
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeAce,
                DoCard::SpadeNine,
                DoCard::SpadeKing,
                DoCard::SpadeNine
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::HeartAce,
                DoCard::HeartNine,
                DoCard::HeartNine,
                DoCard::HeartKing
            ]),
            DoTrick::existing(PLAYER_BOTTOM, vec![
                DoCard::SpadeQueen,
                DoCard::HeartTen,
                DoCard::HeartJack,
                DoCard::DiamondTen
            ]),
            DoTrick::existing(PLAYER_LEFT, vec![
                DoCard::DiamondKing,
                DoCard::HeartJack,
                DoCard::SpadeJack,
                DoCard::DiamondNine
            ]),
            DoTrick::existing(PLAYER_RIGHT, vec![
                DoCard::SpadeAce,
                DoCard::SpadeTen,
                DoCard::ClubNine,
                DoCard::SpadeKing
            ]),
        ]).unwrap();

        let real_hands = [
            hand_from_vec(vec![DoCard::HeartQueen, DoCard::ClubQueen, DoCard::ClubQueen, DoCard::HeartTen, DoCard::HeartKing, DoCard::DiamondAce]),
            hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondTen, DoCard::HeartQueen]),
            hand_from_vec(vec![DoCard::ClubKing, DoCard::SpadeJack, DoCard::ClubJack, DoCard::DiamondAce, DoCard::HeartAce, DoCard::ClubTen]),
            hand_from_vec(vec![DoCard::SpadeTen, DoCard::DiamondJack, DoCard::DiamondKing, DoCard::ClubJack, DoCard::SpadeQueen, DoCard::ClubAce]),
        ];

        // Eine Kruez-Dame ist von BOTTOM zu LEFT gewandert. (Tausch mit Karo-Bube)
        let hands = [
            hand_from_vec(vec![DoCard::HeartQueen, DoCard::DiamondJack, DoCard::ClubQueen, DoCard::HeartTen, DoCard::HeartKing, DoCard::DiamondAce]),
            hand_from_vec(vec![DoCard::DiamondNine, DoCard::ClubQueen, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondTen, DoCard::HeartQueen]),
            hand_from_vec(vec![DoCard::ClubKing, DoCard::SpadeJack, DoCard::ClubJack, DoCard::DiamondAce, DoCard::HeartAce, DoCard::ClubTen]),
            hand_from_vec(vec![DoCard::SpadeTen, DoCard::DiamondJack, DoCard::DiamondKing, DoCard::ClubJack, DoCard::SpadeQueen, DoCard::ClubAce]),
        ];

        assert_eq!(are_hands_consistent(team_state, previous_tricks, real_hands, hands), false);
    }
}