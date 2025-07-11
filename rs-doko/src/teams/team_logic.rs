use crate::card::cards::DoCard;
use crate::hand::hand::{DoHand, hand_contains};
use crate::player::player::DoPlayer;
use crate::player::player_set::{DoPlayerSet, player_set_add};
use crate::reservation::reservation_winning_logic::DoReservationResult;
use crate::trick::trick::DoTrick;
use crate::trick::trick_winning_player_logic::winning_player_in_trick_in_normal_game;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum DoTeamState {
    // In der Vorbehaltsphase, sind noch keine Teams ausgewählt.
    InReservations,
    // Es wurde eine Hochzeit gespielt, diese ist aber noch nicht geklärt.
    WeddingUnsolved { wedding_player : DoPlayer },

    // Wenn die folgenden Fälle eintreten, kann sich das Team im Rest
    // des Spiels nicht mehr ändern.

    // Es wurde eine Hochzeit gespielt und der Partner wurde gefunden
    // oder es handelt sich um ein Solo, weil der Vorbehaltsgewinner
    // keinen Partner gefunden hat. (Dann ist nur er in re_players).
    WeddingSolved { wedding_player: DoPlayer, solved_trick_index: usize, re_players: DoPlayerSet },

    // Es wurde keine Hochzeit gespielt.
    NoWedding { re_players: DoPlayerSet }
}

impl DoTeamState {

    pub fn get_re_players(&self) -> Option<DoPlayerSet> {
        match self {
            DoTeamState::InReservations => { None }
            DoTeamState::WeddingUnsolved { .. } => { None }
            DoTeamState::WeddingSolved { re_players, .. } => { Some(*re_players) }
            DoTeamState::NoWedding { re_players } => { Some(*re_players) }
        }
    }

    pub fn get_re_players_or_empty(&self) -> DoPlayerSet {
        match self.get_re_players() {
            Some(re_players) => { re_players }
            None => { 0 }
        }
    }

    pub fn get_re_players_or_panic(&self) -> DoPlayerSet {
        match self {
            DoTeamState::InReservations => { panic!("Die Teams stehen in der Vorbehaltsphase noch nicht fest.") }
            DoTeamState::WeddingUnsolved { .. } => { panic!("Die Teams stehen noch nicht fest. Die Hochzeit ist ungeklärt.") }
            DoTeamState::WeddingSolved { re_players, .. } => { *re_players }
            DoTeamState::NoWedding { re_players } => { *re_players }
        }
    }
}

/// Gibt zurück, ob der Teamzustand final ist, d.h. ob sich das Team
/// im Rest des Spiels nicht mehr ändern kann.
pub fn is_final_team_state(
    team_state: DoTeamState
) -> bool {
    match team_state {
        DoTeamState::InReservations => false,
        DoTeamState::WeddingUnsolved { .. } => false,
        DoTeamState::WeddingSolved { .. } => true,
        DoTeamState::NoWedding { .. } => true
    }
}

pub fn resolve_team_state(
    reservation_results: DoReservationResult,

    tricks: &[Option<DoTrick>; 12],

    hands: [DoHand; 4],
) -> DoTeamState {
    match reservation_results {
        // Ohne Vorbehalt, spielen die Spieler mit den Kreuz-Damen zusammen.
        DoReservationResult::NoReservation => {
            let mut re_players: DoPlayerSet = 0;

            for i in 0..4 {
                if hand_contains(hands[i], DoCard::ClubQueen) {
                    re_players = player_set_add(re_players, i);
                }
            }

            return DoTeamState::NoWedding { re_players: re_players }
        },

        DoReservationResult::Wedding(wedding_player) => {
            // Mit Hochzeit müssen wir die ersten drei Stiche überprüfen,
            // der erste Spieler außer dem Vorbehaltsgewinner, der einen
            // Stich gewinnt, ist der Partner.
            let mut partner_found_trick: Option<(DoPlayer, usize)> = None;
            let mut completed_tricks = 0;

            for i in 0..3 {
                if let Some(trick) = tricks[i] {
                    if !trick.is_completed() {
                        break;
                    }

                    completed_tricks += 1;

                    let winner = winning_player_in_trick_in_normal_game(&trick);

                    if winner != wedding_player {
                        partner_found_trick = Some((winner, i));
                        break;
                    }
                }
            }

            match partner_found_trick {
                // Ein Partner wurde gefunden.
                Some((trick_winner, trick_solved_index)) => {
                    let mut re_players: DoPlayerSet = 0;

                    re_players = player_set_add(re_players, wedding_player);
                    re_players = player_set_add(re_players, trick_winner);

                    return DoTeamState::WeddingSolved { wedding_player: wedding_player, solved_trick_index: trick_solved_index, re_players: re_players }
                },
                // Es wurde kein Partner (bisher) gefunden.
                None => {
                    if completed_tricks == 3 {
                        // Der Vorbehaltsgewinner hat keinen Partner gefunden. Er spielt ein Solo.
                        let mut re_players: DoPlayerSet = 0;

                        re_players = player_set_add(re_players, wedding_player);

                        return DoTeamState::WeddingSolved { wedding_player: wedding_player, solved_trick_index: 2, re_players: re_players }
                    }

                    // Die Hochzeit ist noch nicht aufgeklärt.
                    return DoTeamState::WeddingUnsolved { wedding_player: wedding_player }
                }

            }
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::{hand::hand::hand_from_vec, player::{player::{PLAYER_BOTTOM, PLAYER_TOP}, player_set::player_set_contains}};
    use crate::player::player::{PLAYER_LEFT, PLAYER_RIGHT};
    use crate::player::player_set::player_set_len;
    use super::*;

    #[test]
    fn test_is_final_team_state() {
        assert!(!is_final_team_state(DoTeamState::InReservations));
        assert!(!is_final_team_state(DoTeamState::WeddingUnsolved { wedding_player: PLAYER_TOP }));
        assert!(is_final_team_state(DoTeamState::WeddingSolved { wedding_player: PLAYER_TOP, solved_trick_index: 0, re_players: 0 }));
        assert!(is_final_team_state(DoTeamState::NoWedding { re_players: 0 }));
    }

    /// Testet den Fall, dass kein Vorbehalt gemacht wurde und
    /// ein Spieler beide Kreuz-Damen hat. Er spielt dann
    /// ein stilles Solo.
    #[test]
    fn test_resolve_team_state_no_reservation_silent_solo() {
        let reservation_results = DoReservationResult::NoReservation;
        let tricks = [None; 12];
        let hands = [
            // PLAYER_BOTTOM
            hand_from_vec(vec![]),
            // PLAYER_LEFT
            hand_from_vec(vec![]),
            // PLAYER_TOP
            hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen]),
            // PLAYER_RIGHT
            hand_from_vec(vec![]),
        ];

        let team_state = resolve_team_state(reservation_results, &tricks, hands);

        match team_state {
            DoTeamState::NoWedding { re_players } => {
                assert!(player_set_contains(re_players, PLAYER_TOP));
                assert_eq!(player_set_len(re_players), 1);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(is_final_team_state(team_state));
    }

    /// Testet den absoluten Standardfall. Kein Vorbehalt, Kreuz-Damen sind verteilt.
    #[test]
    fn test_resolve_team_state_no_reservation_with_partner() {
        let reservation_results = DoReservationResult::NoReservation;
        let tricks = [
            None, None, None, None, None, None, None, None, None, None, None, None
        ];
        let hands = [
            // PLAYER_BOTTOM
            hand_from_vec(vec![DoCard::ClubQueen]),
            // PLAYER_LEFT
            hand_from_vec(vec![]),
            // PLAYER_TOP
            hand_from_vec(vec![DoCard::ClubQueen]),
            // PLAYER_RIGHT
            hand_from_vec(vec![]),
        ];

        let team_state = resolve_team_state(reservation_results, &tricks, hands);

        match team_state {
            DoTeamState::NoWedding { re_players } => {
                assert!(player_set_contains(re_players, PLAYER_BOTTOM));
                assert!(player_set_contains(re_players, PLAYER_TOP));
                assert_eq!(player_set_len(re_players), 2);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(is_final_team_state(team_state));
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde,
    /// aber diese noch nicht aufgeklärt wurde.
    #[test]
    fn test_resolve_team_state_wedding_unsolved() {
        let reservation_results = DoReservationResult::Wedding(PLAYER_TOP);
        let tricks = [
            Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::ClubAce, DoCard::ClubNine, DoCard::DiamondAce, DoCard::ClubNine])),
            Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::ClubAce, DoCard::ClubNine, DoCard::DiamondAce, DoCard::ClubNine])),
             None, None, None, None, None, None, None, None, None, None
        ];
        let hands = [
            // PLAYER_BOTTOM
            hand_from_vec(vec![]),
            // PLAYER_LEFT
            hand_from_vec(vec![]),
            // PLAYER_TOP
            hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen]),
            // PLAYER_RIGHT
            hand_from_vec(vec![]),
        ];

        let team_state = resolve_team_state(reservation_results, &tricks, hands);

        match team_state {
            DoTeamState::WeddingUnsolved { wedding_player } => {
                assert_eq!(wedding_player, PLAYER_TOP);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(!is_final_team_state(team_state));
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde und der Partner
    /// gefunden wurde.
    #[test]
    fn test_resolve_team_state_wedding_with_partner() {
        let reservation_results = DoReservationResult::Wedding(PLAYER_TOP);
        let tricks = [
            Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::ClubAce, DoCard::DiamondAce, DoCard::ClubNine, DoCard::ClubNine])),
            Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::SpadeAce, DoCard::SpadeTen, DoCard::DiamondAce, DoCard::SpadeNine])),
            Some(DoTrick::existing(PLAYER_RIGHT, vec![DoCard::HeartAce, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartNine])),
            None, None, None, None, None, None, None, None, None
        ];
        let hands = [
            // PLAYER_BOTTOM
            hand_from_vec(vec![]),
            // PLAYER_LEFT
            hand_from_vec(vec![]),
            // PLAYER_TOP
            hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen]),
            // PLAYER_RIGHT
            hand_from_vec(vec![]),
        ];

        let team_state = resolve_team_state(reservation_results, &tricks, hands);

        match team_state {
            DoTeamState::WeddingSolved { wedding_player, solved_trick_index, re_players } => {
                assert_eq!(wedding_player, PLAYER_TOP);
                assert_eq!(solved_trick_index, 0);

                assert!(player_set_contains(re_players, PLAYER_TOP));
                assert!(player_set_contains(re_players, PLAYER_LEFT));
                assert_eq!(player_set_len(re_players), 2);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(is_final_team_state(team_state));
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde und kein Partner
    /// gefunden wurde. Der Vorbehaltsgewinner spielt ein Karo-Solo.
    #[test]
    fn test_resolve_team_state_wedding_solo() {
        let reservation_results = DoReservationResult::Wedding(PLAYER_TOP);

        let tricks = [
            Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::ClubAce, DoCard::ClubTen, DoCard::DiamondAce, DoCard::ClubNine])),
            Some(DoTrick::existing(PLAYER_TOP, vec![DoCard::SpadeAce, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeNine])),
            Some(DoTrick::existing(PLAYER_TOP, vec![DoCard::HeartAce, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartNine])),
            None, None, None, None, None, None, None, None, None
        ];
        let hands = [
            // PLAYER_BOTTOM
            hand_from_vec(vec![]),
            // PLAYER_LEFT
            hand_from_vec(vec![]),
            // PLAYER_TOP
            hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen]),
            // PLAYER_RIGHT
            hand_from_vec(vec![]),
        ];

        let team_state = resolve_team_state(reservation_results, &tricks, hands);

        match team_state {
            DoTeamState::WeddingSolved { wedding_player, solved_trick_index, re_players } => {
                assert_eq!(wedding_player, PLAYER_TOP);
                assert_eq!(solved_trick_index, 2);

                assert!(player_set_contains(re_players, PLAYER_TOP));
                assert_eq!(player_set_len(re_players), 1);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(is_final_team_state(team_state));
    }

}