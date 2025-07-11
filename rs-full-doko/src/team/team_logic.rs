use serde::{Deserialize, Serialize};
use crate::card::cards::FdoCard;
use crate::hand::hand::FdoHand;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation_winning_logic::FdoReservationResult;
use crate::trick::trick::FdoTrick;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
pub enum FdoTeamState {
    // In der Vorbehaltsphase, sind noch keine Teams ausgewählt.
    InReservations,

    // Es wurde eine Hochzeit gespielt, diese ist aber noch nicht geklärt.
    WeddingUnsolved { wedding_player : FdoPlayer },

    // Wenn die folgenden Fälle eintreten, kann sich das Team im Rest
    // des Spiels nicht mehr ändern.

    // Es wurde eine Hochzeit gespielt und der Partner wurde gefunden
    // oder es handelt sich um ein Solo, weil der Vorbehaltsgewinner
    // keinen Partner gefunden hat. (Dann ist nur er in re_players).
    WeddingSolved { wedding_player: FdoPlayer, solved_trick_index: usize, re_players: FdoPlayerSet },

    // Es wurde keine Hochzeit gespielt.
    NoWedding { re_players: FdoPlayerSet }
}


impl FdoTeamState {


    /// Ermittelt die Parteien im Spiel, basierend auf den Vorbehaltsrunden,
    /// den bisherigen Stichen und den Karten der Spieler. Auch der Hochzeitsspieler
    /// oder der Solospieler wird ermittelt.
    pub fn resolve(
        reservation_results: FdoReservationResult,

        tricks: &heapless::Vec<FdoTrick, 12>,

        hands: PlayerZeroOrientedArr<FdoHand>,
    ) -> FdoTeamState {
        match reservation_results {
            // Ohne Vorbehalt, spielen die Spieler mit den Kreuz-Damen zusammen.
            FdoReservationResult::NoReservation => {
                let mut re_players: FdoPlayerSet = FdoPlayerSet::empty();

                for player in FdoPlayerSet::all().iter() {
                    if hands[player].contains(FdoCard::ClubQueen) {
                        re_players.insert(player);
                    }
                }

                FdoTeamState::NoWedding { re_players }
            },

            FdoReservationResult::Wedding(wedding_player) => {
                // Mit Hochzeit müssen wir die ersten drei Stiche überprüfen,
                // der erste Spieler außer dem Vorbehaltsgewinner, der einen
                // Stich gewinnt, ist der Partner.
                let mut partner_found_trick: Option<(FdoPlayer, usize)> = None;
                let mut completed_tricks = 0;

                for i in 0..3 {
                    if i >= tricks.len() {
                        break;
                    }

                    let trick = &tricks[i];

                    if !trick.is_completed() {
                        break;
                    }

                    completed_tricks += 1;

                    // Hier ist immer ein Normalspiel gegeben, da das Ergebnis
                    // der Vorbehaltsrunde bereits feststeht (Hochzeit).
                    let winner = trick.winning_player.unwrap();

                    if winner != wedding_player {
                        partner_found_trick = Some((winner, i));
                        break;
                    }
                }

                match partner_found_trick {
                    // Ein Partner wurde gefunden.
                    Some((trick_winner, trick_solved_index)) => {
                        let mut re_players: FdoPlayerSet = FdoPlayerSet::empty();

                        re_players.insert(wedding_player);
                        re_players.insert(trick_winner);

                        FdoTeamState::WeddingSolved { wedding_player: wedding_player, solved_trick_index: trick_solved_index, re_players: re_players }
                    },
                    // Es wurde kein Partner (bisher) gefunden.
                    None => {
                        if completed_tricks == 3 {
                            // Der Vorbehaltsgewinner hat keinen Partner gefunden. Er spielt ein Solo.
                            let mut re_players: FdoPlayerSet = FdoPlayerSet::empty();

                            re_players.insert(wedding_player);

                            return FdoTeamState::WeddingSolved { wedding_player: wedding_player, solved_trick_index: 2, re_players: re_players }
                        }

                        // Die Hochzeit ist noch nicht aufgeklärt.
                        FdoTeamState::WeddingUnsolved { wedding_player: wedding_player }
                    }
                }
            }
            FdoReservationResult::Solo(winning_player, _reservation) => {
                let mut re_players: FdoPlayerSet = FdoPlayerSet::empty();

                re_players.insert(winning_player);

                FdoTeamState::NoWedding { re_players }
            }
        }
    }

    pub fn get_re_players(&self) -> Option<FdoPlayerSet> {
        match self {
            FdoTeamState::InReservations => { None }
            FdoTeamState::WeddingUnsolved { .. } => { None }
            FdoTeamState::WeddingSolved { re_players, .. } => { Some(*re_players) }
            FdoTeamState::NoWedding { re_players } => { Some(*re_players) }
        }
    }

    pub fn get_re_players_or_panic(&self) -> FdoPlayerSet {
        match self {
            FdoTeamState::InReservations => { panic!("Die Teams stehen in der Vorbehaltsphase noch nicht fest.") }
            FdoTeamState::WeddingUnsolved { .. } => { panic!("Die Teams stehen noch nicht fest. Die Hochzeit ist ungeklärt.") }
            FdoTeamState::WeddingSolved { re_players, .. } => { *re_players }
            FdoTeamState::NoWedding { re_players } => { *re_players }
        }
    }

    /// Gibt zurück, ob der Teamzustand final ist, d.h. ob sich das Team
    /// im Rest des Spiels nicht mehr ändern kann.
    pub fn is_final(&self) -> bool {
        match self {
            FdoTeamState::InReservations => false,
            FdoTeamState::WeddingUnsolved { .. } => false,
            FdoTeamState::WeddingSolved { .. } => true,
            FdoTeamState::NoWedding { .. } => true
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reservation::reservation::FdoReservation;
    use super::*;

    #[test]
    fn test_is_final_team_state() {
        assert!(!FdoTeamState::InReservations.is_final());
        assert!(!FdoTeamState::WeddingUnsolved { wedding_player: FdoPlayer::TOP }.is_final());
        assert!(FdoTeamState::WeddingSolved { wedding_player: FdoPlayer::TOP, solved_trick_index: 0, re_players: FdoPlayerSet::empty() }.is_final());
        assert!(FdoTeamState::NoWedding { re_players: FdoPlayerSet::empty() }.is_final());
    }

    /// Testet den Fall, dass kein Vorbehalt gemacht wurde und
    /// ein Spieler beide Kreuz-Damen hat. Er spielt dann
    /// ein stilles Solo.
    #[test]
    fn test_resolve_team_state_no_reservation_silent_solo() {
        let reservation_results = FdoReservationResult::NoReservation;
        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![FdoCard::ClubQueen, FdoCard::ClubQueen]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::NoWedding { re_players } => {
                assert!(re_players.contains(FdoPlayer::TOP));
                assert_eq!(re_players.len(), 1);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(team_state.is_final());
    }

    /// Testet den absoluten Standardfall. Kein Vorbehalt, Kreuz-Damen sind verteilt.
    #[test]
    fn test_resolve_team_state_no_reservation_with_partner() {
        let reservation_results = FdoReservationResult::NoReservation;
        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![FdoCard::ClubQueen]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![FdoCard::ClubQueen]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::NoWedding { re_players } => {
                assert!(re_players.contains(FdoPlayer::BOTTOM));
                assert!(re_players.contains(FdoPlayer::TOP));
                assert_eq!(re_players.len(), 2);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(team_state.is_final());
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde,
    /// aber diese noch nicht aufgeklärt wurde.
    #[test]
    fn test_resolve_team_state_wedding_unsolved() {
        let reservation_results = FdoReservationResult::Wedding(FdoPlayer::TOP);
        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
            FdoTrick::existing(FdoPlayer::BOTTOM, vec![FdoCard::ClubAce, FdoCard::ClubNine, FdoCard::DiamondAce, FdoCard::ClubNine]),
            FdoTrick::existing(FdoPlayer::BOTTOM, vec![FdoCard::ClubAce, FdoCard::ClubNine, FdoCard::DiamondAce, FdoCard::ClubNine]),
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![FdoCard::ClubQueen, FdoCard::ClubQueen]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::WeddingUnsolved { wedding_player } => {
                assert_eq!(wedding_player, FdoPlayer::TOP);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(!team_state.is_final());
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde und der Partner
    /// gefunden wurde.
    #[test]
    fn test_resolve_team_state_wedding_with_partner() {
        let reservation_results = FdoReservationResult::Wedding(FdoPlayer::TOP);
        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
            FdoTrick::existing(FdoPlayer::BOTTOM, vec![FdoCard::ClubAce, FdoCard::DiamondAce, FdoCard::ClubNine, FdoCard::ClubNine]),
            FdoTrick::existing(FdoPlayer::LEFT, vec![FdoCard::SpadeAce, FdoCard::SpadeTen, FdoCard::DiamondAce, FdoCard::SpadeNine]),
            FdoTrick::existing(FdoPlayer::RIGHT, vec![FdoCard::HeartAce, FdoCard::HeartKing, FdoCard::HeartKing, FdoCard::HeartNine])
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![FdoCard::ClubQueen, FdoCard::ClubQueen]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::WeddingSolved { wedding_player, solved_trick_index, re_players } => {
                assert_eq!(wedding_player, FdoPlayer::TOP);
                assert_eq!(solved_trick_index, 0);

                assert!(re_players.contains(FdoPlayer::TOP));
                assert!(re_players.contains(FdoPlayer::LEFT));
                assert_eq!(re_players.len(), 2);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(team_state.is_final());
    }

    /// Testet den Fall, dass eine Hochzeit angesagt wurde und kein Partner
    /// gefunden wurde. Der Vorbehaltsgewinner spielt ein Karo-Solo.
    #[test]
    fn test_resolve_team_state_wedding_solo() {
        let reservation_results = FdoReservationResult::Wedding(FdoPlayer::TOP);

        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
            FdoTrick::existing(FdoPlayer::BOTTOM, vec![FdoCard::ClubAce, FdoCard::ClubTen, FdoCard::DiamondAce, FdoCard::ClubNine]),
            FdoTrick::existing(FdoPlayer::TOP, vec![FdoCard::SpadeAce, FdoCard::SpadeTen, FdoCard::SpadeTen, FdoCard::SpadeNine]),
            FdoTrick::existing(FdoPlayer::TOP, vec![FdoCard::HeartAce, FdoCard::HeartKing, FdoCard::HeartKing, FdoCard::HeartNine])
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![FdoCard::ClubQueen, FdoCard::ClubQueen]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::WeddingSolved { wedding_player, solved_trick_index, re_players } => {
                assert_eq!(wedding_player, FdoPlayer::TOP);
                assert_eq!(solved_trick_index, 2);

                assert!(re_players.contains(FdoPlayer::TOP));
                assert_eq!(re_players.len(), 1);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(team_state.is_final());
    }


    /// Testet den Fall, dass ein Solo angesagt wurde.
    #[test]
    fn test_resolve_team_state_solo() {
        let reservation_results = FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::HeartsSolo);

        let tricks: heapless::Vec<FdoTrick, 12> = heapless::Vec::from_slice(&[
        ]).unwrap();
        let hands = PlayerZeroOrientedArr::from_full([
            // PLAYER_BOTTOM
            FdoHand::from_vec(vec![]),
            // PLAYER_LEFT
            FdoHand::from_vec(vec![]),
            // PLAYER_TOP
            FdoHand::from_vec(vec![]),
            // PLAYER_RIGHT
            FdoHand::from_vec(vec![]),
        ]);

        let team_state = FdoTeamState::resolve(reservation_results, &tricks, hands);

        match team_state {
            FdoTeamState::NoWedding { re_players } => {
                assert!(re_players.contains(FdoPlayer::TOP));
                assert_eq!(re_players.len(), 1);
            }
            _ => panic!("Unexpected team state: {:?}", team_state)
        }

        assert!(team_state.is_final());
    }

    #[test]
    fn test_get_re_players() {
        let team_state = FdoTeamState::NoWedding { re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]) };
        assert_eq!(team_state.get_re_players(), Some(FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])));

        let team_state = FdoTeamState::WeddingUnsolved { wedding_player: FdoPlayer::TOP };
        assert_eq!(team_state.get_re_players(), None);

        let team_state = FdoTeamState::WeddingSolved { wedding_player: FdoPlayer::TOP, solved_trick_index: 0, re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]) };
        assert_eq!(team_state.get_re_players(), Some(FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])));

        let team_state = FdoTeamState::InReservations;
        assert_eq!(team_state.get_re_players(), None);
    }


}