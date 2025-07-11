use serde::{Deserialize, Serialize};
use crate::game_type::game_type::FdoGameType;
use crate::player::player::{ FdoPlayer};
use crate::reservation::reservation::{ FdoReservation};
use crate::reservation::reservation_round::{FdoReservationRound};
use crate::reservation::reservation_winning_logic::FdoReservationResult::NoReservation;

#[derive(Debug, PartialEq, Copy, Clone, Eq, Hash, Serialize, Deserialize)]
pub enum FdoReservationResult {
    NoReservation,
    Solo(FdoPlayer, FdoReservation),
    Wedding(FdoPlayer),
}

impl FdoReservationResult {
    pub fn to_game_type(&self) -> FdoGameType {
        match self {
            FdoReservationResult::NoReservation => FdoGameType::Normal,
            FdoReservationResult::Solo(_, reservation) => match reservation {
                FdoReservation::Healthy => panic!("Healthy is not a valid solo"),
                FdoReservation::Wedding => panic!("Wedding is not a valid solo"),
                FdoReservation::DiamondsSolo => FdoGameType::DiamondsSolo,
                FdoReservation::HeartsSolo => FdoGameType::HeartsSolo,
                FdoReservation::SpadesSolo => FdoGameType::SpadesSolo,
                FdoReservation::ClubsSolo => FdoGameType::ClubsSolo,
                FdoReservation::QueensSolo => FdoGameType::QueensSolo,
                FdoReservation::JacksSolo => FdoGameType::JacksSolo,
                FdoReservation::TrumplessSolo => FdoGameType::TrumplessSolo,
            }
            FdoReservationResult::Wedding(_) => FdoGameType::Wedding,
        }
    }
}

/// Gibt den Gewinner des Vorbehaltsrunde zurück. (Solo schlägt Hochzeit)
pub fn winning_player_in_reservation_round(
    reservation_round: &FdoReservationRound,
) -> FdoReservationResult {
    debug_assert!(reservation_round.is_completed());

    // Der Spieler mit dem bisher "stärksten" Vorbehalt. Das kann nur die
    // Hochzeit sein, da die Soli stärker sind.
    let mut wedding_player_relative_to_reservation_round_begin = None;

    for (player, reservation) in reservation_round
        .reservations
        .iter_with_player() {

        match reservation {
            // ToDo: Hier alle anderen Vorbehalte einfügen :)
            FdoReservation::Wedding => {
                wedding_player_relative_to_reservation_round_begin = Some(player);
            }
            FdoReservation::Healthy => {}
            FdoReservation::DiamondsSolo |
            FdoReservation::HeartsSolo |
            FdoReservation::SpadesSolo |
            FdoReservation::ClubsSolo |
            FdoReservation::QueensSolo |
            FdoReservation::JacksSolo |
            FdoReservation::TrumplessSolo => {
                // Soli schlagen Hochzeit.
                return FdoReservationResult::Solo(player, *reservation);
            }
        }
    }

    match wedding_player_relative_to_reservation_round_begin {
        None => NoReservation,
        Some(wedding_player_relative_to_reservation_round_begin) => FdoReservationResult::Wedding(wedding_player_relative_to_reservation_round_begin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winning_player_in_reservation_round_healthy() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Healthy,
            ],
        );

        assert_eq!(winning_player_in_reservation_round(&reservation_round), FdoReservationResult::NoReservation);
    }

    #[test]
    fn test_winning_player_in_reservation_round_wedding() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
                FdoReservation::Healthy,
            ],
        );

        assert_eq!(
            winning_player_in_reservation_round(&reservation_round),
            FdoReservationResult::Wedding(FdoPlayer::BOTTOM)
        );
    }

    #[test]
    fn test_winning_player_in_reservation_round_wedding_overriden_by_solo() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
                FdoReservation::HeartsSolo,
            ],
        );

        assert_eq!(
            winning_player_in_reservation_round(&reservation_round),
            FdoReservationResult::Solo(FdoPlayer::LEFT, FdoReservation::HeartsSolo)
        );
    }

    #[test]
    fn test_to_game_type() {
        assert_eq!(FdoReservationResult::NoReservation.to_game_type(), FdoGameType::Normal);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::DiamondsSolo).to_game_type(), FdoGameType::DiamondsSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::HeartsSolo).to_game_type(), FdoGameType::HeartsSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::SpadesSolo).to_game_type(), FdoGameType::SpadesSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::ClubsSolo).to_game_type(), FdoGameType::ClubsSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::TrumplessSolo).to_game_type(), FdoGameType::TrumplessSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::QueensSolo).to_game_type(), FdoGameType::QueensSolo);
        assert_eq!(FdoReservationResult::Solo(FdoPlayer::TOP, FdoReservation::JacksSolo).to_game_type(), FdoGameType::JacksSolo);
        assert_eq!(FdoReservationResult::Wedding(FdoPlayer::TOP).to_game_type(), FdoGameType::Wedding);
    }
}