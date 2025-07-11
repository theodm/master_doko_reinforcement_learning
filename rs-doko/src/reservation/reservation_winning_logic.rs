use crate::player::player::{DoPlayer, player_wraparound};
use crate::reservation::reservation::DoReservation;
use crate::reservation::reservation_round::DoReservationRound;
use crate::reservation::reservation_winning_logic::DoReservationResult::NoReservation;

#[derive(Debug, PartialEq, Copy, Clone, Eq, Hash)]
pub enum DoReservationResult {
    NoReservation,
    Wedding(DoPlayer),
}


pub fn winning_player_in_reservation_round(
    reservation_round: &DoReservationRound,
) -> DoReservationResult {
    debug_assert!(reservation_round.is_completed());

    let mut wedding_player_relative_to_reservation_round_begin = None;

    for i in 0..4 {
        let reservation = reservation_round.reservations[i].unwrap();

        match reservation {
            // ToDo: Hier alle anderen Vorbehalte einfügen :)
            DoReservation::Wedding => {
                wedding_player_relative_to_reservation_round_begin = Some(i);
            }
            DoReservation::Healthy => {}
        }

    }

    match wedding_player_relative_to_reservation_round_begin {
        None => NoReservation,
        Some(wedding_player_relative_to_reservation_round_begin) => DoReservationResult::Wedding(player_wraparound(wedding_player_relative_to_reservation_round_begin + reservation_round.start_player))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use crate::reservation::reservation::DoReservation;

    #[test]
    fn test_winning_player_in_reservation_round_healthy() {
        let reservation_round = DoReservationRound::existing(
            PLAYER_TOP,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Healthy,
            ],
        );

        assert_eq!(winning_player_in_reservation_round(&reservation_round), DoReservationResult::NoReservation);
    }

    #[test]
    fn test_winning_player_in_reservation_round_wedding() {
        let reservation_round = DoReservationRound::existing(
            PLAYER_TOP,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Wedding,
                DoReservation::Healthy,
            ],
        );

        assert_eq!(
            winning_player_in_reservation_round(&reservation_round),
            DoReservationResult::Wedding(PLAYER_BOTTOM)
        );
    }
}