use crate::player::player::DoPlayer;
use crate::reservation::reservation::{DoReservation, DoVisibleReservation};
use crate::reservation::reservation_round::DoReservationRound;

/// ToDo: Aktuell macht die Methode mit dem abgespeckten Spiel sehr wenig Sinn.
pub fn get_visible_reservations(
    reservation_round: &DoReservationRound,
    observing_player: DoPlayer,
) -> [Option<DoVisibleReservation>; 4] {
    let mut visible_reservations: [Option<DoVisibleReservation>; 4] = [None; 4];

    let mut is_completed = reservation_round.is_completed();

    for current_player in 0..4 {
        let reservation = reservation_round.reservations[current_player];

        match reservation {
            Some(DoReservation::Healthy) => {
                visible_reservations[current_player] = Some(DoVisibleReservation::Healthy);
            }
            Some(DoReservation::Wedding) => {
                if is_completed || current_player == observing_player {
                    visible_reservations[current_player] = Some(DoVisibleReservation::Wedding);
                } else {
                    visible_reservations[current_player] = Some(DoVisibleReservation::NotRevealed);
                }
            }
            None => {
                visible_reservations[current_player] = None;
            }
        }
    }

    visible_reservations
}