use crate::reservation::reservation::DoReservation;
use crate::reservation::reservation_round::DoReservationRound;


/// Fügt einen gespielten Vorbehalt zu einer Vorbehaltsrunde an der Stelle des aktuellen Spielers
/// hinzu.
///
/// Achtung: Es wird nicht geprüft, ob der aktuelle Spieler den Vorbehalt überhaupt spielen darf. Das
/// muss vom Aufrufer sichergestellt werden. Zusätzlich muss natürlich auch sichergestellt werden,
/// dass der korrekte Spieler den Zug macht.
pub fn play_reservation(
    reservation_round: &mut DoReservationRound,
    played_reservation: DoReservation
) {
    debug_assert!(!reservation_round.is_completed());

    for player in 0..4 {
        let reservation = reservation_round.reservations[player];

        match reservation {
            None => {
                reservation_round.reservations[player] = Some(played_reservation);
                return;
            }
            Some(_) => {}
        }
    }

    panic!("should not happen")
}

#[cfg(test)]
mod tests {
    use crate::player::player::PLAYER_TOP;
    use super::*;

    #[test]
    fn test_play_reservation() {
        let mut reservation_round = DoReservationRound::empty(PLAYER_TOP);
        play_reservation(&mut reservation_round, DoReservation::Healthy);

        assert_eq!(reservation_round.reservations[0], Some(DoReservation::Healthy));
    }

    #[test]
    fn test_play_reservation_full() {
        let mut reservation_round = DoReservationRound::empty(PLAYER_TOP);

        play_reservation(&mut reservation_round, DoReservation::Healthy);
        play_reservation(&mut reservation_round, DoReservation::Healthy);
        play_reservation(&mut reservation_round, DoReservation::Wedding);
        play_reservation(&mut reservation_round, DoReservation::Healthy);

        assert_eq!(reservation_round.reservations[0], Some(DoReservation::Healthy));
        assert_eq!(reservation_round.reservations[1], Some(DoReservation::Healthy));
        assert_eq!(reservation_round.reservations[2], Some(DoReservation::Wedding));
        assert_eq!(reservation_round.reservations[3], Some(DoReservation::Healthy));
        assert!(reservation_round.is_completed());
    }
}