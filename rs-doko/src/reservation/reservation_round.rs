use crate::debug_assert_valid_player;
use crate::player::player::{DoPlayer, player_wraparound};
use crate::reservation::reservation::DoReservation;

/// Stellt den Status der Vorbehaltsrunde dar, also wer sie begonnen hat
/// und welche Vorbehalte angesagt wurden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DoReservationRound {
    // Die Vorbehalte, die in dieser Vorbehaltsrunde angesagt wurden,
    // beginnend mit der Karte des Startspieler.
    pub reservations: [Option<DoReservation>; 4],
    pub start_player: DoPlayer,
}

impl DoReservationRound {
    pub fn empty(start_player: DoPlayer) -> DoReservationRound {
        debug_assert_valid_player!(start_player);

        DoReservationRound {
            reservations: [None, None, None, None],
            start_player,
        }
    }

    /// Gibt den aktuellen Spieler zurück, der als nächstes einen
    /// Vorbehalt ansagen muss. Oder None, wenn die Vorbehaltsrunde
    /// vollständig ist.
    pub fn current_player(&self) -> Option<DoPlayer> {
        if self.is_completed() {
            return None;
        }

        for i in 0..4 {
            if self.reservations[i].is_none() {
                return Some(player_wraparound(self.start_player + i));
            }
        }

        panic!("should not happen");
    }

    /// Gibt an, ob die Vorbehaltsrunde vollständig ist, also alle 4 Vorbehalte
    /// angesagt wurden.
    pub fn is_completed(&self) -> bool {
        self.reservations.iter().all(|reservation| reservation.is_some())
    }

    /// Nur für Testzwecke, kann hier eine bereits bestehende
    /// Vorbehaltsrunde erstellt werden. Im realen Spiel startet eine
    /// Vorbehaltsrunde immer mit der Methode [empty].
    pub fn existing(
        start_player: DoPlayer,
        reservations: Vec<DoReservation>
    ) -> DoReservationRound {
        debug_assert_valid_player!(start_player);
        assert!(reservations.len() <= 4, "Eine Vorbehaltsrunde kann maximal 4 Vorbehalte haben.");

        let mut reservation_round = DoReservationRound::empty(start_player);

        for (index, reservation) in reservations.iter().enumerate() {
            reservation_round.reservations[index] = Some(*reservation);
        }

        reservation_round
    }
}

#[cfg(test)]
mod tests {
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use super::*;
    use crate::reservation::reservation::DoReservation;

    #[test]
    fn test_reservation_round_empty() {
        let reservation_round = DoReservationRound::empty(PLAYER_TOP);

        assert_eq!(reservation_round.current_player(), Some(PLAYER_TOP));
        assert!(!reservation_round.is_completed());
    }

    #[test]
    fn test_reservation_round_existing() {
        let reservation_round = DoReservationRound::existing(
            PLAYER_TOP,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Wedding,
            ],
        );

        assert_eq!(reservation_round.current_player(), Some(PLAYER_LEFT));
        assert!(!reservation_round.is_completed());
    }

    #[test]
    fn test_current_player() {
        let reservation_round = DoReservationRound::empty(PLAYER_LEFT);
        assert_eq!(reservation_round.current_player(), Some(PLAYER_LEFT));

        let reservation_round = DoReservationRound::existing(
            PLAYER_LEFT,
            vec![
                DoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(PLAYER_TOP));

        let reservation_round = DoReservationRound::existing(
            PLAYER_LEFT,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(PLAYER_RIGHT));

        let reservation_round = DoReservationRound::existing(
            PLAYER_LEFT,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Wedding,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(PLAYER_BOTTOM));

        let reservation_round = DoReservationRound::existing(
            PLAYER_LEFT,
            vec![
                DoReservation::Healthy,
                DoReservation::Healthy,
                DoReservation::Wedding,
                DoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), None);

    }
}