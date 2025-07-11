use std::ops::Index;
use serde::{Deserialize, Serialize};
use crate::player::player::{FdoPlayer};
use crate::reservation::reservation::FdoReservation;
use crate::util::po_vec::PlayerOrientedVec;

/// Stellt den Status der Vorbehaltsrunde dar, also wer sie begonnen hat
/// und welche Vorbehalte angesagt wurden.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoReservationRound {
    pub reservations: PlayerOrientedVec<FdoReservation>
}

impl FdoReservationRound {
    pub fn empty(start_player: FdoPlayer) -> FdoReservationRound {
        FdoReservationRound {
            reservations: PlayerOrientedVec::empty(start_player)
        }
    }

    /// Gibt den aktuellen Spieler zurück, der als nächstes einen
    /// Vorbehalt ansagen muss. Oder None, wenn die Vorbehaltsrunde
    /// vollständig ist.
    pub fn current_player(&self) -> Option<FdoPlayer> {
        self.reservations.next_empty_player()
    }

    /// Gibt an, ob die Vorbehaltsrunde vollständig ist, also alle 4 Vorbehalte
    /// angesagt wurden.
    pub fn is_completed(&self) -> bool {
        self.reservations.is_full()
    }

    pub fn starting_player(&self) -> FdoPlayer {
        self.reservations.starting_player
    }

    /// Nur für Testzwecke, kann hier eine bereits bestehende
    /// Vorbehaltsrunde erstellt werden. Im realen Spiel startet eine
    /// Vorbehaltsrunde immer mit der Methode [empty].
    pub fn existing(
        start_player: FdoPlayer,
        reservations: Vec<FdoReservation>
    ) -> FdoReservationRound {
        let mut reservation_round = FdoReservationRound::empty(start_player);

        for reservation in reservations {
            reservation_round.play_reservation(reservation);
        }

        reservation_round
    }

    /// Fügt einen gespielten Vorbehalt zu einer Vorbehaltsrunde an der Stelle des aktuellen Spielers
    /// hinzu.
    ///
    /// Achtung: Es wird nicht geprüft, ob der aktuelle Spieler den Vorbehalt überhaupt spielen darf. Das
    /// muss vom Aufrufer sichergestellt werden. Zusätzlich muss natürlich auch sichergestellt werden,
    /// dass der korrekte Spieler den Zug macht.
    pub fn play_reservation(
        &mut self,
        played_reservation: FdoReservation
    ) {
        debug_assert!(!self.is_completed());

        self.reservations.push(played_reservation);
    }
}

impl Index<FdoPlayer> for FdoReservationRound {
    type Output = FdoReservation;

    fn index(&self, index: FdoPlayer) -> &Self::Output {
        &self.reservations[index]
    }
}


#[cfg(test)]
mod tests {
    use crate::player::player::FdoPlayer;
    use crate::reservation::reservation::FdoReservation;
    use crate::reservation::reservation_round::FdoReservationRound;

    #[test]
    fn test_reservation_round_empty() {
        let reservation_round = FdoReservationRound::empty(FdoPlayer::TOP);

        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::TOP));
        assert!(!reservation_round.is_completed());
    }

    #[test]
    fn test_reservation_round_existing() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
            ],
        );

        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::LEFT));
        assert!(!reservation_round.is_completed());
    }

    #[test]
    fn test_current_player() {
        let reservation_round = FdoReservationRound::empty(FdoPlayer::LEFT);
        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::LEFT));

        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::TOP));

        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::RIGHT));

        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
            ],
        );
        assert_eq!(reservation_round.current_player(), Some(FdoPlayer::BOTTOM));

        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
                FdoReservation::Healthy,
            ],
        );
        assert_eq!(reservation_round.current_player(), None);
    }

    #[test]
    fn test_play_reservation() {
        let mut reservation_round = FdoReservationRound::empty(FdoPlayer::TOP);
        reservation_round.play_reservation(FdoReservation::Healthy);

        assert_eq!(reservation_round.reservations[FdoPlayer::TOP], FdoReservation::Healthy);
    }

    #[test]
    fn test_play_reservation_full() {
        let mut reservation_round = FdoReservationRound::empty(FdoPlayer::TOP);

        reservation_round.play_reservation(FdoReservation::Healthy);
        reservation_round.play_reservation(FdoReservation::Healthy);
        reservation_round.play_reservation(FdoReservation::Wedding);
        reservation_round.play_reservation(FdoReservation::Healthy);

        assert_eq!(reservation_round.reservations[FdoPlayer::TOP], FdoReservation::Healthy);
        assert_eq!(reservation_round.reservations[FdoPlayer::RIGHT], FdoReservation::Healthy);
        assert_eq!(reservation_round.reservations[FdoPlayer::BOTTOM], FdoReservation::Wedding);
        assert_eq!(reservation_round.reservations[FdoPlayer::LEFT], FdoReservation::Healthy);
        assert!(reservation_round.is_completed());
    }
}