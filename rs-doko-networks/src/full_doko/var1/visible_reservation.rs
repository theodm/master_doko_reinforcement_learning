use rs_full_doko::reservation::reservation::{FdoReservation, FdoVisibleReservation};

/// Anzahl der sichtbaren Vorbehalte im Spiel (für die Embedding-Größe).
pub const VISIBLE_RESERVATION_OR_NONE_COUNT: i64 = 11;


/// Kodiert den sichtbaren Vorbehalt. Wird
/// als Embedding innerhalb des neuronalen
/// Netzwerkes verwendet.
pub fn encode_visible_reservation(
    visible_reservation: FdoVisibleReservation
) -> [i64; 1] {
    fn map_visible_reservation(
        reservation: FdoVisibleReservation
    ) -> i64 {
        match reservation {
            FdoVisibleReservation::NoneYet => 0,
            FdoVisibleReservation::NotRevealed => 1,
            FdoVisibleReservation::Healthy => 2,
            FdoVisibleReservation::Wedding => 3,
            FdoVisibleReservation::DiamondsSolo => 4,
            FdoVisibleReservation::HeartsSolo => 5,
            FdoVisibleReservation::SpadesSolo => 6,
            FdoVisibleReservation::ClubsSolo => 7,
            FdoVisibleReservation::QueensSolo => 8,
            FdoVisibleReservation::JacksSolo => 9,
            FdoVisibleReservation::TrumplessSolo => 10
        }
    }

    let reservation_num = map_visible_reservation(visible_reservation);

    debug_assert!(reservation_num < VISIBLE_RESERVATION_OR_NONE_COUNT);
    debug_assert!(reservation_num >= 0);

    [reservation_num]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_reservation_or_none() {
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::NoneYet), [0]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::NotRevealed), [1]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::Healthy), [2]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::Wedding), [3]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::DiamondsSolo), [4]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::HeartsSolo), [5]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::SpadesSolo), [6]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::ClubsSolo), [7]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::QueensSolo), [8]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::JacksSolo), [9]);
        assert_eq!(encode_visible_reservation(FdoVisibleReservation::TrumplessSolo), [10]);
    }
}