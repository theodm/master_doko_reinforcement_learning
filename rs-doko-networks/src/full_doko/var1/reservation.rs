use rs_full_doko::reservation::reservation::FdoReservation;

/// Anzahl der Vorbehalte (oder None) im Spiel (für die Embedding-Größe).
pub const RESERVATION_OR_NONE_COUNT: i64 = 10;


/// Kodiert den Vorbehalt (oder None). Wird
/// als Embedding innerhalb des neuronalen
/// Netzwerkes verwendet.
pub fn encode_reservation_or_none(
    reservation: Option<FdoReservation>
) -> [i64; 1] {
    fn map_reservation(
        reservation: Option<FdoReservation>
    ) -> i64 {
        match reservation {
            None => 0,
            Some(reservation) => {
                match reservation {
                    FdoReservation::Healthy => 1,
                    FdoReservation::Wedding => 2,

                    FdoReservation::DiamondsSolo => 3,
                    FdoReservation::HeartsSolo => 4,
                    FdoReservation::SpadesSolo => 5,
                    FdoReservation::ClubsSolo => 6,

                    FdoReservation::QueensSolo => 7,
                    FdoReservation::JacksSolo => 8,

                    FdoReservation::TrumplessSolo => 9,
                }
            }
        }
    }

    let reservation_num = map_reservation(reservation);

    debug_assert!(reservation_num < RESERVATION_OR_NONE_COUNT);
    debug_assert!(reservation_num >= 0);

    [reservation_num]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_reservation_or_none() {
        assert_eq!(encode_reservation_or_none(None), [0]);

        assert_eq!(encode_reservation_or_none(Some(FdoReservation::Healthy)), [1]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::Wedding)), [2]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::DiamondsSolo)), [3]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::HeartsSolo)), [4]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::SpadesSolo)), [5]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::ClubsSolo)), [6]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::QueensSolo)), [7]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::JacksSolo)), [8]);
        assert_eq!(encode_reservation_or_none(Some(FdoReservation::TrumplessSolo)), [9]);
    }
}