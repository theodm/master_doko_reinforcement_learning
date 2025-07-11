use rs_full_doko::announcement::announcement::FdoAnnouncement;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::reservation::reservation::{FdoReservation, FdoVisibleReservation};

pub fn encode_reservation_or_card_or_pi_announcement(
    announcement: Option<FdoAnnouncement>
) -> [i64; 1] {
    match announcement {
        None => [37],
        Some(announcement) => {
            match announcement {
                FdoAnnouncement::ReContra => [38],
                FdoAnnouncement::CounterReContra => [38],
                FdoAnnouncement::No90 => [39],
                FdoAnnouncement::No60 => [40],
                FdoAnnouncement::No30 => [41],
                FdoAnnouncement::Black => [42],
                FdoAnnouncement::NoAnnouncement => panic!("should not happen")
            }
        }
    }
}

pub fn encode_reservation_or_card_or_pi_reservation(
    reservation: Option<FdoReservation>
) -> [i64; 1] {
    match reservation {
        None => [36],
        Some(reservation) => {
            match reservation {
                FdoReservation::Healthy => [25],
                FdoReservation::Wedding => [26],
                FdoReservation::DiamondsSolo => [27],
                FdoReservation::HeartsSolo => [28],
                FdoReservation::SpadesSolo => [29],
                FdoReservation::ClubsSolo => [30],
                FdoReservation::QueensSolo => [31],
                FdoReservation::JacksSolo => [32],
                FdoReservation::TrumplessSolo => [33],
                // [34] bleibt frei
                // [35] bleibt frei
            }
        }
    }
}

pub fn encode_reservation_or_card_or_reservation(
    reservation: Option<FdoVisibleReservation>
) -> [i64; 1] {
    match reservation {
        None => panic!("should not happen at the time"),
        Some(reservation) => {
            match reservation {
                FdoVisibleReservation::Healthy => [25],
                FdoVisibleReservation::Wedding => [26],
                FdoVisibleReservation::DiamondsSolo => [27],
                FdoVisibleReservation::HeartsSolo => [28],
                FdoVisibleReservation::SpadesSolo => [29],
                FdoVisibleReservation::ClubsSolo => [30],
                FdoVisibleReservation::QueensSolo => [31],
                FdoVisibleReservation::JacksSolo => [32],
                FdoVisibleReservation::TrumplessSolo => [33],
                FdoVisibleReservation::NotRevealed => [34],
                FdoVisibleReservation::NoneYet => [35]
            }
        }
    }

}

pub fn encode_reservation_or_card_or_none_card(
    card: Option<FdoCard>
) -> [i64; 1] {
    match card {
        None => [0],
        Some(card) => {
            match card {
                FdoCard::HeartTen => [1],

                FdoCard::ClubQueen => [2],
                FdoCard::SpadeQueen => [3],
                FdoCard::HeartQueen => [4],
                FdoCard::DiamondQueen => [5],

                FdoCard::ClubJack => [6],
                FdoCard::SpadeJack => [7],
                FdoCard::HeartJack => [8],
                FdoCard::DiamondJack => [9],

                FdoCard::DiamondAce => [10],
                FdoCard::DiamondTen => [11],
                FdoCard::DiamondKing => [12],
                FdoCard::DiamondNine => [13],

                FdoCard::ClubAce => [14],
                FdoCard::ClubTen => [15],
                FdoCard::ClubKing => [16],
                FdoCard::ClubNine => [17],

                FdoCard::SpadeAce => [18],
                FdoCard::SpadeTen => [19],
                FdoCard::SpadeKing => [20],
                FdoCard::SpadeNine => [21],

                FdoCard::HeartAce => [22],
                FdoCard::HeartKing => [23],
                FdoCard::HeartNine => [24],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_full_doko::card::cards::FdoCard;
    use rs_full_doko::reservation::reservation::FdoVisibleReservation;

    #[test]
    fn test_encode_reservation_or_card_or_reservation() {
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::Healthy)), [25]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::Wedding)), [26]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::DiamondsSolo)), [27]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::HeartsSolo)), [28]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::SpadesSolo)), [29]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::ClubsSolo)), [30]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::QueensSolo)), [31]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::JacksSolo)), [32]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::TrumplessSolo)), [33]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::NotRevealed)), [34]);
        assert_eq!(encode_reservation_or_card_or_reservation(Some(FdoVisibleReservation::NoneYet)), [0]);
    }

    #[test]
    #[should_panic(expected = "should not happen at the time")]
    fn test_encode_reservation_or_card_or_reservation_none() {
        encode_reservation_or_card_or_reservation(None);
    }

    #[test]
    fn test_encode_reservation_or_card_or_none_card() {
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartTen)), [1]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubQueen)), [2]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeQueen)), [3]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartQueen)), [4]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondQueen)), [5]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubJack)), [6]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeJack)), [7]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartJack)), [8]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondJack)), [9]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondAce)), [10]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondTen)), [11]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondKing)), [12]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::DiamondNine)), [13]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubAce)), [14]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubTen)), [15]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubKing)), [16]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::ClubNine)), [17]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeAce)), [18]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeTen)), [19]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeKing)), [20]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::SpadeNine)), [21]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartAce)), [22]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartKing)), [23]);
        assert_eq!(encode_reservation_or_card_or_none_card(Some(FdoCard::HeartNine)), [24]);
    }

    #[test]
    #[should_panic(expected = "should not happen at the time")]
    fn test_encode_reservation_or_card_or_none_card_none() {
        encode_reservation_or_card_or_none_card(None);
    }
}