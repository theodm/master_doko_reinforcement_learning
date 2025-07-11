use rs_full_doko::action::action::FdoAction;

pub fn index2action(
    index: usize
) -> FdoAction {
    match index {
        0 => FdoAction::CardDiamondNine,
        1 => FdoAction::CardDiamondTen,
        2 => FdoAction::CardDiamondJack,
        3 => FdoAction::CardDiamondQueen,
        4 => FdoAction::CardDiamondKing,
        5 => FdoAction::CardDiamondAce,

        6 => FdoAction::CardHeartNine,
        7 => FdoAction::CardHeartTen,
        8 => FdoAction::CardHeartJack,
        9 => FdoAction::CardHeartQueen,
        10 => FdoAction::CardHeartKing,
        11 => FdoAction::CardHeartAce,

        12 => FdoAction::CardClubNine,
        13 => FdoAction::CardClubTen,
        14 => FdoAction::CardClubJack,
        15 => FdoAction::CardClubQueen,
        16 => FdoAction::CardClubKing,
        17 => FdoAction::CardClubAce,

        18 => FdoAction::CardSpadeNine,
        19 => FdoAction::CardSpadeTen,
        20 => FdoAction::CardSpadeJack,
        21 => FdoAction::CardSpadeQueen,
        22 => FdoAction::CardSpadeKing,
        23 => FdoAction::CardSpadeAce,

        24 => FdoAction::ReservationHealthy,
        25 => FdoAction::ReservationWedding,
        26 => FdoAction::ReservationDiamondsSolo,
        27 => FdoAction::ReservationHeartsSolo,
        28 => FdoAction::ReservationSpadesSolo,
        29 => FdoAction::ReservationClubsSolo,
        30 => FdoAction::ReservationTrumplessSolo,
        31 => FdoAction::ReservationQueensSolo,
        32 => FdoAction::ReservationJacksSolo,

        33 => FdoAction::AnnouncementReContra,
        34 => FdoAction::AnnouncementNo90,
        35 => FdoAction::AnnouncementNo60,
        36 => FdoAction::AnnouncementNo30,
        37 => FdoAction::AnnouncementBlack,
        38 => FdoAction::NoAnnouncement,

        _ => panic!("Invalid index")
    }
}

pub fn action2index(
    action: FdoAction
) -> usize {
    match action {
        FdoAction::CardDiamondNine => 0,
        FdoAction::CardDiamondTen => 1,
        FdoAction::CardDiamondJack => 2,
        FdoAction::CardDiamondQueen => 3,
        FdoAction::CardDiamondKing => 4,
        FdoAction::CardDiamondAce => 5,
        FdoAction::CardHeartNine => 6,
        FdoAction::CardHeartTen => 7,
        FdoAction::CardHeartJack => 8,
        FdoAction::CardHeartQueen => 9,
        FdoAction::CardHeartKing => 10,
        FdoAction::CardHeartAce => 11,
        FdoAction::CardClubNine => 12,
        FdoAction::CardClubTen => 13,
        FdoAction::CardClubJack => 14,
        FdoAction::CardClubQueen => 15,
        FdoAction::CardClubKing => 16,
        FdoAction::CardClubAce => 17,
        FdoAction::CardSpadeNine => 18,
        FdoAction::CardSpadeTen => 19,
        FdoAction::CardSpadeJack => 20,
        FdoAction::CardSpadeQueen => 21,
        FdoAction::CardSpadeKing => 22,
        FdoAction::CardSpadeAce => 23,
        FdoAction::ReservationHealthy => 24,
        FdoAction::ReservationWedding => 25,
        FdoAction::ReservationDiamondsSolo => 26,
        FdoAction::ReservationHeartsSolo => 27,
        FdoAction::ReservationSpadesSolo => 28,
        FdoAction::ReservationClubsSolo => 29,
        FdoAction::ReservationTrumplessSolo => 30,
        FdoAction::ReservationQueensSolo => 31,
        FdoAction::ReservationJacksSolo => 32,
        FdoAction::AnnouncementReContra => 33,
        FdoAction::AnnouncementNo90 => 34,
        FdoAction::AnnouncementNo60 => 35,
        FdoAction::AnnouncementNo30 => 36,
        FdoAction::AnnouncementBlack => 37,
        FdoAction::NoAnnouncement => 38
    }

}