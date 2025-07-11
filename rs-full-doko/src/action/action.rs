use crate::announcement::announcement::FdoAnnouncement;
use crate::card::cards::FdoCard;
use crate::reservation::reservation::FdoReservation;
use std::fmt::Debug;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCountMacro, EnumIter, Hash)]
pub enum FdoAction {
    CardDiamondNine = 1 << 0,
    CardDiamondTen = 1 << 1,
    CardDiamondJack = 1 << 2,
    CardDiamondQueen = 1 << 3,
    CardDiamondKing = 1 << 4,
    CardDiamondAce = 1 << 5,

    CardHeartNine = 1 << 6,
    CardHeartTen = 1 << 7,
    CardHeartJack = 1 << 8,
    CardHeartQueen = 1 << 9,
    CardHeartKing = 1 << 10,
    CardHeartAce = 1 << 11,

    CardClubNine = 1 << 12,
    CardClubTen = 1 << 13,
    CardClubJack = 1 << 14,
    CardClubQueen = 1 << 15,
    CardClubKing = 1 << 16,
    CardClubAce = 1 << 17,

    CardSpadeNine = 1 << 18,
    CardSpadeTen = 1 << 19,
    CardSpadeJack = 1 << 20,
    CardSpadeQueen = 1 << 21,
    CardSpadeKing = 1 << 22,
    CardSpadeAce = 1 << 23,

    ReservationHealthy = 1 << 24,
    ReservationWedding = 1 << 25,
    ReservationDiamondsSolo = 1 << 26,
    ReservationHeartsSolo = 1 << 27,
    ReservationSpadesSolo = 1 << 28,
    ReservationClubsSolo = 1 << 29,
    ReservationTrumplessSolo = 1 << 30,
    ReservationQueensSolo = 1 << 31,
    ReservationJacksSolo = 1 << 32,

    AnnouncementReContra = 1 << 33,
    AnnouncementNo90 = 1 << 34,
    AnnouncementNo60 = 1 << 35,
    AnnouncementNo30 = 1 << 36,
    AnnouncementBlack = 1 << 37,
    NoAnnouncement = 1 << 38,
}

impl FdoAction {
    pub fn to_index(&self) -> usize {
        match self {
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
            FdoAction::NoAnnouncement => 38,
        }
    }

    pub fn from_index(index: usize) -> FdoAction {
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

            _ => panic!("Invalid action index: {}", index),
        }
    }
}

impl ToString for FdoAction {
    fn to_string(&self) -> String {
        match self {
            FdoAction::CardDiamondNine => "♦9".to_string(),
            FdoAction::CardDiamondTen => "♦10".to_string(),
            FdoAction::CardDiamondJack => "♦J".to_string(),
            FdoAction::CardDiamondQueen => "♦Q".to_string(),
            FdoAction::CardDiamondKing => "♦K".to_string(),
            FdoAction::CardDiamondAce => "♦A".to_string(),

            FdoAction::CardHeartNine => "♥9".to_string(),
            FdoAction::CardHeartTen => "♥10".to_string(),
            FdoAction::CardHeartJack => "♥J".to_string(),
            FdoAction::CardHeartQueen => "♥Q".to_string(),
            FdoAction::CardHeartKing => "♥K".to_string(),
            FdoAction::CardHeartAce => "♥A".to_string(),

            FdoAction::CardClubNine => "♣9".to_string(),
            FdoAction::CardClubTen => "♣10".to_string(),
            FdoAction::CardClubJack => "♣J".to_string(),
            FdoAction::CardClubQueen => "♣Q".to_string(),
            FdoAction::CardClubKing => "♣K".to_string(),
            FdoAction::CardClubAce => "♣A".to_string(),

            FdoAction::CardSpadeNine => "♠9".to_string(),
            FdoAction::CardSpadeTen => "♠10".to_string(),
            FdoAction::CardSpadeJack => "♠J".to_string(),
            FdoAction::CardSpadeQueen => "♠Q".to_string(),
            FdoAction::CardSpadeKing => "♠K".to_string(),
            FdoAction::CardSpadeAce => "♠A".to_string(),

            FdoAction::ReservationHealthy => "Gesund".to_string(),
            FdoAction::ReservationWedding => "Hochzeit".to_string(),
            FdoAction::ReservationDiamondsSolo => "♦-Solo".to_string(),
            FdoAction::ReservationHeartsSolo => "♥-Solo".to_string(),
            FdoAction::ReservationSpadesSolo => "♠-Solo".to_string(),
            FdoAction::ReservationClubsSolo => "♣-Solo".to_string(),
            FdoAction::ReservationTrumplessSolo => "T-Solo".to_string(),
            FdoAction::ReservationQueensSolo => "Q-Solo".to_string(),
            FdoAction::ReservationJacksSolo => "J-Solo".to_string(),

            FdoAction::AnnouncementReContra => "Re/Contra".to_string(),
            FdoAction::AnnouncementNo90 => "Keine 90".to_string(),
            FdoAction::AnnouncementNo60 => "Keine 60".to_string(),
            FdoAction::AnnouncementNo30 => "Keine 30".to_string(),
            FdoAction::AnnouncementBlack => "Schwarz".to_string(),
            FdoAction::NoAnnouncement => "Keine Ansage".to_string(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FdoActionType {
    Card(FdoCard),
    Reservation(FdoReservation),
    Announcement(FdoAnnouncement),
}

impl Debug for FdoActionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoActionType::Card(card) => write!(f, "FdoActionType::Card({:?})", card),
            FdoActionType::Reservation(reservation) => {
                write!(f, "FdoActionType::Reservation({:?})", reservation)
            }
            FdoActionType::Announcement(announcement) => {
                write!(f, "FdoActionType::Announcement({:?})", announcement)
            }
        }
    }
}

pub fn action_to_type(action: FdoAction) -> FdoActionType {
    let is_card_action = action as usize <= FdoAction::CardSpadeAce as usize;
    let is_reservation_action = action as usize <= FdoAction::ReservationJacksSolo as usize;
    let is_announcement_action = action as usize <= FdoAction::NoAnnouncement as usize;

    if is_card_action {
        match action {
            FdoAction::CardDiamondNine => FdoActionType::Card(FdoCard::DiamondNine),
            FdoAction::CardDiamondTen => FdoActionType::Card(FdoCard::DiamondTen),
            FdoAction::CardDiamondJack => FdoActionType::Card(FdoCard::DiamondJack),
            FdoAction::CardDiamondQueen => FdoActionType::Card(FdoCard::DiamondQueen),
            FdoAction::CardDiamondKing => FdoActionType::Card(FdoCard::DiamondKing),
            FdoAction::CardDiamondAce => FdoActionType::Card(FdoCard::DiamondAce),

            FdoAction::CardHeartNine => FdoActionType::Card(FdoCard::HeartNine),
            FdoAction::CardHeartTen => FdoActionType::Card(FdoCard::HeartTen),
            FdoAction::CardHeartJack => FdoActionType::Card(FdoCard::HeartJack),
            FdoAction::CardHeartQueen => FdoActionType::Card(FdoCard::HeartQueen),
            FdoAction::CardHeartKing => FdoActionType::Card(FdoCard::HeartKing),
            FdoAction::CardHeartAce => FdoActionType::Card(FdoCard::HeartAce),

            FdoAction::CardClubNine => FdoActionType::Card(FdoCard::ClubNine),
            FdoAction::CardClubTen => FdoActionType::Card(FdoCard::ClubTen),
            FdoAction::CardClubJack => FdoActionType::Card(FdoCard::ClubJack),
            FdoAction::CardClubQueen => FdoActionType::Card(FdoCard::ClubQueen),
            FdoAction::CardClubKing => FdoActionType::Card(FdoCard::ClubKing),
            FdoAction::CardClubAce => FdoActionType::Card(FdoCard::ClubAce),

            FdoAction::CardSpadeNine => FdoActionType::Card(FdoCard::SpadeNine),
            FdoAction::CardSpadeTen => FdoActionType::Card(FdoCard::SpadeTen),
            FdoAction::CardSpadeJack => FdoActionType::Card(FdoCard::SpadeJack),
            FdoAction::CardSpadeQueen => FdoActionType::Card(FdoCard::SpadeQueen),
            FdoAction::CardSpadeKing => FdoActionType::Card(FdoCard::SpadeKing),
            FdoAction::CardSpadeAce => FdoActionType::Card(FdoCard::SpadeAce),

            _ => panic!("Invalid card action: {:?}", action),
        }
    } else if is_reservation_action {
        match action {
            FdoAction::ReservationHealthy => FdoActionType::Reservation(FdoReservation::Healthy),
            FdoAction::ReservationWedding => FdoActionType::Reservation(FdoReservation::Wedding),
            FdoAction::ReservationDiamondsSolo => {
                FdoActionType::Reservation(FdoReservation::DiamondsSolo)
            }
            FdoAction::ReservationHeartsSolo => {
                FdoActionType::Reservation(FdoReservation::HeartsSolo)
            }
            FdoAction::ReservationSpadesSolo => {
                FdoActionType::Reservation(FdoReservation::SpadesSolo)
            }
            FdoAction::ReservationClubsSolo => {
                FdoActionType::Reservation(FdoReservation::ClubsSolo)
            }
            FdoAction::ReservationTrumplessSolo => {
                FdoActionType::Reservation(FdoReservation::TrumplessSolo)
            }
            FdoAction::ReservationQueensSolo => {
                FdoActionType::Reservation(FdoReservation::QueensSolo)
            }
            FdoAction::ReservationJacksSolo => {
                FdoActionType::Reservation(FdoReservation::JacksSolo)
            }

            _ => panic!("Invalid reservation action: {:?}", action),
        }
    } else if is_announcement_action {
        match action {
            FdoAction::AnnouncementReContra => {
                FdoActionType::Announcement(FdoAnnouncement::ReContra)
            }
            FdoAction::AnnouncementNo90 => FdoActionType::Announcement(FdoAnnouncement::No90),
            FdoAction::AnnouncementNo60 => FdoActionType::Announcement(FdoAnnouncement::No60),
            FdoAction::AnnouncementNo30 => FdoActionType::Announcement(FdoAnnouncement::No30),
            FdoAction::AnnouncementBlack => FdoActionType::Announcement(FdoAnnouncement::Black),

            FdoAction::NoAnnouncement => {
                FdoActionType::Announcement(FdoAnnouncement::NoAnnouncement)
            }

            _ => panic!("Invalid announcement action: {:?}", action),
        }
    } else {
        panic!("Invalid action: {:?}", action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_to_type() {
        assert_eq!(
            action_to_type(FdoAction::CardDiamondNine),
            FdoActionType::Card(FdoCard::DiamondNine)
        );
        assert_eq!(
            action_to_type(FdoAction::CardDiamondTen),
            FdoActionType::Card(FdoCard::DiamondTen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardDiamondJack),
            FdoActionType::Card(FdoCard::DiamondJack)
        );
        assert_eq!(
            action_to_type(FdoAction::CardDiamondQueen),
            FdoActionType::Card(FdoCard::DiamondQueen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardDiamondKing),
            FdoActionType::Card(FdoCard::DiamondKing)
        );
        assert_eq!(
            action_to_type(FdoAction::CardDiamondAce),
            FdoActionType::Card(FdoCard::DiamondAce)
        );

        assert_eq!(
            action_to_type(FdoAction::CardHeartNine),
            FdoActionType::Card(FdoCard::HeartNine)
        );
        assert_eq!(
            action_to_type(FdoAction::CardHeartTen),
            FdoActionType::Card(FdoCard::HeartTen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardHeartJack),
            FdoActionType::Card(FdoCard::HeartJack)
        );
        assert_eq!(
            action_to_type(FdoAction::CardHeartQueen),
            FdoActionType::Card(FdoCard::HeartQueen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardHeartKing),
            FdoActionType::Card(FdoCard::HeartKing)
        );
        assert_eq!(
            action_to_type(FdoAction::CardHeartAce),
            FdoActionType::Card(FdoCard::HeartAce)
        );

        assert_eq!(
            action_to_type(FdoAction::CardClubNine),
            FdoActionType::Card(FdoCard::ClubNine)
        );
        assert_eq!(
            action_to_type(FdoAction::CardClubTen),
            FdoActionType::Card(FdoCard::ClubTen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardClubJack),
            FdoActionType::Card(FdoCard::ClubJack)
        );
        assert_eq!(
            action_to_type(FdoAction::CardClubQueen),
            FdoActionType::Card(FdoCard::ClubQueen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardClubKing),
            FdoActionType::Card(FdoCard::ClubKing)
        );
        assert_eq!(
            action_to_type(FdoAction::CardClubAce),
            FdoActionType::Card(FdoCard::ClubAce)
        );

        assert_eq!(
            action_to_type(FdoAction::CardSpadeNine),
            FdoActionType::Card(FdoCard::SpadeNine)
        );
        assert_eq!(
            action_to_type(FdoAction::CardSpadeTen),
            FdoActionType::Card(FdoCard::SpadeTen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardSpadeJack),
            FdoActionType::Card(FdoCard::SpadeJack)
        );
        assert_eq!(
            action_to_type(FdoAction::CardSpadeQueen),
            FdoActionType::Card(FdoCard::SpadeQueen)
        );
        assert_eq!(
            action_to_type(FdoAction::CardSpadeKing),
            FdoActionType::Card(FdoCard::SpadeKing)
        );
        assert_eq!(
            action_to_type(FdoAction::CardSpadeAce),
            FdoActionType::Card(FdoCard::SpadeAce)
        );

        assert_eq!(
            action_to_type(FdoAction::ReservationHealthy),
            FdoActionType::Reservation(FdoReservation::Healthy)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationWedding),
            FdoActionType::Reservation(FdoReservation::Wedding)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationDiamondsSolo),
            FdoActionType::Reservation(FdoReservation::DiamondsSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationHeartsSolo),
            FdoActionType::Reservation(FdoReservation::HeartsSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationSpadesSolo),
            FdoActionType::Reservation(FdoReservation::SpadesSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationClubsSolo),
            FdoActionType::Reservation(FdoReservation::ClubsSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationTrumplessSolo),
            FdoActionType::Reservation(FdoReservation::TrumplessSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationQueensSolo),
            FdoActionType::Reservation(FdoReservation::QueensSolo)
        );
        assert_eq!(
            action_to_type(FdoAction::ReservationJacksSolo),
            FdoActionType::Reservation(FdoReservation::JacksSolo)
        );

        assert_eq!(
            action_to_type(FdoAction::AnnouncementReContra),
            FdoActionType::Announcement(FdoAnnouncement::ReContra)
        );
        assert_eq!(
            action_to_type(FdoAction::AnnouncementNo90),
            FdoActionType::Announcement(FdoAnnouncement::No90)
        );
        assert_eq!(
            action_to_type(FdoAction::AnnouncementNo60),
            FdoActionType::Announcement(FdoAnnouncement::No60)
        );
        assert_eq!(
            action_to_type(FdoAction::AnnouncementNo30),
            FdoActionType::Announcement(FdoAnnouncement::No30)
        );
        assert_eq!(
            action_to_type(FdoAction::AnnouncementBlack),
            FdoActionType::Announcement(FdoAnnouncement::Black)
        );
        assert_eq!(
            action_to_type(FdoAction::NoAnnouncement),
            FdoActionType::Announcement(FdoAnnouncement::NoAnnouncement)
        );
    }
}
