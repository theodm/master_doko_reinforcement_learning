use std::fmt::{Debug, Formatter};
use enumset::EnumSetType;
use serde::{Deserialize, Serialize};

/// Repräsentiert einen gemachten Vorbehalt. Aber es handelt sich immer
/// um den geheimen Vorbehalt, der zunächst nur für den Spieler sichtbar ist.
/// (Andere Spieler können bis zur entsprechenden Aufklärung erst nur zwischen Vorbehalt und gesund
/// unterscheiden)
#[repr(usize)]
#[derive(Hash, strum_macros::EnumCount, EnumSetType, Serialize, Deserialize)]
pub enum FdoReservation {
    Healthy,
    Wedding,

    DiamondsSolo,
    HeartsSolo,
    SpadesSolo,
    ClubsSolo,

    QueensSolo,
    JacksSolo,

    TrumplessSolo,
}

impl ToString for FdoReservation {
    fn to_string(&self) -> String {
        match self {
            FdoReservation::Healthy => "Gesund".to_string(),
            FdoReservation::Wedding => "Hochzeit".to_string(),
            FdoReservation::DiamondsSolo => "Diamond-Solo".to_string(),
            FdoReservation::HeartsSolo => "Heart-Solo".to_string(),
            FdoReservation::SpadesSolo => "Spade-Solo".to_string(),
            FdoReservation::ClubsSolo => "Club-Solo".to_string(),
            FdoReservation::QueensSolo => "Damen-Solo".to_string(),
            FdoReservation::JacksSolo => "Buben-Solo".to_string(),
            FdoReservation::TrumplessSolo => "Fleischloser".to_string(),
        }
    }
}

impl Debug for FdoReservation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoReservation::Healthy => write!(f, "FdoReservation::Healthy"),
            FdoReservation::Wedding => write!(f, "FdoReservation::Wedding"),
            FdoReservation::DiamondsSolo => write!(f, "FdoReservation::DiamondsSolo"),
            FdoReservation::HeartsSolo => write!(f, "FdoReservation::HeartsSolo"),
            FdoReservation::SpadesSolo => write!(f, "FdoReservation::SpadesSolo"),
            FdoReservation::ClubsSolo => write!(f, "FdoReservation::ClubsSolo"),
            FdoReservation::QueensSolo => write!(f, "FdoReservation::QueensSolo"),
            FdoReservation::JacksSolo => write!(f, "FdoReservation::JacksSolo"),
            FdoReservation::TrumplessSolo => write!(f, "FdoReservation::TrumplessSolo"),
        }
    }
}

/// Repräsentiert einen Vorbehalt eines Spielers aus seiner aktuellen Sicht,
/// denn nicht alle Vorbehalte müssen angezeigt werden. Verfällt ein Vorbehalt
/// eines Spielers, weil ein vorheriger Spieler einen höheren Vorbehalt angesagt hat,
/// so wird dieser Vorbehalt nicht aufgedeckt.
#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum FdoVisibleReservation {
    Wedding = 0,
    Healthy = 1,
    NotRevealed = 2,

    DiamondsSolo = 3,
    HeartsSolo = 4,
    SpadesSolo = 5,
    ClubsSolo = 6,

    QueensSolo = 7,
    JacksSolo = 8,

    TrumplessSolo = 9,
    NoneYet = 10,
}

impl Debug for FdoVisibleReservation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoVisibleReservation::Wedding => write!(f, "FdoVisibleReservation::Wedding"),
            FdoVisibleReservation::Healthy => write!(f, "FdoVisibleReservation::Healthy"),
            FdoVisibleReservation::NotRevealed => write!(f, "FdoVisibleReservation::NotRevealed"),
            FdoVisibleReservation::DiamondsSolo => write!(f, "FdoVisibleReservation::DiamondsSolo"),
            FdoVisibleReservation::HeartsSolo => write!(f, "FdoVisibleReservation::HeartsSolo"),
            FdoVisibleReservation::SpadesSolo => write!(f, "FdoVisibleReservation::SpadesSolo"),
            FdoVisibleReservation::ClubsSolo => write!(f, "FdoVisibleReservation::ClubsSolo"),
            FdoVisibleReservation::QueensSolo => write!(f, "FdoVisibleReservation::QueensSolo"),
            FdoVisibleReservation::JacksSolo => write!(f, "FdoVisibleReservation::JacksSolo"),
            FdoVisibleReservation::TrumplessSolo => write!(f, "FdoVisibleReservation::TrumplessSolo"),
            FdoVisibleReservation::NoneYet => write!(f, "FdoVisibleReservation::NoneYet"),
        }
    }
}