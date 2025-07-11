use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter, Serialize, Deserialize)]
pub enum FdoGameType {
    Normal,

    // ToDo: Brauchen wir einen eigenen Typ für Hochzeit?
    Wedding,

    DiamondsSolo,
    HeartsSolo,
    SpadesSolo,
    ClubsSolo,

    TrumplessSolo,
    QueensSolo,
    JacksSolo
}

impl Display for FdoGameType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            FdoGameType::Normal => "Normalspiel",
            FdoGameType::Wedding => "Hochzeit",
            FdoGameType::DiamondsSolo => "♦-Solo",
            FdoGameType::HeartsSolo => "♥-Solo",
            FdoGameType::SpadesSolo => "♠-Solo",
            FdoGameType::ClubsSolo => "♣-Solo",
            FdoGameType::TrumplessSolo => "Fleischloser",
            FdoGameType::QueensSolo => "Q-Solo",
            FdoGameType::JacksSolo => "J-Solo"
        }.to_string();

        return write!(f, "{}", str);
    }
}