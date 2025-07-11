use std::fmt::Display;
use std::hash::Hash;
use serde::{Deserialize, Serialize};
use strum_macros::{EnumCount, EnumIter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCount, EnumIter, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum FdoCard {
    DiamondNine = 1 << 0,
    DiamondTen = 1 << 1,
    DiamondJack = 1 << 2,
    DiamondQueen = 1 << 3,
    DiamondKing = 1 << 4,
    DiamondAce = 1 << 5,

    HeartNine = 1 << 6,
    HeartTen = 1 << 7,
    HeartJack = 1 << 8,
    HeartQueen = 1 << 9,
    HeartKing = 1 << 10,
    HeartAce = 1 << 11,

    ClubNine = 1 << 12,
    ClubTen = 1 << 13,
    ClubJack = 1 << 14,
    ClubQueen = 1 << 15,
    ClubKing = 1 << 16,
    ClubAce = 1 << 17,

    SpadeNine = 1 << 18,
    SpadeTen = 1 << 19,
    SpadeJack = 1 << 20,
    SpadeQueen = 1 << 21,
    SpadeKing = 1 << 22,
    SpadeAce = 1 << 23,
}

impl FdoCard {
    /// Erzeugt eine Karte aus einem String.
    ///
    /// z.B. "♦9" => ♦9
    pub fn from_str(
        s: &str
    ) -> FdoCard {
        println!("s: {}", s);
        return match s {
            "♦9" => FdoCard::DiamondNine,
            "♦10" => FdoCard::DiamondTen,
            "♦J" => FdoCard::DiamondJack,
            "♦Q" => FdoCard::DiamondQueen,
            "♦K" => FdoCard::DiamondKing,
            "♦A" => FdoCard::DiamondAce,

            "♥9" => FdoCard::HeartNine,
            "♥10" => FdoCard::HeartTen,
            "♥J" => FdoCard::HeartJack,
            "♥Q" => FdoCard::HeartQueen,
            "♥K" => FdoCard::HeartKing,
            "♥A" => FdoCard::HeartAce,

            "♠9" => FdoCard::SpadeNine,
            "♠10" => FdoCard::SpadeTen,
            "♠J" => FdoCard::SpadeJack,
            "♠Q" => FdoCard::SpadeQueen,
            "♠K" => FdoCard::SpadeKing,
            "♠A" => FdoCard::SpadeAce,


            "♣9" => FdoCard::ClubNine,
            "♣10" => FdoCard::ClubTen,
            "♣J" => FdoCard::ClubJack,
            "♣Q" => FdoCard::ClubQueen,
            "♣K" => FdoCard::ClubKing,
            "♣A" => FdoCard::ClubAce,

            _ => panic!("Unknown card: {}", s)
        }
    }

    /// Erzeugt eine Liste von Karten aus einem String.
    ///
    /// z.B. "♦9 ♦10 ♦J ♦Q ♦K ♦A ♥9 ♥10 ♥J ♥Q ♥K ♥A ♣9 ♣10 ♣J ♣Q ♣K ♣A ♠9 ♠10 ♠J ♠Q ♠K ♠A" => [♦9, ♦10, ♦J, ♦Q, ♦K, ♦A, ♥9, ♥10, ♥J, ♥Q, ♥K, ♥A, ♣9, ♣10, ♣J, ♣Q, ♣K, ♣A, ♠9, ♠10, ♠J, ♠Q, ♠K, ♠A]
    pub fn vec_from_str(
        s: &str
    ) -> Vec<FdoCard> {
        let mut cards: Vec<FdoCard> = Vec::new();

        for card_str in s.split_whitespace() {
            let card = FdoCard::from_str(card_str);
            cards.push(card);
        }

        return cards;
    }

}


impl Display for FdoCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            FdoCard::DiamondNine => "♦9",
            FdoCard::DiamondTen => "♦10",
            FdoCard::DiamondJack => "♦J",
            FdoCard::DiamondQueen => "♦Q",
            FdoCard::DiamondKing => "♦K",
            FdoCard::DiamondAce => "♦A",
            FdoCard::HeartNine => "♥9",
            FdoCard::HeartTen => "♥10",
            FdoCard::HeartJack => "♥J",
            FdoCard::HeartQueen => "♥Q",
            FdoCard::HeartKing => "♥K",
            FdoCard::HeartAce => "♥A",
            FdoCard::ClubNine => "♣9",
            FdoCard::ClubTen => "♣10",
            FdoCard::ClubJack => "♣J",
            FdoCard::ClubQueen => "♣Q",
            FdoCard::ClubKing => "♣K",
            FdoCard::ClubAce => "♣A",
            FdoCard::SpadeNine => "♠9",
            FdoCard::SpadeTen => "♠10",
            FdoCard::SpadeJack => "♠J",
            FdoCard::SpadeQueen => "♠Q",
            FdoCard::SpadeKing => "♠K",
            FdoCard::SpadeAce => "♠A"
        }.to_string();

        return write!(f, "{}", str);
    }
}

