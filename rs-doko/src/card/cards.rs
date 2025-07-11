use std::fmt::Display;
use std::hash::Hash;
use strum_macros::EnumCount;

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCount, Hash)]
#[repr(usize)]
pub enum DoCard {
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

impl Display for DoCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            DoCard::DiamondNine => "♦9",
            DoCard::DiamondTen => "♦10",
            DoCard::DiamondJack => "♦J",
            DoCard::DiamondQueen => "♦Q",
            DoCard::DiamondKing => "♦K",
            DoCard::DiamondAce => "♦A",
            DoCard::HeartNine => "♥9",
            DoCard::HeartTen => "♥10",
            DoCard::HeartJack => "♥J",
            DoCard::HeartQueen => "♥Q",
            DoCard::HeartKing => "♥K",
            DoCard::HeartAce => "♥A",
            DoCard::ClubNine => "♣9",
            DoCard::ClubTen => "♣10",
            DoCard::ClubJack => "♣J",
            DoCard::ClubQueen => "♣Q",
            DoCard::ClubKing => "♣K",
            DoCard::ClubAce => "♣A",
            DoCard::SpadeNine => "♠9",
            DoCard::SpadeTen => "♠10",
            DoCard::SpadeJack => "♠J",
            DoCard::SpadeQueen => "♠Q",
            DoCard::SpadeKing => "♠K",
            DoCard::SpadeAce => "♠A"
        }.to_string();

        return write!(f, "{}", str);
    }
}