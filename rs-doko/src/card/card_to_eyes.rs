use crate::card::cards::DoCard;

/// Gibt den Augenwert der Karte [card] zurück.
pub fn card_to_eyes(card: DoCard) -> u32 {
    match card {
        DoCard::DiamondNine => 0,
        DoCard::DiamondTen => 10,
        DoCard::DiamondJack => 2,
        DoCard::DiamondQueen => 3,
        DoCard::DiamondKing => 4,
        DoCard::DiamondAce => 11,

        DoCard::HeartNine => 0,
        DoCard::HeartTen => 10,
        DoCard::HeartJack => 2,
        DoCard::HeartQueen => 3,
        DoCard::HeartKing => 4,
        DoCard::HeartAce => 11,

        DoCard::SpadeNine => 0,
        DoCard::SpadeTen => 10,
        DoCard::SpadeJack => 2,
        DoCard::SpadeQueen => 3,
        DoCard::SpadeKing => 4,
        DoCard::SpadeAce => 11,

        DoCard::ClubNine => 0,
        DoCard::ClubTen => 10,
        DoCard::ClubJack => 2,
        DoCard::ClubQueen => 3,
        DoCard::ClubKing => 4,
        DoCard::ClubAce => 11,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_to_eyes() {
        assert_eq!(card_to_eyes(DoCard::DiamondNine), 0);
        assert_eq!(card_to_eyes(DoCard::DiamondTen), 10);
        assert_eq!(card_to_eyes(DoCard::DiamondJack), 2);
        assert_eq!(card_to_eyes(DoCard::DiamondQueen), 3);
        assert_eq!(card_to_eyes(DoCard::DiamondKing), 4);
        assert_eq!(card_to_eyes(DoCard::DiamondAce), 11);

        assert_eq!(card_to_eyes(DoCard::HeartNine), 0);
        assert_eq!(card_to_eyes(DoCard::HeartTen), 10);
        assert_eq!(card_to_eyes(DoCard::HeartJack), 2);
        assert_eq!(card_to_eyes(DoCard::HeartQueen), 3);
        assert_eq!(card_to_eyes(DoCard::HeartKing), 4);
        assert_eq!(card_to_eyes(DoCard::HeartAce), 11);

        assert_eq!(card_to_eyes(DoCard::SpadeNine), 0);
        assert_eq!(card_to_eyes(DoCard::SpadeTen), 10);
        assert_eq!(card_to_eyes(DoCard::SpadeJack), 2);
        assert_eq!(card_to_eyes(DoCard::SpadeQueen), 3);
        assert_eq!(card_to_eyes(DoCard::SpadeKing), 4);
        assert_eq!(card_to_eyes(DoCard::SpadeAce), 11);

        assert_eq!(card_to_eyes(DoCard::ClubNine), 0);
        assert_eq!(card_to_eyes(DoCard::ClubTen), 10);
        assert_eq!(card_to_eyes(DoCard::ClubJack), 2);
        assert_eq!(card_to_eyes(DoCard::ClubQueen), 3);
        assert_eq!(card_to_eyes(DoCard::ClubKing), 4);
        assert_eq!(card_to_eyes(DoCard::ClubAce), 11);
    }
}