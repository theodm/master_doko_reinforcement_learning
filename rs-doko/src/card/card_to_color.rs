use crate::card::cards::DoCard;
use crate::basic::color::DoColor;

/// Gibt die Farbe der Karte [card] im Normalspiel zurück.
///
/// z.B.:
/// DoCard::DiamondNine => DoColor::Trump
/// DoCard::ClubNine => DoColor::Club
pub fn card_to_color_in_normal_game(card: DoCard) -> DoColor {
    match card {
        // Im Normalspiel sind die Karo-Karten Trumpf
        DoCard::DiamondNine => DoColor::Trump,
        DoCard::DiamondTen => DoColor::Trump,
        DoCard::DiamondJack => DoColor::Trump,
        DoCard::DiamondQueen => DoColor::Trump,
        DoCard::DiamondKing => DoColor::Trump,
        DoCard::DiamondAce => DoColor::Trump,

        DoCard::HeartNine => DoColor::Heart,
        DoCard::HeartTen => DoColor::Trump,
        DoCard::HeartJack => DoColor::Trump,
        DoCard::HeartQueen => DoColor::Trump,
        DoCard::HeartKing => DoColor::Heart,
        DoCard::HeartAce => DoColor::Heart,

        DoCard::SpadeNine => DoColor::Spade,
        DoCard::SpadeTen => DoColor::Spade,
        DoCard::SpadeJack => DoColor::Trump,
        DoCard::SpadeQueen => DoColor::Trump,
        DoCard::SpadeKing => DoColor::Spade,
        DoCard::SpadeAce => DoColor::Spade,

        DoCard::ClubNine => DoColor::Club,
        DoCard::ClubTen => DoColor::Club,
        DoCard::ClubJack => DoColor::Trump,
        DoCard::ClubQueen => DoColor::Trump,
        DoCard::ClubKing => DoColor::Club,
        DoCard::ClubAce => DoColor::Club
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_card_to_color_in_normal_game() {
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondNine), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondTen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondJack), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondQueen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondKing), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::DiamondAce), DoColor::Trump);

        assert_eq!(card_to_color_in_normal_game(DoCard::HeartNine), DoColor::Heart);
        assert_eq!(card_to_color_in_normal_game(DoCard::HeartTen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::HeartJack), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::HeartQueen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::HeartKing), DoColor::Heart);
        assert_eq!(card_to_color_in_normal_game(DoCard::HeartAce), DoColor::Heart);

        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeNine), DoColor::Spade);
        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeTen), DoColor::Spade);
        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeJack), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeQueen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeKing), DoColor::Spade);
        assert_eq!(card_to_color_in_normal_game(DoCard::SpadeAce), DoColor::Spade);

        assert_eq!(card_to_color_in_normal_game(DoCard::ClubNine), DoColor::Club);
        assert_eq!(card_to_color_in_normal_game(DoCard::ClubTen), DoColor::Club);
        assert_eq!(card_to_color_in_normal_game(DoCard::ClubJack), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::ClubQueen), DoColor::Trump);
        assert_eq!(card_to_color_in_normal_game(DoCard::ClubKing), DoColor::Club);
        assert_eq!(card_to_color_in_normal_game(DoCard::ClubAce), DoColor::Club);
    }
}