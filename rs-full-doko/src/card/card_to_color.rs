use crate::basic::color::FdoColor;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;


/// Gibt die Farbe der Karte [card] basierend auf dem Spieltyp zurück.
///
/// z.B.:
/// (FdoCard::DiamondNine, FdoGameType::Normal) => FdoColor::Trump
/// (FdoCard::DiamondTen, FdoGameType::HeartsSolo) => FdoColor::Diamond
pub fn card_to_color(
    card: FdoCard,
    game_type: FdoGameType,
) -> FdoColor {
    /// Im Normalspiel.
    fn card_to_color_normal(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Trump,
            FdoCard::DiamondTen => FdoColor::Trump,
            FdoCard::DiamondJack => FdoColor::Trump,
            FdoCard::DiamondQueen => FdoColor::Trump,
            FdoCard::DiamondKing => FdoColor::Trump,
            FdoCard::DiamondAce => FdoColor::Trump,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Trump,
            FdoCard::HeartJack => FdoColor::Trump,
            FdoCard::HeartQueen => FdoColor::Trump,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Trump,
            FdoCard::SpadeQueen => FdoColor::Trump,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Trump,
            FdoCard::ClubQueen => FdoColor::Trump,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    /// Im Herz-Solo.
    fn card_to_color_heart(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Trump,
            FdoCard::DiamondQueen => FdoColor::Trump,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Trump,
            FdoCard::HeartTen => FdoColor::Trump,
            FdoCard::HeartJack => FdoColor::Trump,
            FdoCard::HeartQueen => FdoColor::Trump,
            FdoCard::HeartKing => FdoColor::Trump,
            FdoCard::HeartAce => FdoColor::Trump,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Trump,
            FdoCard::SpadeQueen => FdoColor::Trump,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Trump,
            FdoCard::ClubQueen => FdoColor::Trump,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    /// Im Pik-Solo.
    fn card_to_color_spade(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Trump,
            FdoCard::DiamondQueen => FdoColor::Trump,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Trump,
            FdoCard::HeartJack => FdoColor::Trump,
            FdoCard::HeartQueen => FdoColor::Trump,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Trump,
            FdoCard::SpadeTen => FdoColor::Trump,
            FdoCard::SpadeJack => FdoColor::Trump,
            FdoCard::SpadeQueen => FdoColor::Trump,
            FdoCard::SpadeKing => FdoColor::Trump,
            FdoCard::SpadeAce => FdoColor::Trump,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Trump,
            FdoCard::ClubQueen => FdoColor::Trump,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    // Im Kreuz-Solo.
    fn card_to_color_clubs(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Trump,
            FdoCard::DiamondQueen => FdoColor::Trump,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Trump,
            FdoCard::HeartJack => FdoColor::Trump,
            FdoCard::HeartQueen => FdoColor::Trump,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Trump,
            FdoCard::SpadeQueen => FdoColor::Trump,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Trump,
            FdoCard::ClubTen => FdoColor::Trump,
            FdoCard::ClubJack => FdoColor::Trump,
            FdoCard::ClubQueen => FdoColor::Trump,
            FdoCard::ClubKing => FdoColor::Trump,
            FdoCard::ClubAce => FdoColor::Trump
        }
    }

    fn card_to_color_trumpless(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Diamond,
            FdoCard::DiamondQueen => FdoColor::Diamond,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Heart,
            FdoCard::HeartJack => FdoColor::Heart,
            FdoCard::HeartQueen => FdoColor::Heart,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Spade,
            FdoCard::SpadeQueen => FdoColor::Spade,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Club,
            FdoCard::ClubQueen => FdoColor::Club,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    fn card_to_color_queens(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Diamond,
            FdoCard::DiamondQueen => FdoColor::Trump,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Heart,
            FdoCard::HeartJack => FdoColor::Heart,
            FdoCard::HeartQueen => FdoColor::Trump,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Spade,
            FdoCard::SpadeQueen => FdoColor::Trump,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Club,
            FdoCard::ClubQueen => FdoColor::Trump,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    fn card_to_color_jacks(card: FdoCard) -> FdoColor {
        match card {
            FdoCard::DiamondNine => FdoColor::Diamond,
            FdoCard::DiamondTen => FdoColor::Diamond,
            FdoCard::DiamondJack => FdoColor::Trump,
            FdoCard::DiamondQueen => FdoColor::Diamond,
            FdoCard::DiamondKing => FdoColor::Diamond,
            FdoCard::DiamondAce => FdoColor::Diamond,

            FdoCard::HeartNine => FdoColor::Heart,
            FdoCard::HeartTen => FdoColor::Heart,
            FdoCard::HeartJack => FdoColor::Trump,
            FdoCard::HeartQueen => FdoColor::Heart,
            FdoCard::HeartKing => FdoColor::Heart,
            FdoCard::HeartAce => FdoColor::Heart,

            FdoCard::SpadeNine => FdoColor::Spade,
            FdoCard::SpadeTen => FdoColor::Spade,
            FdoCard::SpadeJack => FdoColor::Trump,
            FdoCard::SpadeQueen => FdoColor::Spade,
            FdoCard::SpadeKing => FdoColor::Spade,
            FdoCard::SpadeAce => FdoColor::Spade,

            FdoCard::ClubNine => FdoColor::Club,
            FdoCard::ClubTen => FdoColor::Club,
            FdoCard::ClubJack => FdoColor::Trump,
            FdoCard::ClubQueen => FdoColor::Club,
            FdoCard::ClubKing => FdoColor::Club,
            FdoCard::ClubAce => FdoColor::Club
        }
    }

    match game_type {
        FdoGameType::Normal => card_to_color_normal(card),
        // Bei Hochzeit gelten die gleichen Regeln wie im Normalspiel.
        FdoGameType::Wedding => card_to_color_normal(card),
        // Bei einem Karo-Solo gelten die gleichen Regeln wie im Normalspiel.
        FdoGameType::DiamondsSolo => card_to_color_normal(card),
        FdoGameType::HeartsSolo => card_to_color_heart(card),
        FdoGameType::SpadesSolo => card_to_color_spade(card),
        FdoGameType::ClubsSolo => card_to_color_clubs(card),

        FdoGameType::TrumplessSolo => card_to_color_trumpless(card),

        FdoGameType::QueensSolo => card_to_color_queens(card),
        FdoGameType::JacksSolo => card_to_color_jacks(card)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_card_to_color() {
        // Normalspiel
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::Normal) == FdoColor::Trump);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::Normal) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::Normal) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::Normal) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::Normal) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::Normal) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::Normal) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::Normal) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::Normal) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::Normal) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::Normal) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::Normal) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::Normal) == FdoColor::Club);

        // Hochzeit
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::Wedding) == FdoColor::Trump);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::Wedding) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::Wedding) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::Wedding) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::Wedding) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::Wedding) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::Wedding) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::Wedding) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::Wedding) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::Wedding) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::Wedding) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::Wedding) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::Wedding) == FdoColor::Club);

        // Karo-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::DiamondsSolo) == FdoColor::Trump);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::DiamondsSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::DiamondsSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::DiamondsSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::DiamondsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::DiamondsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::DiamondsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::DiamondsSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::DiamondsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::DiamondsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::DiamondsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::DiamondsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::DiamondsSolo) == FdoColor::Club);

        // Herz-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::HeartsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::HeartsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::HeartsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::HeartsSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::HeartsSolo) == FdoColor::Trump);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::HeartsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::HeartsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::HeartsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::HeartsSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::HeartsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::HeartsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::HeartsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::HeartsSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::HeartsSolo) == FdoColor::Club);

        // Pik-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::SpadesSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::SpadesSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::SpadesSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::SpadesSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::SpadesSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::SpadesSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::SpadesSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::SpadesSolo) == FdoColor::Trump);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::SpadesSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::SpadesSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::SpadesSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::SpadesSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::SpadesSolo) == FdoColor::Club);

        // Kreuz-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::ClubsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::ClubsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::ClubsSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::ClubsSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::ClubsSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::ClubsSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::ClubsSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::ClubsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::ClubsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::ClubsSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::ClubsSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::ClubsSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::ClubsSolo) == FdoColor::Trump);

        // Trumpflos
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::TrumplessSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::TrumplessSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::TrumplessSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::TrumplessSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::TrumplessSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::TrumplessSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::TrumplessSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::TrumplessSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::TrumplessSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::TrumplessSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::TrumplessSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::TrumplessSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::TrumplessSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::TrumplessSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::TrumplessSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::TrumplessSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::TrumplessSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::TrumplessSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::TrumplessSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::TrumplessSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::TrumplessSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::TrumplessSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::TrumplessSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::TrumplessSolo) == FdoColor::Club);

        // Damen-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::QueensSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::QueensSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::QueensSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::QueensSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::QueensSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::QueensSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::QueensSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::QueensSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::QueensSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::QueensSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::QueensSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::QueensSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::QueensSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::QueensSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::QueensSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::QueensSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::QueensSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::QueensSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::QueensSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::QueensSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::QueensSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::QueensSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::QueensSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::QueensSolo) == FdoColor::Club);

        // Buben-Solo
        assert!(card_to_color(FdoCard::DiamondNine, FdoGameType::JacksSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondTen, FdoGameType::JacksSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondJack, FdoGameType::JacksSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::DiamondQueen, FdoGameType::JacksSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondKing, FdoGameType::JacksSolo) == FdoColor::Diamond);
        assert!(card_to_color(FdoCard::DiamondAce, FdoGameType::JacksSolo) == FdoColor::Diamond);

        assert!(card_to_color(FdoCard::HeartNine, FdoGameType::JacksSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartTen, FdoGameType::JacksSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartJack, FdoGameType::JacksSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::HeartQueen, FdoGameType::JacksSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartKing, FdoGameType::JacksSolo) == FdoColor::Heart);
        assert!(card_to_color(FdoCard::HeartAce, FdoGameType::JacksSolo) == FdoColor::Heart);

        assert!(card_to_color(FdoCard::SpadeNine, FdoGameType::JacksSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeTen, FdoGameType::JacksSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeJack, FdoGameType::JacksSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::SpadeQueen, FdoGameType::JacksSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeKing, FdoGameType::JacksSolo) == FdoColor::Spade);
        assert!(card_to_color(FdoCard::SpadeAce, FdoGameType::JacksSolo) == FdoColor::Spade);

        assert!(card_to_color(FdoCard::ClubNine, FdoGameType::JacksSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubTen, FdoGameType::JacksSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubJack, FdoGameType::JacksSolo) == FdoColor::Trump);
        assert!(card_to_color(FdoCard::ClubQueen, FdoGameType::JacksSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubKing, FdoGameType::JacksSolo) == FdoColor::Club);
        assert!(card_to_color(FdoCard::ClubAce, FdoGameType::JacksSolo) == FdoColor::Club);



    }
}