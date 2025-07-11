use crate::card::cards::{ FdoCard};
use crate::game_type::game_type::FdoGameType;


/// Gibt Bit-Masken für die jeweiligen Farben in einem Spieltyp zurück. Damit kann
/// eine Hand auf diese Karten gefiltert werden.
pub fn get_color_masks_for_game_type(
    game_type: FdoGameType
) -> (u64, u64, u64, u64, u64) {
    return match game_type {
        FdoGameType::Normal | FdoGameType::Wedding | FdoGameType::DiamondsSolo => {
            let trump_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::HeartJack as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::ClubJack as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::ClubQueen as u64
                | FdoCard::HeartTen as u64;

            let diamond_color_mask = 0u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
        FdoGameType::HeartsSolo => {
            let trump_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartAce as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::HeartJack as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::ClubJack as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::ClubQueen as u64
                | FdoCard::HeartTen as u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = 0u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
        FdoGameType::SpadesSolo => {
            let trump_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::HeartJack as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::ClubJack as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::ClubQueen as u64
                | FdoCard::HeartTen as u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = 0u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
        FdoGameType::ClubsSolo => {
            let trump_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::HeartJack as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::ClubJack as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::ClubQueen as u64
                | FdoCard::HeartTen as u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = 0u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
        FdoGameType::TrumplessSolo => {
            let trump_color_mask = 0u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartJack as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartTen as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubJack as u64
                | FdoCard::ClubQueen as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
        FdoGameType::QueensSolo => {
            let trump_color_mask = FdoCard::DiamondQueen as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::ClubQueen as u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondJack as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartJack as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartTen as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubJack as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)

        }
        FdoGameType::JacksSolo => {
            let trump_color_mask = FdoCard::DiamondJack as u64
                | FdoCard::HeartJack as u64
                | FdoCard::SpadeJack as u64
                | FdoCard::ClubJack as u64;

            let diamond_color_mask = FdoCard::DiamondNine as u64
                | FdoCard::DiamondQueen as u64
                | FdoCard::DiamondKing as u64
                | FdoCard::DiamondTen as u64
                | FdoCard::DiamondAce as u64;

            let hearts_color_mask = FdoCard::HeartNine as u64
                | FdoCard::HeartQueen as u64
                | FdoCard::HeartKing as u64
                | FdoCard::HeartTen as u64
                | FdoCard::HeartAce as u64;

            let spades_color_mask = FdoCard::SpadeNine as u64
                | FdoCard::SpadeQueen as u64
                | FdoCard::SpadeKing as u64
                | FdoCard::SpadeTen as u64
                | FdoCard::SpadeAce as u64;

            let clubs_color_mask = FdoCard::ClubNine as u64
                | FdoCard::ClubKing as u64
                | FdoCard::ClubTen as u64
                | FdoCard::ClubAce as u64
                | FdoCard::ClubQueen as u64;

            (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask)
        }
    };





}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;
    use crate::basic::color::FdoColor;
    use crate::card::card_to_color::card_to_color;
    use crate::card::cards::FdoCard;
    use crate::game_type::game_type::FdoGameType;

    #[test]
    fn test_masks() {
        for card in FdoCard::iter() {
            for game_type in FdoGameType::iter() {
                let color_in_game = card_to_color(card, game_type);

                let (trump_color_mask, diamond_color_mask, hearts_color_mask, spades_color_mask, clubs_color_mask) = super::get_color_masks_for_game_type(game_type);

                let is_trump = trump_color_mask & (card as u64) != 0;
                let is_diamond = diamond_color_mask & (card as u64) != 0;
                let is_hearts = hearts_color_mask & (card as u64) != 0;
                let is_spades = spades_color_mask & (card as u64) != 0;
                let is_clubs = clubs_color_mask & (card as u64) != 0;

                match color_in_game {
                    FdoColor::Trump => {
                        assert_eq!(is_trump, true);
                        assert_eq!(is_diamond, false);
                        assert_eq!(is_hearts, false);
                        assert_eq!(is_spades, false);
                        assert_eq!(is_clubs, false);
                    },
                    FdoColor::Diamond => {
                        assert_eq!(is_trump, false);
                        assert_eq!(is_diamond, true);
                        assert_eq!(is_hearts, false);
                        assert_eq!(is_spades, false);
                        assert_eq!(is_clubs, false);

                    },
                    FdoColor::Heart => {
                        assert_eq!(is_trump, false);
                        assert_eq!(is_diamond, false);
                        assert_eq!(is_hearts, true);
                        assert_eq!(is_spades, false);
                        assert_eq!(is_clubs, false);
                    }
                    FdoColor::Spade => {
                        assert_eq!(is_trump, false);
                        assert_eq!(is_diamond, false);
                        assert_eq!(is_hearts, false);
                        assert_eq!(is_spades, true);
                        assert_eq!(is_clubs, false);
                    }
                    FdoColor::Club => {
                        assert_eq!(is_trump, false);
                        assert_eq!(is_diamond, false);
                        assert_eq!(is_hearts, false);
                        assert_eq!(is_spades, false);
                        assert_eq!(is_clubs, true);
                    }
                }

            }
        }


    }

}