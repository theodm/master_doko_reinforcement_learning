use crate::basic::color::DoColor;
use crate::card::card_to_color::card_to_color_in_normal_game;
use crate::trick::trick::DoTrick;



/// Gibt die Stichfarbe zurück. Ist der Stich noch leer, wird [None] zurückgegeben.
///
/// Bsp.:
///    cards = [None, None, None, None] => None
///    cards = [Some(DoCard::HeartAce), None, None, None] => Some(DoColor::Heart)
pub fn trick_color_in_normal_game(
    trick: &DoTrick,
) -> Option<DoColor> {
    let trick_is_empty = trick.cards.iter().all(|card| card.is_none());

    if trick_is_empty {
        return None;
    }

    let first_card = trick.cards[0].unwrap();
    let color = card_to_color_in_normal_game(first_card);

    Some(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::cards::DoCard;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_TOP};

    #[test]
    fn test_trick_color_in_normal_game_not_started() {
        let trick = DoTrick::empty(PLAYER_BOTTOM);

        assert_eq!(trick_color_in_normal_game(&trick), None);
    }

    #[test]
    fn test_trick_color_in_normal_game_started_color() {
        let cards = vec![
            DoCard::HeartAce,
            DoCard::DiamondNine,
            DoCard::HeartAce
        ];

        let trick = DoTrick::existing(PLAYER_TOP, cards);

        assert_eq!(trick_color_in_normal_game(&trick), Some(DoColor::Heart));
    }

    #[test]
    fn test_trick_color_in_normal_game_started_trump() {
        let cards = vec![
            DoCard::HeartTen,
            DoCard::DiamondNine,
            DoCard::HeartAce
        ];

        let trick = DoTrick::existing(PLAYER_TOP, cards);

        assert_eq!(trick_color_in_normal_game(&trick), Some(DoColor::Trump));
    }

}