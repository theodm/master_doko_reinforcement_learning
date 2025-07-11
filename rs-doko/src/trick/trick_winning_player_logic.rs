use crate::card::card_in_trick_logic::is_greater_in_trick_in_normal_game;
use crate::player::player::{player_wraparound, DoPlayer};
use crate::trick::trick::DoTrick;
use crate::trick::trick_color_logic::trick_color_in_normal_game;


/// Gibt den Gewinner eines vollständigen Stichs zurück.
///
/// Bsp.:
///    DoTrick::existing(PLAYER_TOP, vec![DoCard::ClubTen, DoCard::SpadeAce, DoCard::ClubQueen, DoCard::HeartTen]) => PLAYER_LEFT
pub fn winning_player_in_trick_in_normal_game(
    trick: &DoTrick,
) -> DoPlayer {
    debug_assert!(trick.is_completed());

    let trick_color = trick_color_in_normal_game(trick)
        .unwrap();

    // Nun wird reihum überprüft, ob eine Karte höher ist als die andere,
    // die höchste Karte gewinnt.
    let mut winning_card = trick.cards[0].unwrap();
    let mut winning_player_relative_to_trick_begin: usize = 0;

    for i in 0..4 {
        let card = trick.cards[i].unwrap();

        if is_greater_in_trick_in_normal_game(card, winning_card, trick_color) {
            winning_card = card;
            winning_player_relative_to_trick_begin = i;
        }
    }

    return player_wraparound(winning_player_relative_to_trick_begin + trick.start_player);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::cards::DoCard;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};

    #[test]
    fn test_winning_player_in_trick_in_normal_game() {
        // 3 Beispielstiche in einem Normalspiel
        let trick = DoTrick::existing(
            PLAYER_TOP,
            vec![
                DoCard::ClubTen,
                DoCard::SpadeAce,
                DoCard::ClubQueen,
                DoCard::HeartTen,
            ],
        );

        assert_eq!(winning_player_in_trick_in_normal_game(&trick), PLAYER_LEFT);

        let trick = DoTrick::existing(
            PLAYER_LEFT,
            vec![
                DoCard::ClubTen,
                DoCard::ClubTen,
                DoCard::ClubAce,
                DoCard::ClubAce,
            ],
        );

        assert_eq!(winning_player_in_trick_in_normal_game(&trick), PLAYER_RIGHT);

        let trick = DoTrick::existing(
            PLAYER_BOTTOM,
            vec![
                DoCard::HeartKing,
                DoCard::HeartKing,
                DoCard::ClubTen,
                DoCard::ClubTen,
            ],
        );

        assert_eq!(winning_player_in_trick_in_normal_game(&trick), PLAYER_BOTTOM);
    }


}

