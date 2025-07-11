use crate::basic::color::DoColor;
use crate::card::card_to_color::card_to_color_in_normal_game;
use crate::card::card_to_eyes::card_to_eyes;
use crate::card::cards::DoCard;

/// Gibt an, ob die Trumpfkarte [current_card_in_trick] die im Stich
/// später als die Trumpfkarte [previous_card_in_trick] gespielt wurde,
/// die vorherige Trumpfkarte im Normalspiel schlägt.
///
/// Bsp.:
///     previous_card_in_trick = DoCard::DiamondNine
///     current_card_in_trick = DoCard::DiamondAce
///    => true
///
/// Es muss sich bei den Karten um Trumpfkarten handeln. Es ist im Wesentlichen
/// eine Hilfsfunktion für die Funktion [is_greater_in_trick_in_normal_game].
fn is_greater_trump_in_trick_in_normal_game(
    current_card_in_trick: DoCard,
    previous_card_in_trick: DoCard,
) -> bool {
    debug_assert!(card_to_color_in_normal_game(previous_card_in_trick) == DoColor::Trump);
    debug_assert!(card_to_color_in_normal_game(current_card_in_trick) == DoColor::Trump);
    fn trump_to_rank(card: DoCard) -> u8 {
        match card {
            DoCard::DiamondNine => 0,
            DoCard::DiamondKing => 1,
            DoCard::DiamondTen => 2,
            DoCard::DiamondAce => 3,

            DoCard::DiamondJack => 4,
            DoCard::HeartJack => 5,
            DoCard::SpadeJack => 6,
            DoCard::ClubJack => 7,

            DoCard::DiamondQueen => 8,
            DoCard::HeartQueen => 9,
            DoCard::SpadeQueen => 10,
            DoCard::ClubQueen => 11,

            DoCard::HeartTen => 12,
            _ => panic!("This card is not a trump card: {:?}", card),
        }
    }

    let previous_rank = trump_to_rank(previous_card_in_trick);
    let current_rank = trump_to_rank(current_card_in_trick);

    previous_rank < current_rank
}

/// Gibt an, ob die Karte [current_card_in_trick] die im Stich später als die Karte
/// [previous_card_in_trick] gespielt wurde, die vorherige Karte im Normalspiel schlägt.
///
/// Bsp.:
///    previous_card_in_trick = DoCard::ClubNine
///    current_card_in_trick = DoCard::ClubAce
/// => true
pub fn is_greater_in_trick_in_normal_game(
    current_card_in_trick: DoCard,
    previous_card_in_trick: DoCard,
    trick_color: DoColor,
) -> bool {
    let current_card_color = card_to_color_in_normal_game(current_card_in_trick);
    let previous_card_color = card_to_color_in_normal_game(previous_card_in_trick);

    let current_card_is_trump = current_card_color == DoColor::Trump;
    let previous_card_is_trump = previous_card_color == DoColor::Trump;

    // Nachfolgender Trumpf gewinnt immer gegen vorherigen Nicht-Trumpf.
    if current_card_is_trump && !previous_card_is_trump {
        return true;
    }

    // Nachfolgender Nicht-Trumpf verliert immer gegen vorherigen Trumpf.
    if !current_card_is_trump && previous_card_is_trump {
        return false;
    }

    // Nachfolgender Trumpf gewinnt gegen vorherigen Trumpf, wenn er
    // eine höhere Trumpfkarte ist.
    if current_card_is_trump && previous_card_is_trump {
        return is_greater_trump_in_trick_in_normal_game(current_card_in_trick, previous_card_in_trick);
    }

    let current_card_is_trick_color = current_card_color == trick_color;
    let previous_card_is_trick_color = previous_card_color == trick_color;

    // Wenn die aktuelle Karte eine Farbkarte der Stichfarbe ist und die vorherige Karte nicht,
    // dann gewinnt die aktuelle Karte. (Die Methode sollte dann aber eigentlich nie vom zugrundeliegenden
    // Algorithmus aufgerufen werden, da immer nur die bisher höchsten Karten betrachtet werden. Wir
    // decken den Fall aber für Vollständigkeit ab.)
    if current_card_is_trick_color && !previous_card_is_trick_color {
        return true;
    }

    // Wenn beide Karten Farbkarten der Stichfarbe sind, dann gewinnt die Karte mit dem höheren Rang.
    if current_card_is_trick_color && previous_card_is_trick_color {
        // Klingt komisch ist aber so. Der Augenwert einer Karte hat den gleichen
        // Rang wie Farbkarten im Spiel. Also können wir auch die Methode zum
        // Vergleich des Ranges von Karten verwenden.
        return card_to_eyes(current_card_in_trick) > card_to_eyes(previous_card_in_trick);
    }

    // Ansonsten ist die Karte nicht höher.
    false
}



#[cfg(test)]
mod tests {
    use crate::basic::color::DoColor;
    use crate::card::card_in_trick_logic::is_greater_in_trick_in_normal_game;
    use crate::card::cards::DoCard;

    #[test]
    fn test_is_greater_in_trick_in_normal_game() {
        // In einem Farbstich gewinnt die höhere Karte
        assert!(is_greater_in_trick_in_normal_game(DoCard::ClubAce, DoCard::ClubTen, DoColor::Club));
        assert!(is_greater_in_trick_in_normal_game(DoCard::ClubTen, DoCard::ClubKing, DoColor::Club));
        assert!(is_greater_in_trick_in_normal_game(DoCard::ClubKing, DoCard::ClubNine, DoColor::Club));

        // In einem Farbstich ist eine gleiche Karte nicht größer
        assert!(!is_greater_in_trick_in_normal_game(DoCard::ClubAce, DoCard::ClubAce, DoColor::Club));
        assert!(!is_greater_in_trick_in_normal_game(DoCard::SpadeKing, DoCard::SpadeKing, DoColor::Spade));
        assert!(!is_greater_in_trick_in_normal_game(DoCard::HeartNine, DoCard::HeartNine, DoColor::Heart));

        // In einem Farbstich verliert die niedrigere Karte
        assert!(!is_greater_in_trick_in_normal_game(DoCard::ClubTen, DoCard::ClubAce, DoColor::Club));
        assert!(!is_greater_in_trick_in_normal_game(DoCard::SpadeNine, DoCard::SpadeAce, DoColor::Spade));

        // Trumpf sticht einen Farbstich
        assert!(is_greater_in_trick_in_normal_game(DoCard::ClubQueen, DoCard::ClubAce, DoColor::Club));

        // Abwurf ist kleiner
        assert!(!is_greater_in_trick_in_normal_game(DoCard::ClubAce, DoCard::HeartAce, DoColor::Heart));

        // Höherer Trumpf ist höher
        assert!(is_greater_in_trick_in_normal_game(DoCard::ClubQueen, DoCard::ClubJack, DoColor::Trump));
        assert!(is_greater_in_trick_in_normal_game(DoCard::HeartTen, DoCard::ClubQueen, DoColor::Trump));
    }

}