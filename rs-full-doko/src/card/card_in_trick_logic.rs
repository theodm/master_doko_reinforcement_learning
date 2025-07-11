use crate::basic::color::FdoColor;
use crate::card::card_to_color::card_to_color;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;

/// Gibt an, ob die Trumpfkarte [current_card_in_trick] die im Stich
/// später als die Trumpfkarte [previous_card_in_trick] gespielt wurde,
/// die vorherige Trumpfkarte schlägt.
///
/// Bsp.:
///     previous_card_in_trick = DoCard::DiamondNine
///     current_card_in_trick = DoCard::DiamondAce
///    => true
///
/// Es muss sich bei den Karten um Trumpfkarten handeln. Es ist im Wesentlichen
/// eine Hilfsfunktion für die Funktion [is_greater_in_trick_in_normal_game].
fn is_greater_trump_in_trick(
    current_card_in_trick: FdoCard,
    previous_card_in_trick: FdoCard,
    game_type: FdoGameType,
) -> bool {
    debug_assert!(game_type != FdoGameType::TrumplessSolo);
    debug_assert!(card_to_color(previous_card_in_trick, game_type) == FdoColor::Trump);
    debug_assert!(card_to_color(current_card_in_trick, game_type) == FdoColor::Trump);

    /// Gibt den Rang der Karte [card] zurück.
    ///
    /// Gilt für folgende Spieltypen:
    /// - Normalspiel
    /// - Hochzeit
    /// - Karo-Solo
    /// - Herz-Solo
    /// - Pik-Solo
    /// - Kreuz-Solo
    /// - Buben-Solo
    /// - Damen-Solo
    fn trump_to_rank(card: FdoCard) -> u8 {
        match card {
            // Für Normalspiel, Hochzeit und Karo-Solo
            FdoCard::DiamondNine => 0,
            FdoCard::DiamondKing => 1,
            FdoCard::DiamondTen => 2,
            FdoCard::DiamondAce => 3,

            // Für Herz-Solo
            FdoCard::HeartNine => 0,
            FdoCard::HeartKing => 1,
            FdoCard::HeartAce => 2,

            // Für Pik-Solo
            FdoCard::SpadeNine => 0,
            FdoCard::SpadeKing => 1,
            FdoCard::SpadeTen => 2,
            FdoCard::SpadeAce => 3,

            // Für Kreuz-Solo
            FdoCard::ClubNine => 0,
            FdoCard::ClubKing => 1,
            FdoCard::ClubTen => 2,
            FdoCard::ClubAce => 3,

            // Für Normalspiel, Hochzeit und Karo-Solo und Buben-Solo
            FdoCard::DiamondJack => 4,
            FdoCard::HeartJack => 5,
            FdoCard::SpadeJack => 6,
            FdoCard::ClubJack => 7,

            // Für Normalspiel, Hochzeit und Karo-Solo und Damen-Solo
            FdoCard::DiamondQueen => 8,
            FdoCard::HeartQueen => 9,
            FdoCard::SpadeQueen => 10,
            FdoCard::ClubQueen => 11,

            FdoCard::HeartTen => 12
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
pub fn is_greater_in_trick(
    current_card_in_trick: FdoCard,
    previous_card_in_trick: FdoCard,
    trick_color: FdoColor,
    game_type: FdoGameType,
) -> bool {
    let current_card_color = card_to_color(current_card_in_trick, game_type);
    let previous_card_color = card_to_color(previous_card_in_trick, game_type);

    let current_card_is_trump = current_card_color == FdoColor::Trump;
    let previous_card_is_trump = previous_card_color == FdoColor::Trump;

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
        return is_greater_trump_in_trick(current_card_in_trick, previous_card_in_trick, game_type);
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
        return current_card_in_trick.eyes() > previous_card_in_trick.eyes();
    }

    // Ansonsten ist die Karte nicht höher.
    false
}


#[cfg(test)]
mod tests {
    use FdoGameType::{DiamondsSolo, Normal, Wedding};
    use crate::basic::color::FdoColor;
    use crate::card::card_in_trick_logic::is_greater_in_trick;
    use crate::card::cards::FdoCard;
    use crate::game_type::game_type::FdoGameType;

    fn str_to_card(s: &str) -> FdoCard {
        FdoCard::from_str(s)
    }

    fn str_to_color(s: &str) -> FdoColor {
        match s {
            "♦" => FdoColor::Diamond,
            "♥" => FdoColor::Heart,
            "♠" => FdoColor::Spade,
            "♣" => FdoColor::Club,
            "T" => FdoColor::Trump,
            _ => panic!("Unknown color: {}", s),
        }
    }

    fn str_to_colors(s: &str) -> Vec<FdoColor> {
        s.chars().map(|c| str_to_color(&c.to_string())).collect()
    }

    fn str_to_cards(s: &str) -> Vec<FdoCard> {
        FdoCard::vec_from_str(s)
    }
    fn is_smaller(
        current_card_in_trick: &str, // Die erste Karte, z.B. "♦9"
        colors: &str,                // Eine Liste von Farben, z.B. "T♥"
        cards: &str,                 // Eine Liste von Vergleichskarten, z.B. "♦K ♦10"
        game_types: &[FdoGameType],  // Eine Liste von Spieltypen
    ) {
        let current_card_in_trick = str_to_card(current_card_in_trick);
        let colors = str_to_colors(colors);
        let cards = str_to_cards(cards);

        for &game_type in game_types {
            for &color in &colors {
                for &card2 in &cards {
                    assert!(
                        !is_greater_in_trick(current_card_in_trick, card2, color, game_type),
                        "Assertion failed: {:?} should not be greater than {:?} in color {:?} and game type {:?}",
                        current_card_in_trick,
                        card2,
                        color,
                        game_type
                    );
                }
            }
        }
    }

    fn is_greater(
        current_card_in_trick: &str, // Die erste Karte, z.B. "♦9"
        colors: &str,                // Eine Liste von Farben, z.B. "T♥"
        cards: &str,                 // Eine Liste von Vergleichskarten, z.B. "♦K ♦10"
        game_types: &[FdoGameType],  // Eine Liste von Spieltypen
    ) {
        let current_card_in_trick = str_to_card(current_card_in_trick);
        let colors = str_to_colors(colors);
        let cards = str_to_cards(cards);

        for &game_type in game_types {
            for &color in &colors {
                for &card2 in &cards {
                    assert!(
                        is_greater_in_trick(current_card_in_trick, card2, color, game_type),
                        "Assertion failed: {:?} should be greater than {:?} in color {:?} and game type {:?}",
                        current_card_in_trick,
                        card2,
                        color,
                        game_type
                    );
                }
            }
        }
    }

    #[test]
    fn test_is_greater_in_trick() {
        // Normalspiel, Hochzeit und Karo-Solo
        let normal_like_game_types = [
            Normal,
            Wedding,
            DiamondsSolo,
        ];

        is_smaller("♦9", "T♥♠♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦9", "T♥♠♣", "♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♦K", "T♥♠♣", "♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦K", "T♥♠♣", "♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♦10", "T♥♠♣", "♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦10", "T♥♠♣", "♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♦A", "T♥♠♣", "♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦A", "T♥♠♣", "♦10 ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♦J", "T♥♠♣", "♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦J", "T♥♠♣", "♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♥J", "T♥♠♣", "♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♥J", "T♥♠♣", "♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♠J", "T♥♠♣", "♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠J", "T♥♠♣", "♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♣J", "T♥♠♣", "♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♣J", "T♥♠♣", "♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_smaller("♦Q", "T♥♠♣", "♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♦Q", "T♥♠♣", "♣J ♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_smaller("♥Q", "T♥♠♣", "♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♥Q", "T♥♠♣", "♦Q ♣J ♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_smaller("♠Q", "T♥♠♣", "♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠Q", "T♥♠♣", "♥Q ♦Q ♣J ♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_smaller("♣Q", "T♥♠♣", "♣Q ♥10", &normal_like_game_types);
        is_greater("♣Q", "T♥♠♣", "♠Q ♥Q ♦Q ♣J ♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_smaller("♥10", "T♥♠♣", "♥10", &normal_like_game_types);
        is_greater("♥10", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q ♣J ♠J ♥J ♦J ♦A ♦K ♦9 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &normal_like_game_types);

        is_smaller("♥9", "T♠♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♥9", "T♠♣", "", &normal_like_game_types);
        is_smaller("♥9", "♥", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♥9", "♥", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♥9", "♥", "♥9 ♥K ♥A", &normal_like_game_types);
        is_greater("♥9", "♥", "", &normal_like_game_types);

        is_smaller("♥K", "T♠♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♥K", "T♠♣", "", &normal_like_game_types);
        is_smaller("♥K", "♥", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♥K", "♥", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♥K", "♥", "♥K ♥A", &normal_like_game_types);
        is_greater("♥K", "♥", "♥9", &normal_like_game_types);

        is_smaller("♥A", "T♠♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♥A", "T♠♣", "", &normal_like_game_types);
        is_smaller("♥A", "♥", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♥A", "♥", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_smaller("♥A", "♥", "♥A", &normal_like_game_types);
        is_greater("♥A", "♥", "♥9 ♥K", &normal_like_game_types);

        is_smaller("♠9", "T♥♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♠9", "T♥♣", "", &normal_like_game_types);
        is_smaller("♠9", "♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠9", "♠", "♣9 ♣K ♣10 ♣A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♠9", "♠", "♠9 ♠K ♠10 ♠A", &normal_like_game_types);
        is_greater("♠9", "♠", "", &normal_like_game_types);

        is_smaller("♠K", "T♥♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♠K", "T♥♣", "", &normal_like_game_types);
        is_smaller("♠K", "♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠K", "♠", "♣9 ♣K ♣10 ♣A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &normal_like_game_types);
        is_greater("♠K", "♠", "♠9", &normal_like_game_types);

        is_smaller("♠10", "T♥♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♠10", "T♥♣", "", &normal_like_game_types);
        is_smaller("♠10", "♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠10", "♠", "♣9 ♣K ♣10 ♣A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♠10", "♠", "♠10 ♠A", &normal_like_game_types);
        is_greater("♠10", "♠", "♠K ♠9", &normal_like_game_types);

        is_smaller("♠A", "T♥♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♠A", "T♥♣", "", &normal_like_game_types);
        is_smaller("♠A", "♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♠A", "♠", "♣9 ♣K ♣10 ♣A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♠A", "♠", "♠A", &normal_like_game_types);
        is_greater("♠A", "♠", "♠10 ♠K ♠9", &normal_like_game_types);

        is_smaller("♣9", "T♥♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣9", "T♥♠", "", &normal_like_game_types);
        is_smaller("♣9", "♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♣9", "♣", "♠9 ♠K ♠10 ♠A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♣9", "♣", "♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣9", "♣", "", &normal_like_game_types);

        is_smaller("♣K", "T♥♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣K", "T♥♠", "", &normal_like_game_types);
        is_smaller("♣K", "♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♣K", "♣", "♠9 ♠K ♠10 ♠A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣K", "♣", "♣9", &normal_like_game_types);

        is_smaller("♣10", "T♥♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣10", "T♥♠", "", &normal_like_game_types);
        is_smaller("♣10", "♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♣10", "♣", "♠9 ♠K ♠10 ♠A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♣10", "♣", "♣10 ♣A", &normal_like_game_types);
        is_greater("♣10", "♣", "♣9 ♣K", &normal_like_game_types);

        is_smaller("♣A", "T♥♠", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &normal_like_game_types);
        is_greater("♣A", "T♥♠", "", &normal_like_game_types);
        is_smaller("♣A", "♣", "♦9 ♦K ♦10 ♦A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &normal_like_game_types);
        is_greater("♣A", "♣", "♠9 ♠K ♠10 ♠A ♥9 ♥K ♥A", &normal_like_game_types);
        is_smaller("♣A", "♣", "♣A", &normal_like_game_types);
        is_greater("♣A", "♣", "♣9 ♣K ♣10", &normal_like_game_types);

        // Herz-Solo
        let heart_solo = [FdoGameType::HeartsSolo];

        is_smaller("♥9", "T♦♠♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♥9", "T♦♠♣", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♥K", "T♦♠♣", "♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♥K", "T♦♠♣", "♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♥A", "T♦♠♣", "♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♥A", "T♦♠♣", "♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦J", "T♦♠♣", "♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦J", "T♦♠♣", "♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♥J", "T♦♠♣", "♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♥J", "T♦♠♣", "♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠J", "T♦♠♣", "♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠J", "T♦♠♣", "♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♣J", "T♦♠♣", "♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♣J", "T♦♠♣", "♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦Q", "T♦♠♣", "♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦Q", "T♦♠♣", "♣J ♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♥Q", "T♦♠♣", "♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♥Q", "T♦♠♣", "♦Q ♣J ♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠Q", "T♦♠♣", "♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠Q", "T♦♠♣", "♥Q ♦Q ♣J ♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♣Q", "T♦♠♣", "♣Q ♥10", &heart_solo);
        is_greater("♣Q", "T♦♠♣", "♠Q ♥Q ♦Q ♣J ♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♥10", "T♦♠♣", "♥10", &heart_solo);
        is_greater("♥10", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q ♣J ♠J ♥J ♦J ♥A ♥K ♥9 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);

        is_smaller("♦9", "T♠♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♦9", "T♠♣", "", &heart_solo);
        is_smaller("♦9", "♦", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦9", "♦", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦9", "♦", "♦9 ♦K ♦10 ♦A", &heart_solo);
        is_greater("♦9", "♦", "", &heart_solo);

        is_smaller("♦K", "T♠♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♦K", "T♠♣", "", &heart_solo);
        is_smaller("♦K", "♦", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦K", "♦", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &heart_solo);
        is_greater("♦K", "♦", "♦9", &heart_solo);

        is_smaller("♦10", "T♠♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♦10", "T♠♣", "", &heart_solo);
        is_smaller("♦10", "♦", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦10", "♦", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &heart_solo);
        is_greater("♦10", "♦", "♦9 ♦K", &heart_solo);

        is_smaller("♦A", "T♠♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♦A", "T♠♣", "", &heart_solo);
        is_smaller("♦A", "♦", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♦A", "♦", "♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♦A", "♦", "♦A", &heart_solo);
        is_greater("♦A", "♦", "♦9 ♦K ♦10", &heart_solo);

        is_smaller("♠9", "T♦♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♠9", "T♦♣", "", &heart_solo);
        is_smaller("♠9", "♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠9", "♠", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠9", "♠", "♠9 ♠K ♠10 ♠A", &heart_solo);
        is_greater("♠9", "♠", "", &heart_solo);

        is_smaller("♠K", "T♦♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♠K", "T♦♣", "", &heart_solo);
        is_smaller("♠K", "♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠K", "♠", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &heart_solo);
        is_greater("♠K", "♠", "♠9", &heart_solo);

        is_smaller("♠10", "T♦♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♠10", "T♦♣", "", &heart_solo);
        is_smaller("♠10", "♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠10", "♠", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠10", "♠", "♠10 ♠A", &heart_solo);
        is_greater("♠10", "♠", "♠K ♠9", &heart_solo);

        is_smaller("♠A", "T♦♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♠A", "T♦♣", "", &heart_solo);
        is_smaller("♠A", "♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♠A", "♠", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_smaller("♠A", "♠", "♠A", &heart_solo);
        is_greater("♠A", "♠", "♠10 ♠K ♠9", &heart_solo);

        is_smaller("♣9", "T♦♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♣9", "T♦♠", "", &heart_solo);
        is_smaller("♣9", "♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♣9", "♣", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &heart_solo);
        is_smaller("♣9", "♣", "♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♣9", "♣", "", &heart_solo);

        is_smaller("♣K", "T♦♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♣K", "T♦♠", "", &heart_solo);
        is_smaller("♣K", "♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♣K", "♣", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &heart_solo);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &heart_solo);
        is_greater("♣K", "♣", "♣9", &heart_solo);

        is_smaller("♣10", "T♦♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♣10", "T♦♠", "", &heart_solo);
        is_smaller("♣10", "♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♣10", "♣", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &heart_solo);
        is_smaller("♣10", "♣", "♣10 ♣A", &heart_solo);
        is_greater("♣10", "♣", "♣K ♣9", &heart_solo);

        is_smaller("♣A", "T♦♠", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A ♣9 ♣K ♣10 ♣A", &heart_solo);
        is_greater("♣A", "T♦♠", "", &heart_solo);
        is_smaller("♣A", "♣", "♥9 ♥K ♥A ♦J ♥J ♠J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &heart_solo);
        is_greater("♣A", "♣", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &heart_solo);
        is_smaller("♣A", "♣", "♣A", &heart_solo);
        is_greater("♣A", "♣", "♣10 ♣K ♣9", &heart_solo);

        // Pik-Solo
        let spade_solo = [FdoGameType::SpadesSolo];

        is_smaller("♠9", "T♥♦♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♠9", "T♥♦♣", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♠K", "T♥♦♣", "♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♠K", "T♥♦♣", "♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♠10", "T♥♦♣", "♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♠10", "T♥♦♣", "♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♠A", "T♥♦♣", "♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♠A", "T♥♦♣", "♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦J", "T♥♦♣", "♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦J", "T♥♦♣", "♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥J", "T♥♦♣", "♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♥J", "T♥♦♣", "♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♣J", "T♥♦♣", "♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♣J", "T♥♦♣", "♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦Q", "T♥♦♣", "♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦Q", "T♥♦♣", "♣J ♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥Q", "T♥♦♣", "♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♥Q", "T♥♦♣", "♦Q ♣J ♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♠Q", "T♥♦♣", "♠Q ♣Q ♥10", &spade_solo);
        is_greater("♠Q", "T♥♦♣", "♥Q ♦Q ♣J ♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♣Q", "T♥♦♣", "♣Q ♥10", &spade_solo);
        is_greater("♣Q", "T♥♦♣", "♠Q ♥Q ♦Q ♣J ♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥10", "T♥♦♣", "♥10", &spade_solo);
        is_greater("♥10", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q ♣J ♥J ♦J ♠A ♠10 ♠K ♠9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);

        is_smaller("♦9", "T♥♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♦9", "T♥♣", "", &spade_solo);
        is_smaller("♦9", "♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦9", "♦", "♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦9", "♦", "♦9 ♦K ♦10 ♦A", &spade_solo);
        is_greater("♦9", "♦", "", &spade_solo);

        is_smaller("♦K", "T♥♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♦K", "T♥♣", "", &spade_solo);
        is_smaller("♦K", "♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦K", "♦", "♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &spade_solo);
        is_greater("♦K", "♦", "♦9", &spade_solo);

        is_smaller("♦10", "T♥♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♦10", "T♥♣", "", &spade_solo);
        is_smaller("♦10", "♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦10", "♦", "♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &spade_solo);
        is_greater("♦10", "♦", "♦K ♦9", &spade_solo);

        is_smaller("♦A", "T♥♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♦A", "T♥♣", "", &spade_solo);
        is_smaller("♦A", "♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♦A", "♦", "♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♦A", "♦", "♦A", &spade_solo);
        is_greater("♦A", "♦", "♦9 ♦K ♦10", &spade_solo);

        is_smaller("♥9", "T♦♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♥9", "T♦♣", "", &spade_solo);
        is_smaller("♥9", "♥", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♥9", "♥", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥9", "♥", "♥9 ♥K ♥A", &spade_solo);
        is_greater("♥9", "♥", "", &spade_solo);

        is_smaller("♥K", "T♦♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♥K", "T♦♣", "", &spade_solo);
        is_smaller("♥K", "♥", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♥K", "♥", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥K", "♥", "♥K ♥A", &spade_solo);
        is_greater("♥K", "♥", "♥9", &spade_solo);

        is_smaller("♥A", "T♦♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♥A", "T♦♣", "", &spade_solo);
        is_smaller("♥A", "♥", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♥A", "♥", "♦9 ♦K ♦10 ♦A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_smaller("♥A", "♥", "♥A", &spade_solo);
        is_greater("♥A", "♥", "♥K ♥9", &spade_solo);

        is_smaller("♣9", "T♥♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♣9", "T♥♦", "", &spade_solo);
        is_smaller("♣9", "♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♣9", "♣", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &spade_solo);
        is_smaller("♣9", "♣", "♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♣9", "♣", "", &spade_solo);

        is_smaller("♣K", "T♥♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♣K", "T♥♦", "", &spade_solo);
        is_smaller("♣K", "♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♣K", "♣", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &spade_solo);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &spade_solo);
        is_greater("♣K", "♣", "♣9", &spade_solo);

        is_smaller("♣10", "T♥♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♣10", "T♥♦", "", &spade_solo);
        is_smaller("♣10", "♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♣10", "♣", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &spade_solo);
        is_smaller("♣10", "♣", "♣10 ♣A", &spade_solo);
        is_greater("♣10", "♣", "♣K ♣9", &spade_solo);

        is_smaller("♣A", "T♥♦", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♣9 ♣K ♣10 ♣A", &spade_solo);
        is_greater("♣A", "T♥♦", "", &spade_solo);
        is_smaller("♣A", "♣", "♠9 ♠K ♠10 ♠A ♦J ♥J ♣J ♦Q ♥Q ♠Q ♣Q ♥10", &spade_solo);
        is_greater("♣A", "♣", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &spade_solo);
        is_smaller("♣A", "♣", "♣A", &spade_solo);
        is_greater("♣A", "♣", "♣10 ♣K ♣9", &spade_solo);

        // Kreuz-Solo
        let club_solo = [FdoGameType::ClubsSolo];

        is_smaller("♣9", "T♥♦♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♣9", "T♥♦♠", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♣K", "T♥♦♠", "♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♣K", "T♥♦♠", "♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♣10", "T♥♦♠", "♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♣10", "T♥♦♠", "♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♣A", "T♥♦♠", "♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♣A", "T♥♦♠", "♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦J", "T♥♦♠", "♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦J", "T♥♦♠", "♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥J", "T♥♦♠", "♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♥J", "T♥♦♠", "♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♠J", "T♥♦♠", "♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♠J", "T♥♦♠", "♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦Q", "T♥♦♠", "♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦Q", "T♥♦♠", "♠J ♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥Q", "T♥♦♠", "♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♥Q", "T♥♦♠", "♦Q ♠J ♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♠Q", "T♥♦♠", "♠Q ♣Q ♥10", &club_solo);
        is_greater("♠Q", "T♥♦♠", "♥Q ♦Q ♠J ♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♣Q", "T♥♦♠", "♣Q ♥10", &club_solo);
        is_greater("♣Q", "T♥♦♠", "♠Q ♥Q ♦Q ♠J ♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥10", "T♥♦♠", "♥10", &club_solo);
        is_greater("♥10", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q ♠J ♥J ♦J ♣A ♣10 ♣K ♣9 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);

        is_smaller("♦9", "T♥♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♦9", "T♥♠", "", &club_solo);
        is_smaller("♦9", "♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦9", "♦", "♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦9", "♦", "♦9 ♦K ♦10 ♦A", &club_solo);
        is_greater("♦9", "♦", "", &club_solo);

        is_smaller("♦K", "T♥♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♦K", "T♥♠", "", &club_solo);
        is_smaller("♦K", "♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦K", "♦", "♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &club_solo);
        is_greater("♦K", "♦", "♦9", &club_solo);

        is_smaller("♦10", "T♥♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♦10", "T♥♠", "", &club_solo);
        is_smaller("♦10", "♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦10", "♦", "♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &club_solo);
        is_greater("♦10", "♦", "♦K ♦9", &club_solo);

        is_smaller("♦A", "T♥♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♦A", "T♥♠", "", &club_solo);
        is_smaller("♦A", "♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♦A", "♦", "♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♦A", "♦", "♦A", &club_solo);
        is_greater("♦A", "♦", "♦K ♦10 ♦9", &club_solo);

        is_smaller("♥9", "T♦♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♥9", "T♦♠", "", &club_solo);
        is_smaller("♥9", "♥", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♥9", "♥", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥9", "♥", "♥9 ♥K ♥A", &club_solo);
        is_greater("♥9", "♥", "", &club_solo);

        is_smaller("♥K", "T♦♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♥K", "T♦♠", "", &club_solo);
        is_smaller("♥K", "♥", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♥K", "♥", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥K", "♥", "♥K ♥A", &club_solo);
        is_greater("♥K", "♥", "♥9", &club_solo);

        is_smaller("♥A", "T♦♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♥A", "T♦♠", "", &club_solo);
        is_smaller("♥A", "♥", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♥A", "♥", "♦9 ♦K ♦10 ♦A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_smaller("♥A", "♥", "♥A", &club_solo);
        is_greater("♥A", "♥", "♥K ♥9", &club_solo);

        is_smaller("♠9", "T♥♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♠9", "T♥♦", "", &club_solo);
        is_smaller("♠9", "♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♠9", "♠", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &club_solo);
        is_smaller("♠9", "♠", "♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♠9", "♠", "", &club_solo);

        is_smaller("♠K", "T♥♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♠K", "T♥♦", "", &club_solo);
        is_smaller("♠K", "♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♠K", "♠", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &club_solo);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &club_solo);
        is_greater("♠K", "♠", "♠9", &club_solo);

        is_smaller("♠10", "T♥♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♠10", "T♥♦", "", &club_solo);
        is_smaller("♠10", "♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♠10", "♠", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &club_solo);
        is_smaller("♠10", "♠", "♠10 ♠A", &club_solo);
        is_greater("♠10", "♠", "♠K ♠9", &club_solo);

        is_smaller("♠A", "T♥♦", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10 ♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A ♠9 ♠K ♠10 ♠A", &club_solo);
        is_greater("♠A", "T♥♦", "", &club_solo);
        is_smaller("♠A", "♠", "♣9 ♣K ♣10 ♣A ♦J ♥J ♠J ♦Q ♥Q ♠Q ♣Q ♥10", &club_solo);
        is_greater("♠A", "♠", "♦9 ♦K ♦10 ♦A ♥9 ♥K ♥A", &club_solo);
        is_smaller("♠A", "♠", "♠A", &club_solo);
        is_greater("♠A", "♠", "♠10 ♠K ♠9", &club_solo);

        // Buben-Solo
        let jacks_solo = [FdoGameType::JacksSolo];

        is_greater("♦J", "T♥♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦J", "T♥♦♠♣", "♦J ♥J ♠J ♣J", &jacks_solo);
        is_greater("♦J", "T♥♦♠♣", "", &jacks_solo);

        is_greater("♥J", "T♥♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥J", "T♥♦♠♣", "♥J ♠J ♣J", &jacks_solo);
        is_greater("♥J", "T♥♦♠♣", "♦J", &jacks_solo);

        is_greater("♠J", "T♥♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠J", "T♥♦♠♣", "♠J ♣J", &jacks_solo);
        is_greater("♠J", "T♥♦♠♣", "♥J ♦J", &jacks_solo);

        is_greater("♣J", "T♥♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♣J", "T♥♦♠♣", "♣J", &jacks_solo);
        is_greater("♣J", "T♥♦♠♣", "♠J ♥J ♦J", &jacks_solo);

        is_smaller("♦9", "T♥♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♦9", "T♥♠♣", "♥9 ♥K ♥Q ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦9", "♦", "♦9 ♦Q ♦K ♦10 ♦A", &jacks_solo);
        is_greater("♦9", "♦", "", &jacks_solo);

        is_smaller("♦Q", "T♥♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♦Q", "T♥♠♣", "♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦Q", "♦", "♦Q ♦K ♦10 ♦A", &jacks_solo);
        is_greater("♦Q", "♦", "♦9", &jacks_solo);

        is_smaller("♦K", "T♥♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♦K", "T♥♠♣", "♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &jacks_solo);
        is_greater("♦K", "♦", "♦Q ♦9", &jacks_solo);

        is_smaller("♦10", "T♥♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♦10", "T♥♠♣", "♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &jacks_solo);
        is_greater("♦10", "♦", "♦K ♦Q ♦9", &jacks_solo);

        is_smaller("♦A", "T♥♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♦A", "T♥♠♣", "♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♦A", "♦", "♦A", &jacks_solo);
        is_greater("♦A", "♦", "♦10 ♦K ♦9", &jacks_solo);

        is_smaller("♥9", "T♦♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♥9", "T♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥9", "♥", "♥9 ♥Q ♥K ♥10 ♥A", &jacks_solo);
        is_greater("♥9", "♥", "", &jacks_solo);

        is_smaller("♥Q", "T♦♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♥Q", "T♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥Q", "♥", "♥Q ♥K ♥10 ♥A", &jacks_solo);
        is_greater("♥Q", "♥", "♥9", &jacks_solo);

        is_smaller("♥K", "T♦♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♥K", "T♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥K", "♥", "♥K ♥10 ♥A", &jacks_solo);
        is_greater("♥K", "♥", "♥Q ♥9", &jacks_solo);

        is_smaller("♥10", "T♦♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♥10", "T♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥10", "♥", "♥10 ♥A", &jacks_solo);
        is_greater("♥10", "♥", "♥Q ♥K ♥9", &jacks_solo);

        is_smaller("♥A", "T♦♠♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♥A", "T♦♠♣", "♦9 ♦Q ♦K ♦10 ♦A ♠9 ♠Q ♠K ♠10 ♠A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♥A", "♥", "♥A", &jacks_solo);
        is_greater("♥A", "♥", "♥Q ♥K ♥10 ♥9", &jacks_solo);

        is_smaller("♠9", "T♥♦♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♠9", "T♥♦♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠9", "♠", "♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_greater("♠9", "♠", "", &jacks_solo);

        is_smaller("♠Q", "T♥♦♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♠Q", "T♥♦♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠Q", "♠", "♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_greater("♠Q", "♠", "♠9", &jacks_solo);

        is_smaller("♠K", "T♥♦♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♠K", "T♥♦♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &jacks_solo);
        is_greater("♠K", "♠", "♠Q ♠9", &jacks_solo);

        is_smaller("♠10", "T♥♦♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♠10", "T♥♦♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠10", "♠", "♠10 ♠A", &jacks_solo);
        is_greater("♠10", "♠", "♠Q ♠K ♠9", &jacks_solo);

        is_smaller("♠A", "T♥♦♣", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♠A", "T♥♦♣", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_smaller("♠A", "♠", "♠A", &jacks_solo);
        is_greater("♠A", "♠", "♠Q ♠K ♠10 ♠9", &jacks_solo);

        is_smaller("♣9", "T♥♦♠", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♣9", "T♥♦♠", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_smaller("♣9", "♣", "♣9 ♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_greater("♣9", "♣", "", &jacks_solo);

        is_smaller("♣Q", "T♥♦♠", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♣Q", "T♥♦♠", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_smaller("♣Q", "♣", "♣Q ♣K ♣10 ♣A", &jacks_solo);
        is_greater("♣Q", "♣", "♣9", &jacks_solo);

        is_smaller("♣K", "T♥♦♠", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♣K", "T♥♦♠", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &jacks_solo);
        is_greater("♣K", "♣", "♣Q ♣9", &jacks_solo);

        is_smaller("♣10", "T♥♦♠", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♣10", "T♥♦♠", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_smaller("♣10", "♣", "♣10 ♣A", &jacks_solo);
        is_greater("♣10", "♣", "♣Q ♣K ♣9", &jacks_solo);

        is_smaller("♣A", "T♥♦♠", "♣J ♠J ♥J ♦J", &jacks_solo);
        is_smaller("♣A", "T♥♦♠", "♦9 ♦Q ♦K ♦10 ♦A ♥9 ♥Q ♥K ♥10 ♥A ♠9 ♠Q ♠K ♠10 ♠A", &jacks_solo);
        is_smaller("♣A", "♣", "♣A", &jacks_solo);
        is_greater("♣A", "♣", "♣Q ♣K ♣10 ♣9", &jacks_solo);

        let queens_solo = [FdoGameType::QueensSolo];

        is_greater("♦Q", "T♥♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦Q", "T♥♦♠♣", "♦Q ♥Q ♠Q ♣Q", &queens_solo);
        is_greater("♦Q", "T♥♦♠♣", "", &queens_solo);

        is_greater("♥Q", "T♥♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥Q", "T♥♦♠♣", "♥Q ♠Q ♣Q", &queens_solo);
        is_greater("♥Q", "T♥♦♠♣", "♦Q", &queens_solo);

        is_greater("♠Q", "T♥♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠Q", "T♥♦♠♣", "♠Q ♣Q", &queens_solo);
        is_greater("♠Q", "T♥♦♠♣", "♥Q ♦Q", &queens_solo);

        is_greater("♣Q", "T♥♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♣Q", "T♥♦♠♣", "♣Q", &queens_solo);
        is_greater("♣Q", "T♥♦♠♣", "♠Q ♥Q ♦Q", &queens_solo);

        is_smaller("♦9", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♦9", "T♥♠♣", "♥9 ♥K ♥J ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦9", "♦", "♦9 ♦J ♦K ♦10 ♦A", &queens_solo);
        is_greater("♦9", "♦", "", &queens_solo);

        is_smaller("♦J", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♦J", "T♥♠♣", "♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦J", "♦", "♦J ♦K ♦10 ♦A", &queens_solo);
        is_greater("♦J", "♦", "♦9", &queens_solo);

        is_smaller("♦K", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♦K", "T♥♠♣", "♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &queens_solo);
        is_greater("♦K", "♦", "♦J ♦9", &queens_solo);

        is_smaller("♦10", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♦10", "T♥♠♣", "♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &queens_solo);
        is_greater("♦10", "♦", "♦K ♦J ♦9", &queens_solo);

        is_smaller("♦A", "T♥♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♦A", "T♥♠♣", "♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♦A", "♦", "♦A", &queens_solo);
        is_greater("♦A", "♦", "♦10 ♦K ♦9", &queens_solo);

        is_smaller("♥9", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♥9", "T♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥9", "♥", "♥9 ♥J ♥K ♥10 ♥A", &queens_solo);
        is_greater("♥9", "♥", "", &queens_solo);

        is_smaller("♥J", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♥J", "T♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥J", "♥", "♥J ♥K ♥10 ♥A", &queens_solo);
        is_greater("♥J", "♥", "♥9", &queens_solo);

        is_smaller("♥K", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♥K", "T♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥K", "♥", "♥K ♥10 ♥A", &queens_solo);
        is_greater("♥K", "♥", "♥J ♥9", &queens_solo);

        is_smaller("♥10", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♥10", "T♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥10", "♥", "♥10 ♥A", &queens_solo);
        is_greater("♥10", "♥", "♥J ♥K ♥9", &queens_solo);

        is_smaller("♥A", "T♦♠♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♥A", "T♦♠♣", "♦9 ♦J ♦K ♦10 ♦A ♠9 ♠J ♠K ♠10 ♠A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♥A", "♥", "♥A", &queens_solo);
        is_greater("♥A", "♥", "♥J ♥K ♥10 ♥9", &queens_solo);

        is_smaller("♠9", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♠9", "T♥♦♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠9", "♠", "♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_greater("♠9", "♠", "", &queens_solo);

        is_smaller("♠J", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♠J", "T♥♦♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠J", "♠", "♠J ♠K ♠10 ♠A", &queens_solo);
        is_greater("♠J", "♠", "♠9", &queens_solo);

        is_smaller("♠K", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♠K", "T♥♦♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &queens_solo);
        is_greater("♠K", "♠", "♠J ♠9", &queens_solo);

        is_smaller("♠10", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♠10", "T♥♦♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠10", "♠", "♠10 ♠A", &queens_solo);
        is_greater("♠10", "♠", "♠J ♠K ♠9", &queens_solo);

        is_smaller("♠A", "T♥♦♣", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♠A", "T♥♦♣", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_smaller("♠A", "♠", "♠A", &queens_solo);
        is_greater("♠A", "♠", "♠J ♠K ♠10 ♠9", &queens_solo);

        is_smaller("♣9", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♣9", "T♥♦♠", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_smaller("♣9", "♣", "♣9 ♣J ♣K ♣10 ♣A", &queens_solo);
        is_greater("♣9", "♣", "", &queens_solo);

        is_smaller("♣J", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♣J", "T♥♦♠", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_smaller("♣J", "♣", "♣J ♣K ♣10 ♣A", &queens_solo);
        is_greater("♣J", "♣", "♣9", &queens_solo);

        is_smaller("♣K", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♣K", "T♥♦♠", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &queens_solo);
        is_greater("♣K", "♣", "♣J ♣9", &queens_solo);

        is_smaller("♣10", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♣10", "T♥♦♠", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_smaller("♣10", "♣", "♣10 ♣A", &queens_solo);
        is_greater("♣10", "♣", "♣J ♣K ♣9", &queens_solo);

        is_smaller("♣A", "T♥♦♠", "♣Q ♠Q ♥Q ♦Q", &queens_solo);
        is_smaller("♣A", "T♥♦♠", "♦9 ♦J ♦K ♦10 ♦A ♥9 ♥J ♥K ♥10 ♥A ♠9 ♠J ♠K ♠10 ♠A", &queens_solo);
        is_smaller("♣A", "♣", "♣A", &queens_solo);
        is_greater("♣A", "♣", "♣J ♣K ♣10 ♣9", &queens_solo);

        // Fleischloser
        let trumpless_solo = [FdoGameType::TrumplessSolo];

        is_smaller("♦9", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦9", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦9", "♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A", &trumpless_solo);
        is_greater("♦9", "♦", "", &trumpless_solo);

        is_smaller("♦J", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦J", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦J", "♦", "♦J ♦Q ♦K ♦10 ♦A", &trumpless_solo);
        is_greater("♦J", "♦", "♦9", &trumpless_solo);

        is_smaller("♦Q", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦Q", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦Q", "♦", "♦Q ♦K ♦10 ♦A", &trumpless_solo);
        is_greater("♦Q", "♦", "♦9 ♦J", &trumpless_solo);

        is_smaller("♦K", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦K", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦K", "♦", "♦K ♦10 ♦A", &trumpless_solo);
        is_greater("♦K", "♦", "♦9 ♦J ♦Q", &trumpless_solo);

        is_smaller("♦10", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦10", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦10", "♦", "♦10 ♦A", &trumpless_solo);
        is_greater("♦10", "♦", "♦9 ♦J ♦Q ♦K", &trumpless_solo);

        is_smaller("♦A", "♥♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♦K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♦A", "♥♠♣", "", &trumpless_solo);
        is_smaller("♦A", "♦", "♦A", &trumpless_solo);
        is_greater("♦A", "♦", "♦9 ♦J ♦Q ♦K ♦10", &trumpless_solo);

        // Herz
        is_smaller("♥9", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥9", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥9", "♥", "♥9 ♥J ♥Q ♥K ♥10 ♥A", &trumpless_solo);
        is_greater("♥9", "♥", "", &trumpless_solo);

        is_smaller("♥J", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥J", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥J", "♥", "♥J ♥Q ♥K ♥10 ♥A", &trumpless_solo);
        is_greater("♥J", "♥", "♥9", &trumpless_solo);

        is_smaller("♥Q", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥Q", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥Q", "♥", "♥Q ♥K ♥10 ♥A", &trumpless_solo);
        is_greater("♥Q", "♥", "♥9 ♥J", &trumpless_solo);

        is_smaller("♥K", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥K", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥K", "♥", "♥K ♥10 ♥A", &trumpless_solo);
        is_greater("♥K", "♥", "♥9 ♥J ♥Q", &trumpless_solo);

        is_smaller("♥10", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥10", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥10", "♥", "♥10 ♥A", &trumpless_solo);
        is_greater("♥10", "♥", "♥9 ♥J ♥Q ♥K", &trumpless_solo);

        is_smaller("♥A", "♦♠♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♥A", "♦♠♣", "", &trumpless_solo);
        is_smaller("♥A", "♥", "♥A", &trumpless_solo);
        is_greater("♥A", "♥", "♥9 ♥J ♥Q ♥K ♥10", &trumpless_solo);

        // Pik
        is_smaller("♠9", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠9", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠9", "♠", "♠9 ♠J ♠Q ♠K ♠10 ♠A", &trumpless_solo);
        is_greater("♠9", "♠", "", &trumpless_solo);

        is_smaller("♠J", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠J", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠J", "♠", "♠J ♠Q ♠K ♠10 ♠A", &trumpless_solo);
        is_greater("♠J", "♠", "♠9", &trumpless_solo);

        is_smaller("♠Q", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠Q", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠Q", "♠", "♠Q ♠K ♠10 ♠A", &trumpless_solo);
        is_greater("♠Q", "♠", "♠9 ♠J", &trumpless_solo);

        is_smaller("♠K", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠K", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠K", "♠", "♠K ♠10 ♠A", &trumpless_solo);
        is_greater("♠K", "♠", "♠9 ♠J ♠Q", &trumpless_solo);

        is_smaller("♠10", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠10", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠10", "♠", "♠10 ♠A", &trumpless_solo);
        is_greater("♠10", "♠", "♠9 ♠J ♠Q ♠K", &trumpless_solo);

        is_smaller("♠A", "♥♦♣", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♠A", "♥♦♣", "", &trumpless_solo);
        is_smaller("♠A", "♠", "♠A", &trumpless_solo);
        is_greater("♠A", "♠", "♠9 ♠J ♠Q ♠K ♠10", &trumpless_solo);

        // Kreuz
        is_smaller("♣9", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣9", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣9", "♣", "♣9 ♣J ♣Q ♣K ♣10 ♣A", &trumpless_solo);
        is_greater("♣9", "♣", "", &trumpless_solo);

        is_smaller("♣J", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣J", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣J", "♣", "♣J ♣Q ♣K ♣10 ♣A", &trumpless_solo);
        is_greater("♣J", "♣", "♣9", &trumpless_solo);

        is_smaller("♣Q", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣Q", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣Q", "♣", "♣Q ♣K ♣10 ♣A", &trumpless_solo);
        is_greater("♣Q", "♣", "♣9 ♣J", &trumpless_solo);

        is_smaller("♣K", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣K", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣K", "♣", "♣K ♣10 ♣A", &trumpless_solo);
        is_greater("♣K", "♣", "♣9 ♣J ♣Q", &trumpless_solo);

        is_smaller("♣10", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣10", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣10", "♣", "♣10 ♣A", &trumpless_solo);
        is_greater("♣10", "♣", "♣9 ♣J ♣Q ♣K", &trumpless_solo);

        is_smaller("♣A", "♥♠♦", "♦9 ♦J ♦Q ♦K ♦10 ♦A ♥9 ♥J ♥Q ♥K ♥10 ♥A ♠9 ♠J ♠Q ♠K ♦10 ♠A ♣9 ♣J ♣Q ♣K ♦10 ♣A", &trumpless_solo);
        is_greater("♣A", "♥♠♦", "", &trumpless_solo);
        is_smaller("♣A", "♣", "♣A", &trumpless_solo);
        is_greater("♣A", "♣", "♣9 ♣J ♣Q ♣K ♣10", &trumpless_solo);

    }
}