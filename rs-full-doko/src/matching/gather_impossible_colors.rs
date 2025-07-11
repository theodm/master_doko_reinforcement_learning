use enumset::EnumSet;
use crate::basic::color::FdoColor;
use crate::card::card_to_color::card_to_color;
use crate::game_type::game_type::FdoGameType;
use crate::trick::trick::FdoTrick;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

/// Berechnet für alle vorangegenagenen Stiche die Farben,
/// welche die Spieler nicht mehr haben könnten. (Sie können keine Farbe mehr haben,
/// wenn sie diese Farbe nicht bedient haben.)
pub fn gather_impossible_colors(
    previous_tricks: &heapless::Vec<FdoTrick, 12>,

    game_type: Option<FdoGameType>,
) -> PlayerZeroOrientedArr<EnumSet<FdoColor>> {
    let mut impossible_colors = PlayerZeroOrientedArr::from_full([EnumSet::empty(); 4]);

    let game_type = match game_type {
        None => {
            // Das Spiel ist noch nicht begonnen, also haben wir noch keine Kenntnisse.
            return impossible_colors;
        }
        Some(game_type) => game_type,
    };

    // Wenn sich dieses Wissen als falsch erweist, dann können wir die Farben
    // der Spieler einschränken.
    for trick in previous_tricks.iter() {
        let trick_color = trick.color(game_type);

        let trick_color = match trick_color {
            None => continue,
            Some(trick_color) => trick_color,
        };

        for (player, card) in trick.iter_with_player() {
            if card_to_color(*card, game_type) != trick_color {
                impossible_colors[player].insert(trick_color);
            }
        }
    }

    return impossible_colors;
}