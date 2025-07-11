use crate::basic::color::DoColor;
use crate::card::card_to_color::card_to_color_in_normal_game;
use crate::card::cards::DoCard;
use crate::debug_assert_valid_player;
use crate::player::player::{player_wraparound, DoPlayer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DoTrick {
    // Die Karten, die in diesem Stich gespielt wurden,
    // beginnend mit der Karte des Startspieler.
    pub cards: [Option<DoCard>; 4],
    pub start_player: DoPlayer,
}

impl DoTrick {
    pub fn empty(start_player: DoPlayer) -> DoTrick {
        debug_assert_valid_player!(start_player);

        DoTrick {
            cards: [None, None, None, None],
            start_player,
        }
    }

    /// ToDo: Kommentieren und Testen!
    pub fn color(&self) -> Option<DoColor> {
        self.cards[0].map(|card| card_to_color_in_normal_game(card))
    }

    /// Gibt den aktuellen Spieler zurück, der als nächstes eine
    /// Karte spielen muss. Oder None, wenn der Stich vollständig ist.
    pub fn current_player(&self) -> Option<DoPlayer> {
        if self.is_completed() {
            return None;
        }

        for i in 0..4 {
            if self.cards[i].is_none() {
                return Some(player_wraparound(self.start_player + i));
            }
        }

        panic!("should not happen");
    }


    /// Gibt an, ob der Stich vollständig ist, also alle 4 Karten
    /// gespielt wurden.
    pub fn is_completed(&self) -> bool {
        self.cards.iter().all(|card| card.is_some())
    }

    /// Nur für Testzwecke, kann hier ein bereits bestehender
    /// Stich erstellt werden. Im realen Spiel startet ein Stich
    /// immer mit der Methode [empty].
    pub fn existing(
        start_player: DoPlayer,
        cards: Vec<DoCard>
    ) -> DoTrick {
        debug_assert_valid_player!(start_player);
        debug_assert!(cards.len() <= 4, "Ein Stich kann maximal 4 Karten haben.");

        let mut trick = DoTrick::empty(start_player);

        for (index, card) in cards.iter().enumerate() {
            trick.cards[index] = Some(*card);
        }

        trick
    }

}

#[cfg(test)]
mod tests {
    use crate::card::cards::DoCard;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP, DoPlayer};
    use crate::trick::trick::DoTrick;

    #[test]
    fn test_empty() {
        let trick = DoTrick::empty(PLAYER_BOTTOM);

        assert_eq!(trick.start_player, PLAYER_BOTTOM);
        assert_eq!(trick.cards, [None, None, None, None]);
        assert_eq!(trick.is_completed(), false);
        assert_eq!(trick.current_player(), Some(PLAYER_BOTTOM));
    }

    #[test]
    fn test_existing() {
        let cards = vec![
            DoCard::DiamondKing,
            DoCard::DiamondNine,
            DoCard::HeartAce
        ];

        let trick = DoTrick::existing(PLAYER_TOP, cards);

        assert_eq!(trick.start_player, PLAYER_TOP);
        assert_eq!(trick.cards[0], Some(DoCard::DiamondKing));
        assert_eq!(trick.cards[1], Some(DoCard::DiamondNine));
        assert_eq!(trick.cards[2], Some(DoCard::HeartAce));
        assert_eq!(trick.cards[3], None);
        assert_eq!(trick.is_completed(), false);
        assert_eq!(trick.current_player(), Some(PLAYER_LEFT));
    }

    #[test]
    fn test_is_completed() {
        let cards = vec![
            DoCard::DiamondKing,
            DoCard::DiamondNine,
            DoCard::HeartAce,
            DoCard::ClubTen
        ];

        let trick = DoTrick::existing(PLAYER_TOP, cards);

        assert_eq!(trick.is_completed(), true);
    }

    #[test]
    fn test_current_player() {
        let trick = DoTrick::empty(PLAYER_LEFT);
        assert_eq!(trick.current_player(), Some(PLAYER_LEFT));

        let trick = DoTrick::existing(
            PLAYER_LEFT,
            vec![
                DoCard::DiamondKing,
            ],
        );
        assert_eq!(trick.current_player(), Some(PLAYER_TOP));

        let trick = DoTrick::existing(
            PLAYER_LEFT,
            vec![
                DoCard::DiamondKing,
                DoCard::DiamondNine,
            ],
        );
        assert_eq!(trick.current_player(), Some(PLAYER_RIGHT));

        let trick = DoTrick::existing(
            PLAYER_LEFT,
            vec![
                DoCard::DiamondKing,
                DoCard::DiamondNine,
                DoCard::HeartAce,
            ],
        );
        assert_eq!(trick.current_player(), Some(PLAYER_BOTTOM));

        let trick = DoTrick::existing(
            PLAYER_LEFT,
            vec![
                DoCard::DiamondKing,
                DoCard::DiamondNine,
                DoCard::HeartAce,
                DoCard::ClubTen,
            ],
        );
        assert_eq!(trick.current_player(), None);
    }
}

