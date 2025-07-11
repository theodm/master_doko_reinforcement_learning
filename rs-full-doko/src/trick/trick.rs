use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use crate::basic::color::FdoColor;
use crate::card::card_to_color::card_to_color;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use crate::player::player::FdoPlayer;
use crate::util::po_vec::PlayerOrientedVec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoTrick {
    // Die Karten, die in diesem Stich gespielt wurden
    pub cards: PlayerOrientedVec<FdoCard>,

    pub winning_player: Option<FdoPlayer>,
    pub winning_card: Option<FdoCard>
}

impl FdoTrick {
    pub fn empty(start_player: FdoPlayer) -> FdoTrick {
        FdoTrick {
            cards: PlayerOrientedVec::empty(start_player),
            winning_player: None,
            winning_card: None,
        }
    }

    /// Gibt die Farbe des Stichs zurück. Dies ist die Farbe der
    /// ersten Karte, die gespielt wurde. Ist der Stich noch leer,
    /// wird None zurückgegeben.
    pub fn color(&self, game_type: FdoGameType) -> Option<FdoColor> {
        if self.cards.len() == 0 {
            return None;
        }

        Some(card_to_color(self.cards[self.starting_player()], game_type))
    }

    /// Gibt den aktuellen Spieler zurück, der als nächstes eine
    /// Karte spielen muss. Oder None, wenn der Stich vollständig ist.
    pub fn current_player(&self) -> Option<FdoPlayer> {
        return self.cards.next_empty_player();
    }


    /// Gibt an, ob der Stich vollständig ist, also alle 4 Karten
    /// gespielt wurden.
    pub fn is_completed(&self) -> bool {
        self.cards.is_full()
    }

    /// Nur für Testzwecke, kann hier ein bereits bestehender
    /// Stich erstellt werden. Im realen Spiel startet ein Stich
    /// immer mit der Methode [empty].
    pub fn existing(
        start_player: FdoPlayer,
        cards: Vec<FdoCard>,
    ) -> FdoTrick {
        debug_assert!(cards.len() <= 4, "Ein Stich kann maximal 4 Karten haben.");

        let mut trick = FdoTrick::empty(start_player);

        for card in cards {
            trick.play_card(card, FdoGameType::Normal);
        }

        trick
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn eyes(&self) -> u32 {
        debug_assert!(self.is_completed());

        self
            .cards
            .iter()
            .map(|it| it.eyes())
            .sum()
    }

    pub fn play_card(
        &mut self,
        played_card: FdoCard,

        game_type: FdoGameType
    ) {
        debug_assert!(!self.is_completed());

        self.cards.push(played_card);

        if self.is_completed() {
            let (
                winning_player,
                winning_card
            ) = self
                .calc_winning_player_in_trick(game_type);

            self.winning_player = Some(winning_player);
            self.winning_card = Some(winning_card);
        }
    }

    pub fn starting_player(
        &self
    ) -> FdoPlayer {
        self.cards.starting_player
    }

    pub fn iter_with_player(
        &self
    ) -> impl Iterator<Item = (FdoPlayer, &FdoCard)> {
        self.cards.iter_with_player()
    }

}

#[cfg(test)]
mod tests {
    use crate::basic::color::FdoColor;
    use crate::card::cards::FdoCard;
    use crate::game_type::game_type::FdoGameType;
    use crate::player::player::FdoPlayer;
    use crate::trick::trick::FdoTrick;
    use crate::util::po_vec::PlayerOrientedVec;

    #[test]
    fn test_empty() {
        let trick = FdoTrick::empty(FdoPlayer::BOTTOM);

        assert_eq!(trick.starting_player(), FdoPlayer::BOTTOM);
        assert_eq!(trick.cards, PlayerOrientedVec::empty(FdoPlayer::BOTTOM));
        assert_eq!(trick.is_completed(), false);
        assert_eq!(trick.current_player(), Some(FdoPlayer::BOTTOM));
    }

    #[test]
    fn test_existing() {
        let cards = vec![
            FdoCard::DiamondKing,
            FdoCard::DiamondNine,
            FdoCard::HeartAce,
        ];

        let trick = FdoTrick::existing(FdoPlayer::TOP, cards);

        assert_eq!(trick.starting_player(), FdoPlayer::TOP);
        assert_eq!(trick.cards.storage[0], FdoCard::DiamondKing);
        assert_eq!(trick.cards.storage[1], FdoCard::DiamondNine);
        assert_eq!(trick.cards.storage[2], FdoCard::HeartAce);
        assert_eq!(trick.cards.storage.len(), 3);
        assert_eq!(trick.is_completed(), false);
        assert_eq!(trick.current_player(), Some(FdoPlayer::LEFT));
    }

    #[test]
    fn test_is_completed() {
        let cards = vec![
            FdoCard::DiamondKing,
            FdoCard::DiamondNine,
            FdoCard::HeartAce,
            FdoCard::ClubTen,
        ];

        let trick = FdoTrick::existing(FdoPlayer::TOP, cards);

        assert_eq!(trick.is_completed(), true);
    }

    #[test]
    fn test_current_player() {
        let trick = FdoTrick::empty(FdoPlayer::LEFT);
        assert_eq!(trick.current_player(), Some(FdoPlayer::LEFT));

        let trick = FdoTrick::existing(
            FdoPlayer::LEFT,
            vec![
                FdoCard::DiamondKing,
            ],
        );
        assert_eq!(trick.current_player(), Some(FdoPlayer::TOP));

        let trick = FdoTrick::existing(
            FdoPlayer::LEFT,
            vec![
                FdoCard::DiamondKing,
                FdoCard::DiamondNine,
            ],
        );
        assert_eq!(trick.current_player(), Some(FdoPlayer::RIGHT));

        let trick = FdoTrick::existing(
            FdoPlayer::LEFT,
            vec![
                FdoCard::DiamondKing,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
            ],
        );
        assert_eq!(trick.current_player(), Some(FdoPlayer::BOTTOM));

        let trick = FdoTrick::existing(
            FdoPlayer::LEFT,
            vec![
                FdoCard::DiamondKing,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
                FdoCard::ClubTen,
            ],
        );
        assert_eq!(trick.current_player(), None);
    }

    #[test]
    fn test_trick_color() {
        // Stich ist leer
        let trick = FdoTrick::empty(FdoPlayer::BOTTOM);

        assert_eq!(trick.color(FdoGameType::Normal), None);

        // Normalspiel: Stich hat eine Farbe
        let trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartAce,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
            ],
        );

        assert_eq!(trick.color(FdoGameType::Normal), Some(FdoColor::Heart));

        // Normalspiel: Stich hat Trumpf
        let trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartTen,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
            ],
        );

        assert_eq!(trick.color(FdoGameType::Normal), Some(FdoColor::Trump));

        // Herz-Solo: Stich ist Trumpf
        let trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartAce,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
            ],
        );

        assert_eq!(trick.color(FdoGameType::HeartsSolo), Some(FdoColor::Trump));
    }

    #[test]
    fn test_eyes() {
        let trick = FdoTrick::existing(
            FdoPlayer::BOTTOM,
            vec![
                FdoCard::DiamondKing,
                FdoCard::DiamondNine,
                FdoCard::HeartAce,
                FdoCard::ClubTen,
            ],
        );

        assert_eq!(trick.eyes(), 25);

        let trick = FdoTrick::existing(
            FdoPlayer::BOTTOM,
            vec![
                FdoCard::DiamondJack,
                FdoCard::DiamondNine,
                FdoCard::DiamondQueen,
                FdoCard::ClubTen,
            ],
        );
        assert_eq!(trick.eyes(), 15);
    }

    #[test]
    fn test_play_card() {
        let mut trick = FdoTrick::empty(FdoPlayer::TOP);
        trick.play_card(FdoCard::ClubTen, FdoGameType::Normal);

        assert_eq!(trick.cards[FdoPlayer::TOP], FdoCard::ClubTen);
        assert_eq!(trick.cards.storage.len(), 1);

        let mut trick = FdoTrick::empty(FdoPlayer::TOP);

        trick.play_card(FdoCard::ClubTen, FdoGameType::Normal);
        trick.play_card(FdoCard::ClubTen, FdoGameType::Normal);
        trick.play_card(FdoCard::ClubAce, FdoGameType::Normal);
        trick.play_card(FdoCard::ClubAce, FdoGameType::Normal);

        assert_eq!(trick.cards[FdoPlayer::TOP], FdoCard::ClubTen);
        assert_eq!(trick.cards[FdoPlayer::RIGHT], FdoCard::ClubTen);
        assert_eq!(trick.cards[FdoPlayer::BOTTOM], FdoCard::ClubAce);
        assert_eq!(trick.cards[FdoPlayer::LEFT], FdoCard::ClubAce);

        assert_eq!(trick.winning_player, Some(FdoPlayer::BOTTOM));
        assert_eq!(trick.winning_card, Some(FdoCard::ClubAce));

        assert!(trick.is_completed());
    }



}

