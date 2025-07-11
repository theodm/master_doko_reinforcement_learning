use crate::card::card_in_trick_logic::is_greater_in_trick;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use crate::player::player::FdoPlayer;
use crate::trick::trick::FdoTrick;

impl FdoTrick {


    /// Gibt den Gewinner eines vollständigen Stichs zurück. Berechnet ihn immer neu. Stattdessen
    /// besser die Property `winning_player` und `winning_card` verwenden.
    ///
    /// Bsp.:
    ///    DoTrick::existing(PLAYER_TOP, vec![DoCard::ClubTen, DoCard::SpadeAce, DoCard::ClubQueen, DoCard::HeartTen]) => PLAYER_LEFT
    pub fn calc_winning_player_in_trick(
        self: &FdoTrick,

        game_type: FdoGameType,
    ) -> (FdoPlayer, FdoCard) {
        debug_assert!(self.is_completed());

        let trick_color = self
            .color(game_type)
            .unwrap();


        let mut winning_card = None;
        let mut winning_player = None;

        for (player, card) in self.cards.iter_with_player() {
            if winning_card.is_none() {
                winning_card = Some(card);
                winning_player = Some(player);
                continue;
            }

            if is_greater_in_trick(*card, *winning_card.unwrap(), trick_color, game_type) {
                winning_card = Some(card);
                winning_player = Some(player);
            }
        }

        (winning_player.unwrap(), *winning_card.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::cards::FdoCard;

    #[test]
    fn test_winning_player_in_trick_in_normal_game() {
        // 3 Beispielstiche in einem Normalspiel
        let trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::ClubTen,
                FdoCard::SpadeAce,
                FdoCard::ClubQueen,
                FdoCard::HeartTen,
            ],
        );

        assert_eq!(trick.calc_winning_player_in_trick(FdoGameType::Normal), (FdoPlayer::LEFT, FdoCard::HeartTen));

        let trick = FdoTrick::existing(
            FdoPlayer::LEFT,
            vec![
                FdoCard::ClubTen,
                FdoCard::ClubTen,
                FdoCard::ClubAce,
                FdoCard::ClubAce,
            ],
        );

        assert_eq!(trick.calc_winning_player_in_trick(FdoGameType::Normal), (FdoPlayer::RIGHT, FdoCard::ClubAce));

        let trick = FdoTrick::existing(
            FdoPlayer::BOTTOM,
            vec![
                FdoCard::HeartKing,
                FdoCard::HeartKing,
                FdoCard::ClubTen,
                FdoCard::ClubTen,
            ],
        );

        assert_eq!(trick.calc_winning_player_in_trick(FdoGameType::Normal), (FdoPlayer::BOTTOM, FdoCard::HeartKing));

        // Herz-Solo
        let trick = FdoTrick::existing(
            FdoPlayer::BOTTOM,
            vec![
                FdoCard::ClubNine,
                FdoCard::HeartKing,
                FdoCard::ClubTen,
                FdoCard::ClubTen,
            ],
        );

        assert_eq!(trick.calc_winning_player_in_trick(FdoGameType::HeartsSolo), (FdoPlayer::LEFT, FdoCard::HeartKing));

    }


}

