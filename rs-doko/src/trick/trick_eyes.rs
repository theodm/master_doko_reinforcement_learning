use crate::card::card_to_eyes::card_to_eyes;
use crate::trick::trick::DoTrick;

pub fn trick_eyes(
    trick: &DoTrick
) -> u32 {
    debug_assert!(trick.is_completed());

    let mut eyes = 0;

    for card in &trick.cards {
        match card {
            Some(card) => {
                eyes += card_to_eyes(*card);
            }
            None => {}
        }
    }

    eyes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::cards::DoCard;
    use crate::player::player::PLAYER_BOTTOM;

    #[test]
    fn test_trick_eyes() {
        let trick = DoTrick::existing(
            PLAYER_BOTTOM,
            vec![
                DoCard::DiamondKing,
                DoCard::DiamondNine,
                DoCard::HeartAce,
                DoCard::ClubTen
            ],
        );

        assert_eq!(trick_eyes(&trick), 25);

        let trick = DoTrick::existing(
            PLAYER_BOTTOM,
            vec![
                DoCard::DiamondJack,
                DoCard::DiamondNine,
                DoCard::DiamondQueen,
                DoCard::ClubTen
            ],
        );
        assert_eq!(trick_eyes(&trick), 15);
    }

}