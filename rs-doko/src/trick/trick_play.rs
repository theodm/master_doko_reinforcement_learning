use crate::card::cards::DoCard;
use crate::reservation::reservation::DoReservation;
use crate::reservation::reservation_round::DoReservationRound;
use crate::trick::trick::DoTrick;

pub fn play_card(
    trick: &mut DoTrick,
    played_card: DoCard
) {
    debug_assert!(!trick.is_completed());

    for cardIndexInTrick in 0..4 {
        let card = trick.cards[cardIndexInTrick];

        match card {
            None => {
                trick.cards[cardIndexInTrick] = Some(played_card);
                return;
            }
            Some(_) => {

            }
        }
    }

    panic!("should not happen")
}

#[cfg(test)]
mod tests {
    use crate::player::player::PLAYER_TOP;
    use super::*;

    #[test]
    fn test_play_card() {
        let mut trick = DoTrick::empty(PLAYER_TOP);
        play_card(&mut trick, DoCard::ClubTen);

        assert_eq!(trick.cards[0], Some(DoCard::ClubTen));
    }

    #[test]
    fn test_play_card_full() {
        let mut trick = DoTrick::empty(PLAYER_TOP);

        play_card(&mut trick, DoCard::ClubTen);
        play_card(&mut trick, DoCard::ClubTen);
        play_card(&mut trick, DoCard::ClubAce);
        play_card(&mut trick, DoCard::ClubAce);

        assert_eq!(trick.cards[0], Some(DoCard::ClubTen));
        assert_eq!(trick.cards[1], Some(DoCard::ClubTen));
        assert_eq!(trick.cards[2], Some(DoCard::ClubAce));
        assert_eq!(trick.cards[3], Some(DoCard::ClubAce));
        assert!(trick.is_completed());
    }
}