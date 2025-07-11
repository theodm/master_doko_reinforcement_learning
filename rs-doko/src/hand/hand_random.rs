use rand::prelude::{SmallRng};
use rand::seq::SliceRandom;
use crate::card::cards::DoCard;
use crate::hand::hand::{DoHand, hand_add};

macro_rules! duplicate_array {
    ($($elem:expr),*) => {
        [
            $($elem, $elem),*
        ]
    };
}
const AVAILABLE_CARDS: [DoCard; 48] = duplicate_array![
    DoCard::DiamondNine,
    DoCard::DiamondTen,
    DoCard::DiamondJack,
    DoCard::DiamondQueen,
    DoCard::DiamondKing,
    DoCard::DiamondAce,

    DoCard::HeartNine,
    DoCard::HeartTen,
    DoCard::HeartJack,
    DoCard::HeartQueen,
    DoCard::HeartKing,
    DoCard::HeartAce,

    DoCard::ClubNine,
    DoCard::ClubTen,
    DoCard::ClubJack,
    DoCard::ClubQueen,
    DoCard::ClubKing,
    DoCard::ClubAce,

    DoCard::SpadeNine,
    DoCard::SpadeTen,
    DoCard::SpadeJack,
    DoCard::SpadeQueen,
    DoCard::SpadeKing,
    DoCard::SpadeAce
];

/// Erstellt vier zufällige Hände der Spieler
pub fn distribute_cards(
    mut rng: &mut SmallRng
) -> [DoHand; 4] {
    let mut cards_to_distribute = AVAILABLE_CARDS;

    // Shuffle the available cards
    cards_to_distribute.shuffle(&mut rng);

    let mut hands: [DoHand; 4] = [0; 4];
    for i in 0..4 {
        for j in 0..12 {
            hands[i] = hand_add(hands[i], cards_to_distribute[i * 12 + j]);
        }
    }

    hands
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn test_distribute_cards() {
        let mut rng = SmallRng::seed_from_u64(42);

        let hands = distribute_cards(&mut rng);

        // print hands in binary
        for i in 0..4 {
            println!("{:024b}", hands[i]);
        }

        assert_eq!(hands[0], 0b000000000001000000000010001101001111001000011);
        assert_eq!(hands[1], 0b100000000000000000100001100001110100010100110);
        assert_eq!(hands[2], 0b000000000000000000000111011010000010111001001);
        assert_eq!(hands[3], 0b000000000000000010000100010110111000100111000);
    }
}