use crate::card::cards::DoCard;
use crate::util::bitflag::bitflag::{bitflag_add, bitflag_contains, bitflag_remove};
use std::num;
use strum::EnumCount;
use crate::action::action::DoAction;
use crate::action::allowed_actions::DoAllowedActions;
use crate::card::cards::DoCard::ClubQueen;

/// Die Hand eines Spielers.
///
/// Wird als Bitflag gespeichert, welches die Werte von DoCard zweimal für beide
/// Vorkommen enthält. Einmal an der Position 0 und einmal an der Position 24 (SECOND_CARD_SHIFT_OFFSET).
pub type DoHand = u64;

const SECOND_CARD_SHIFT_OFFSET: u64 = 24;


/// Gibt zurück, ob die Hand [hand] die Karte [card] mindestens einmal enthält.
pub fn hand_contains(hand: DoHand, card: DoCard) -> bool {
    bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64) || bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET)
}

/// Gibt zurück, ob die Hand [hand] beide Exemplare der Karte [card] enthält.
pub fn hand_contains_both(hand: DoHand, card: DoCard) -> bool {
    bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64) && bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET)
}

/// Fügt die Karte [card] zur Hand [hand] hinzu.
pub fn hand_add(hand: DoHand, card: DoCard) -> DoHand {
    if hand_contains(hand, card) {
        bitflag_add::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET)
    } else {
        bitflag_add::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64)
    }
}

/// Entfernt eine Instanz der Karte [card] von der Hand [hand].
pub fn hand_remove(hand: DoHand, card: DoCard) -> DoHand {
    if bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64) {
        bitflag_remove::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64)
    } else if bitflag_contains::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET) {
        bitflag_remove::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET)
    } else {
        panic!("Karte {:?} nicht in Hand", card);
    }
}

// player_marriage: None
// tricks: [Some(DoTrick { cards: [
// Some(HeartNine), Some(HeartTen), Some(HeartNine), Some(HeartKing)
// ], start_player: 2 }), Some(DoTrick { cards: [
// Some(DiamondTen), Some(DiamondKing), Some(HeartJack), Some(DiamondTen)
// ], start_player: 3 }), Some(DoTrick { cards: [
// Some(ClubAce), Some(SpadeJack), Some(ClubKing), Some(ClubTen)
// ], start_player: 1 }), Some(DoTrick { cards: [
// Some(SpadeAce), Some(SpadeNine), Some(SpadeAce), Some(SpadeTen)
// ], start_player: 2 }), Some(DoTrick { cards: [
// Some(HeartAce), Some(ClubNine), Some(HeartAce), Some(DiamondKing)
// ], start_player: 2 }), Some(DoTrick { cards: [
// Some(ClubKing), Some(DiamondNine), Some(ClubAce), Some(SpadeQueen)
// ], start_player: 1 }), Some(DoTrick { cards: [
// Some(DiamondJack), Some(HeartQueen), Some(DiamondQueen), Some(ClubQueen)
// ], start_player: 0 }), Some(DoTrick { cards: [
// Some(DiamondAce), Some(ClubQueen), Some(DiamondNine), Some(DiamondQueen)
// ], start_player: 3 }), Some(DoTrick { cards: [
// Some(DiamondAce), Some(SpadeQueen), Some(HeartQueen), Some(HeartTen)
// ], start_player: 0 }), Some(DoTrick { cards: [
// Some(SpadeTen), Some(DiamondJack), Some(SpadeNine), Some(HeartKing)
// ], start_player: 3 }), Some(DoTrick { cards: [
// Some(ClubJack), None, None, None
// ], start_player: 0 }), None]
// player_hand_observing_player: 4198400
// player_hands_length: [1, 2, 2, 2]
// observing_player: 1
// 2199024041984

/// Gibt die Anzahl der Karten in der Hand zurück.
pub fn hand_len(hand: DoHand) -> usize {
    return u64::count_ones(hand) as usize;
}

pub fn hand_to_vec_sorted_by_rank(
    hand: DoHand
) -> heapless::Vec<DoCard, 12> {
    let mut cards: heapless::Vec::<DoCard, 12> = heapless::Vec::new();
    macro_rules! add_card {
        ($card:expr) => {
            unsafe {
                if (hand_contains_both(hand, $card)) {
                    cards.push_unchecked($card);
                    cards.push_unchecked($card);
                } else if (hand_contains(hand, $card)) {
                    cards.push_unchecked($card);
                }
            }
        };
        () => {};
    }

    // Nach absteigenden Werten!!
    add_card!(DoCard::HeartTen);

    add_card!(DoCard::ClubQueen);
    add_card!(DoCard::SpadeQueen);
    add_card!(DoCard::HeartQueen);
    add_card!(DoCard::DiamondQueen);

    add_card!(DoCard::ClubJack);
    add_card!(DoCard::SpadeJack);
    add_card!(DoCard::HeartJack);
    add_card!(DoCard::DiamondJack);

    add_card!(DoCard::DiamondAce);
    add_card!(DoCard::DiamondTen);
    add_card!(DoCard::DiamondKing);
    add_card!(DoCard::DiamondNine);

    add_card!(DoCard::ClubAce);
    add_card!(DoCard::ClubTen);
    add_card!(DoCard::ClubKing);
    add_card!(DoCard::ClubNine);

    add_card!(DoCard::SpadeAce);
    add_card!(DoCard::SpadeTen);
    add_card!(DoCard::SpadeKing);
    add_card!(DoCard::SpadeNine);

    add_card!(DoCard::HeartAce);
    add_card!(DoCard::HeartKing);
    add_card!(DoCard::HeartNine);

    cards
}

pub fn hand_to_vec(hand: DoHand) -> Vec<DoCard> {
    let mut cards: Vec<DoCard> = Vec::with_capacity(DoCard::COUNT);

    macro_rules! add_card {
        ($card:expr) => {
            if (hand_contains_both(hand, $card)) {
                cards.push($card);
                cards.push($card);
            } else if (hand_contains(hand, $card)) {
                cards.push($card);
            }
        };
        () => {};
    }

    add_card!(DoCard::DiamondNine);
    add_card!(DoCard::DiamondTen);
    add_card!(DoCard::DiamondJack);
    add_card!(DoCard::DiamondQueen);
    add_card!(DoCard::DiamondKing);
    add_card!(DoCard::DiamondAce);

    add_card!(DoCard::HeartNine);
    add_card!(DoCard::HeartTen);
    add_card!(DoCard::HeartJack);
    add_card!(DoCard::HeartQueen);
    add_card!(DoCard::HeartKing);
    add_card!(DoCard::HeartAce);

    add_card!(DoCard::ClubNine);
    add_card!(DoCard::ClubTen);
    add_card!(DoCard::ClubJack);
    add_card!(DoCard::ClubQueen);
    add_card!(DoCard::ClubKing);
    add_card!(DoCard::ClubAce);

    add_card!(DoCard::SpadeNine);
    add_card!(DoCard::SpadeTen);
    add_card!(DoCard::SpadeJack);
    add_card!(DoCard::SpadeQueen);
    add_card!(DoCard::SpadeKing);
    add_card!(DoCard::SpadeAce);

    cards
}


/// Für Testzwecke, kann eine Hand aus einem Vektor von Karten erstellt werden.
pub fn hand_from_vec(cards: Vec<DoCard>) -> DoHand {
    let mut hand: DoHand = 0;

    for card in cards {
        if hand_contains(hand, card) {
            hand = bitflag_add::<DoHand, { DoCard::COUNT * 2 }>(hand, (card as u64) << SECOND_CARD_SHIFT_OFFSET);
        } else {
            hand = bitflag_add::<DoHand, { DoCard::COUNT * 2 }>(hand, card as u64);
        }
    }

    hand
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_contains() {
        let hand = hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondTen]);

        assert!(hand_contains(hand, DoCard::DiamondNine));
        assert!(hand_contains(hand, DoCard::DiamondTen));
        assert!(!hand_contains(hand, DoCard::DiamondJack));
    }

    #[test]
    fn test_hand_contains_both() {
        let hand = hand_from_vec(vec![DoCard::DiamondTen, DoCard::DiamondNine, DoCard::DiamondNine]);

        assert!(hand_contains_both(hand, DoCard::DiamondNine));
        assert!(!hand_contains_both(hand, DoCard::DiamondTen));
    }

    #[test]
    fn test_hand_from_vec() {
        let hand = hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondTen]);

        assert!(hand_contains(hand, DoCard::DiamondNine));
        assert!(hand_contains(hand, DoCard::DiamondTen));
        assert!(!hand_contains(hand, DoCard::DiamondJack));
    }

    #[test]
    fn test_hand_add() {
        let hand = hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondTen]);
        let hand = hand_add(hand, DoCard::DiamondJack);

        assert!(hand_contains(hand, DoCard::DiamondNine));
        assert!(hand_contains(hand, DoCard::DiamondTen));
        assert!(hand_contains(hand, DoCard::DiamondJack));
    }

    #[test]
    fn test_hand_len() {
        let hand = hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondNine]);

        assert_eq!(hand_len(hand), 3);
    }

    #[test]
    fn test_hand_remove() {
        let hand = hand_from_vec(vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondNine]);
        let hand = hand_remove(hand, DoCard::DiamondNine);

        assert!(hand_contains(hand, DoCard::DiamondNine));
        assert!(!hand_contains_both(hand, DoCard::DiamondNine));
        assert!(hand_contains(hand, DoCard::DiamondTen));

        let hand = hand_remove(hand, DoCard::DiamondNine);

        assert!(!hand_contains(hand, DoCard::DiamondNine));
        assert!(hand_contains(hand, DoCard::DiamondTen));
    }

}




