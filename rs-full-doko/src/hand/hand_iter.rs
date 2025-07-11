use crate::card::cards::FdoCard;
use crate::hand::hand::FdoHand;
// Adjust the import as needed

/// The shift offset for the second copy of a card.
const SECOND_CARD_SHIFT_OFFSET: u64 = 24;

/// An iterator over all cards in an `FdoHand`.
///
/// Each occurrence of a card is yielded as a separate item.
pub struct FdoHandIter {
    // A copy of the underlying bit mask.
    bits: u64,
}

impl Iterator for FdoHandIter {
    type Item = FdoCard;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            return None;
        }
        // Find the lowest set bit.
        let pos = self.bits.trailing_zeros() as u64;
        // Clear that bit.
        self.bits &= self.bits - 1;
        // Determine the exponent (or index) corresponding to the card.
        let card_exponent = if pos < SECOND_CARD_SHIFT_OFFSET {
            pos
        } else {
            pos - SECOND_CARD_SHIFT_OFFSET
        };
        // Compute the card's discriminant value. For example, if card_exponent is 3,
        // then card_value becomes 1 << 3 (i.e. 8). In your enum, 8 corresponds to DiamondQueen.
        let card_value = 1usize << card_exponent;
        // SAFETY: We assume that `card_value` is exactly one of the discriminants defined in FdoCard.
        Some(unsafe { std::mem::transmute(card_value) })
    }
}

impl FdoHand {
    /// Returns an iterator over all cards in the hand.
    ///
    /// Each occurrence is yielded separately. (For example, if a card is present twice,
    /// it will appear twice in the iterator.)
    pub fn iter(&self) -> FdoHandIter {
        FdoHandIter { bits: self.0.0 }
    }
}

/// Optional: implement IntoIterator for &FdoHand so you can use it in a for‑loop directly.
impl<'a> IntoIterator for &'a FdoHand {
    type Item = FdoCard;
    type IntoIter = FdoHandIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod iter_tests {
    use super::*;
    use crate::card::cards::FdoCard;

    #[test]
    fn test_iter() {
        // Example: Create a hand from a string (adjust according to your helpers).
        let hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣Q"); // Note: ♣Q appears twice.
        let mut cards: Vec<FdoCard> = hand.iter().collect();
        // Sort by the underlying usize value for deterministic comparison.
        cards.sort_by_key(|card| *card as usize);
        let mut expected = FdoCard::vec_from_str("♣Q ♣Q ♣10 ♥9");
        expected.sort_by_key(|card| *card as usize);
        assert_eq!(cards, expected);
    }
}
