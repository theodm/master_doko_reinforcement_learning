use rs_full_doko::card::cards::FdoCard;

/// Anzahl der Karten im Spiel (für die Embedding-Größe).
pub const CARD_OR_NONE_COUNT: i64 = 25;

/// Kodiert die Karte (oder None). Wird
/// als Embedding innerhalb des neuronalen
/// Netzwerkes verwendet.
pub fn encode_card_or_none(
    card: Option<FdoCard>
) -> [i64; 1] {
    pub fn map_card(card: Option<FdoCard>) -> i64 {
        match card {
            None => 0,
            Some(card) => {
                match card {
                    FdoCard::DiamondNine => 1,
                    FdoCard::DiamondTen => 2,
                    FdoCard::DiamondJack => 3,
                    FdoCard::DiamondQueen => 4,
                    FdoCard::DiamondKing => 5,
                    FdoCard::DiamondAce => 6,

                    FdoCard::HeartNine => 7,
                    FdoCard::HeartTen => 8,
                    FdoCard::HeartJack => 9,
                    FdoCard::HeartQueen => 10,
                    FdoCard::HeartKing => 11,
                    FdoCard::HeartAce => 12,

                    FdoCard::ClubNine => 13,
                    FdoCard::ClubTen => 14,
                    FdoCard::ClubJack => 15,
                    FdoCard::ClubQueen => 16,
                    FdoCard::ClubKing => 17,
                    FdoCard::ClubAce => 18,

                    FdoCard::SpadeNine => 19,
                    FdoCard::SpadeTen => 20,
                    FdoCard::SpadeJack => 21,
                    FdoCard::SpadeQueen => 22,
                    FdoCard::SpadeKing => 23,
                    FdoCard::SpadeAce => 24
                }
            }
        }
    }

    let card_num = map_card(card);

    debug_assert!(card_num < CARD_OR_NONE_COUNT);
    debug_assert!(card_num >= 0);

    [card_num as i64]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_card_or_none() {
        assert_eq!(encode_card_or_none(None), [0]);

        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondNine)), [1]);
        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondTen)), [2]);
        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondJack)), [3]);
        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondQueen)), [4]);
        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondKing)), [5]);
        assert_eq!(encode_card_or_none(Some(FdoCard::DiamondAce)), [6]);

        assert_eq!(encode_card_or_none(Some(FdoCard::HeartNine)), [7]);
        assert_eq!(encode_card_or_none(Some(FdoCard::HeartTen)), [8]);
        assert_eq!(encode_card_or_none(Some(FdoCard::HeartJack)), [9]);
        assert_eq!(encode_card_or_none(Some(FdoCard::HeartQueen)), [10]);
        assert_eq!(encode_card_or_none(Some(FdoCard::HeartKing)), [11]);
        assert_eq!(encode_card_or_none(Some(FdoCard::HeartAce)), [12]);

        assert_eq!(encode_card_or_none(Some(FdoCard::ClubNine)), [13]);
        assert_eq!(encode_card_or_none(Some(FdoCard::ClubTen)), [14]);
        assert_eq!(encode_card_or_none(Some(FdoCard::ClubJack)), [15]);
        assert_eq!(encode_card_or_none(Some(FdoCard::ClubQueen)), [16]);
        assert_eq!(encode_card_or_none(Some(FdoCard::ClubKing)), [17]);
        assert_eq!(encode_card_or_none(Some(FdoCard::ClubAce)), [18]);

        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeNine)), [19]);
        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeTen)), [20]);
        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeJack)), [21]);
        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeQueen)), [22]);
        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeKing)), [23]);
        assert_eq!(encode_card_or_none(Some(FdoCard::SpadeAce)), [24]);
    }
}