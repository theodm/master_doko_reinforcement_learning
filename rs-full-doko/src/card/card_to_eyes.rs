use crate::card::cards::FdoCard;

impl FdoCard {
    /// Gibt den Augenwert der Karte zurück.
    ///
    /// z.B.:
    /// FdoCard::DiamondNine => 0
    pub fn eyes(
        &self,
    ) -> u32 {
        match self {
            FdoCard::DiamondNine => 0,
            FdoCard::DiamondTen => 10,
            FdoCard::DiamondJack => 2,
            FdoCard::DiamondQueen => 3,
            FdoCard::DiamondKing => 4,
            FdoCard::DiamondAce => 11,

            FdoCard::HeartNine => 0,
            FdoCard::HeartTen => 10,
            FdoCard::HeartJack => 2,
            FdoCard::HeartQueen => 3,
            FdoCard::HeartKing => 4,
            FdoCard::HeartAce => 11,

            FdoCard::SpadeNine => 0,
            FdoCard::SpadeTen => 10,
            FdoCard::SpadeJack => 2,
            FdoCard::SpadeQueen => 3,
            FdoCard::SpadeKing => 4,
            FdoCard::SpadeAce => 11,

            FdoCard::ClubNine => 0,
            FdoCard::ClubTen => 10,
            FdoCard::ClubJack => 2,
            FdoCard::ClubQueen => 3,
            FdoCard::ClubKing => 4,
            FdoCard::ClubAce => 11,
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_to_eyes() {
        assert_eq!(FdoCard::DiamondNine.eyes(), 0);
        assert_eq!(FdoCard::DiamondTen.eyes(), 10);
        assert_eq!(FdoCard::DiamondJack.eyes(), 2);
        assert_eq!(FdoCard::DiamondQueen.eyes(), 3);
        assert_eq!(FdoCard::DiamondKing.eyes(), 4);
        assert_eq!(FdoCard::DiamondAce.eyes(), 11);

        assert_eq!(FdoCard::HeartNine.eyes(), 0);
        assert_eq!(FdoCard::HeartTen.eyes(), 10);
        assert_eq!(FdoCard::HeartJack.eyes(), 2);
        assert_eq!(FdoCard::HeartQueen.eyes(), 3);
        assert_eq!(FdoCard::HeartKing.eyes(), 4);
        assert_eq!(FdoCard::HeartAce.eyes(), 11);

        assert_eq!(FdoCard::SpadeNine.eyes(), 0);
        assert_eq!(FdoCard::SpadeTen.eyes(), 10);
        assert_eq!(FdoCard::SpadeJack.eyes(), 2);
        assert_eq!(FdoCard::SpadeQueen.eyes(), 3);
        assert_eq!(FdoCard::SpadeKing.eyes(), 4);
        assert_eq!(FdoCard::SpadeAce.eyes(), 11);

        assert_eq!(FdoCard::ClubNine.eyes(), 0);
        assert_eq!(FdoCard::ClubTen.eyes(), 10);
        assert_eq!(FdoCard::ClubJack.eyes(), 2);
        assert_eq!(FdoCard::ClubQueen.eyes(), 3);
        assert_eq!(FdoCard::ClubKing.eyes(), 4);
        assert_eq!(FdoCard::ClubAce.eyes(), 11);
    }
}