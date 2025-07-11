use crate::card::cards::DoCard;
use crate::reservation::reservation::DoReservation;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCountMacro, Hash)]
pub enum DoAction {
    CardDiamondNine = 1 << 0,
    CardDiamondTen = 1 << 1,
    CardDiamondJack = 1 << 2,
    CardDiamondQueen = 1 << 3,
    CardDiamondKing = 1 << 4,
    CardDiamondAce = 1 << 5,

    CardHeartNine = 1 << 6,
    CardHeartTen = 1 << 7,
    CardHeartJack = 1 << 8,
    CardHeartQueen = 1 << 9,
    CardHeartKing = 1 << 10,
    CardHeartAce = 1 << 11,

    CardClubNine = 1 << 12,
    CardClubTen = 1 << 13,
    CardClubJack = 1 << 14,
    CardClubQueen = 1 << 15,
    CardClubKing = 1 << 16,
    CardClubAce = 1 << 17,

    CardSpadeNine = 1 << 18,
    CardSpadeTen = 1 << 19,
    CardSpadeJack = 1 << 20,
    CardSpadeQueen = 1 << 21,
    CardSpadeKing = 1 << 22,
    CardSpadeAce = 1 << 23,

    ReservationHealthy = 1 << 24,
    ReservationWedding = 1 << 25,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoActionType {
    Card(DoCard),
    Reservation(DoReservation),
}

pub fn action_to_type(action: DoAction) -> DoActionType {
    let is_card_action = action as usize <= DoAction::CardSpadeAce as usize;
    let is_reservation_action = action as usize > DoAction::CardSpadeAce as usize;

    if is_card_action {
        match action {
            DoAction::CardDiamondNine => DoActionType::Card(DoCard::DiamondNine),
            DoAction::CardDiamondTen => DoActionType::Card(DoCard::DiamondTen),
            DoAction::CardDiamondJack => DoActionType::Card(DoCard::DiamondJack),
            DoAction::CardDiamondQueen => DoActionType::Card(DoCard::DiamondQueen),
            DoAction::CardDiamondKing => DoActionType::Card(DoCard::DiamondKing),
            DoAction::CardDiamondAce => DoActionType::Card(DoCard::DiamondAce),

            DoAction::CardHeartNine => DoActionType::Card(DoCard::HeartNine),
            DoAction::CardHeartTen => DoActionType::Card(DoCard::HeartTen),
            DoAction::CardHeartJack => DoActionType::Card(DoCard::HeartJack),
            DoAction::CardHeartQueen => DoActionType::Card(DoCard::HeartQueen),
            DoAction::CardHeartKing => DoActionType::Card(DoCard::HeartKing),
            DoAction::CardHeartAce => DoActionType::Card(DoCard::HeartAce),

            DoAction::CardClubNine => DoActionType::Card(DoCard::ClubNine),
            DoAction::CardClubTen => DoActionType::Card(DoCard::ClubTen),
            DoAction::CardClubJack => DoActionType::Card(DoCard::ClubJack),
            DoAction::CardClubQueen => DoActionType::Card(DoCard::ClubQueen),
            DoAction::CardClubKing => DoActionType::Card(DoCard::ClubKing),
            DoAction::CardClubAce => DoActionType::Card(DoCard::ClubAce),

            DoAction::CardSpadeNine => DoActionType::Card(DoCard::SpadeNine),
            DoAction::CardSpadeTen => DoActionType::Card(DoCard::SpadeTen),
            DoAction::CardSpadeJack => DoActionType::Card(DoCard::SpadeJack),
            DoAction::CardSpadeQueen => DoActionType::Card(DoCard::SpadeQueen),
            DoAction::CardSpadeKing => DoActionType::Card(DoCard::SpadeKing),
            DoAction::CardSpadeAce => DoActionType::Card(DoCard::SpadeAce),

            _ => panic!("Invalid card action: {:?}", action),
        }
    } else if is_reservation_action {
        match action {
            DoAction::ReservationHealthy => DoActionType::Reservation(DoReservation::Healthy),
            DoAction::ReservationWedding => DoActionType::Reservation(DoReservation::Wedding),

            _ => panic!("Invalid reservation action: {:?}", action),
        }
    } else {
        panic!("Invalid action: {:?}", action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_card_action_to_card() {
        assert_eq!(action_to_type(DoAction::CardDiamondNine), DoActionType::Card(DoCard::DiamondNine));
        assert_eq!(action_to_type(DoAction::CardDiamondTen), DoActionType::Card(DoCard::DiamondTen));
        assert_eq!(action_to_type(DoAction::CardDiamondJack), DoActionType::Card(DoCard::DiamondJack));
        assert_eq!(action_to_type(DoAction::CardDiamondQueen), DoActionType::Card(DoCard::DiamondQueen));
        assert_eq!(action_to_type(DoAction::CardDiamondKing), DoActionType::Card(DoCard::DiamondKing));
        assert_eq!(action_to_type(DoAction::CardDiamondAce), DoActionType::Card(DoCard::DiamondAce));

        assert_eq!(action_to_type(DoAction::CardHeartNine), DoActionType::Card(DoCard::HeartNine));
        assert_eq!(action_to_type(DoAction::CardHeartTen), DoActionType::Card(DoCard::HeartTen));
        assert_eq!(action_to_type(DoAction::CardHeartJack), DoActionType::Card(DoCard::HeartJack));
        assert_eq!(action_to_type(DoAction::CardHeartQueen), DoActionType::Card(DoCard::HeartQueen));
        assert_eq!(action_to_type(DoAction::CardHeartKing), DoActionType::Card(DoCard::HeartKing));
        assert_eq!(action_to_type(DoAction::CardHeartAce), DoActionType::Card(DoCard::HeartAce));

        assert_eq!(action_to_type(DoAction::CardClubNine), DoActionType::Card(DoCard::ClubNine));
        assert_eq!(action_to_type(DoAction::CardClubTen), DoActionType::Card(DoCard::ClubTen));
        assert_eq!(action_to_type(DoAction::CardClubJack), DoActionType::Card(DoCard::ClubJack));
        assert_eq!(action_to_type(DoAction::CardClubQueen), DoActionType::Card(DoCard::ClubQueen));
        assert_eq!(action_to_type(DoAction::CardClubKing), DoActionType::Card(DoCard::ClubKing));
        assert_eq!(action_to_type(DoAction::CardClubAce), DoActionType::Card(DoCard::ClubAce));

        assert_eq!(action_to_type(DoAction::CardSpadeNine), DoActionType::Card(DoCard::SpadeNine));
        assert_eq!(action_to_type(DoAction::CardSpadeTen), DoActionType::Card(DoCard::SpadeTen));
        assert_eq!(action_to_type(DoAction::CardSpadeJack), DoActionType::Card(DoCard::SpadeJack));
        assert_eq!(action_to_type(DoAction::CardSpadeQueen), DoActionType::Card(DoCard::SpadeQueen));
        assert_eq!(action_to_type(DoAction::CardSpadeKing), DoActionType::Card(DoCard::SpadeKing));
        assert_eq!(action_to_type(DoAction::CardSpadeAce), DoActionType::Card(DoCard::SpadeAce));

        assert_eq!(action_to_type(DoAction::ReservationHealthy), DoActionType::Reservation(DoReservation::Healthy));
        assert_eq!(action_to_type(DoAction::ReservationWedding), DoActionType::Reservation(DoReservation::Wedding));
    }

}