use crate::card::cards::DoCard;

pub const TRUMP_COLOR_MASK_NORMAL_GAME: usize = DoCard::DiamondNine as usize
    | DoCard::DiamondKing as usize
    | DoCard::DiamondTen as usize
    | DoCard::DiamondAce as usize
    | DoCard::DiamondJack as usize
    | DoCard::HeartJack as usize
    | DoCard::SpadeJack as usize
    | DoCard::ClubJack as usize
    | DoCard::DiamondQueen as usize
    | DoCard::HeartQueen as usize
    | DoCard::SpadeQueen as usize
    | DoCard::ClubQueen as usize
    | DoCard::HeartTen as usize;

pub const HEART_COLOR_MASK_NORMAL_GAME: usize = DoCard::HeartNine as usize
    | DoCard::HeartKing as usize
    | DoCard::HeartAce as usize;

pub const SPADE_COLOR_MASK_NORMAL_GAME: usize = DoCard::SpadeNine as usize
    | DoCard::SpadeKing as usize
    | DoCard::SpadeTen as usize
    | DoCard::SpadeAce as usize;

pub const CLUB_COLOR_MASK_NORMAL_GAME: usize = DoCard::ClubNine as usize
    | DoCard::ClubKing as usize
    | DoCard::ClubTen as usize
    | DoCard::ClubAce as usize;

#[cfg(test)]
mod tests {
    use crate::card::card_color_masks::{CLUB_COLOR_MASK_NORMAL_GAME, HEART_COLOR_MASK_NORMAL_GAME, SPADE_COLOR_MASK_NORMAL_GAME, TRUMP_COLOR_MASK_NORMAL_GAME};

    #[test]
    fn test_masks() {
        assert_eq!(TRUMP_COLOR_MASK_NORMAL_GAME, 0b00110000_11000011_10111111);
        assert_eq!(HEART_COLOR_MASK_NORMAL_GAME, 0b00000000_00001100_01000000);
        assert_eq!(SPADE_COLOR_MASK_NORMAL_GAME, 0b11001100_00000000_00000000);
        assert_eq!(CLUB_COLOR_MASK_NORMAL_GAME,  0b00000011_00110000_00000000);
    }

}