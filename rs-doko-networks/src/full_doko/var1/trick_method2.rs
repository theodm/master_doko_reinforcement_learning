use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::trick::trick::FdoTrick;
use crate::full_doko::var1::player::encode_player_or_none;

pub fn encode_tricks_method_2(
    tricks: &heapless::Vec<FdoTrick, 12>,

    current_player: FdoPlayer
) -> ([i64; 48], [i64; 48]) {
    let zero_player = encode_player_or_none(None, current_player)[0];

    let mut card_was_played_by_player
        = [zero_player; 48];
    let mut card_was_played_at_position
        = [0; 48];

    let mut card_position = 0;

    for trick in tricks.iter() {
        for (player, card) in trick.iter_with_player() {
            let index = match card {
                FdoCard::HeartTen => 0,

                FdoCard::ClubQueen => 1,
                FdoCard::SpadeQueen => 2,
                FdoCard::HeartQueen => 3,
                FdoCard::DiamondQueen => 4,

                FdoCard::ClubJack => 5,
                FdoCard::SpadeJack => 6,
                FdoCard::HeartJack => 7,
                FdoCard::DiamondJack => 8,

                FdoCard::DiamondAce => 9,
                FdoCard::DiamondTen => 10,
                FdoCard::DiamondKing => 11,
                FdoCard::DiamondNine => 12,

                FdoCard::ClubAce => 13,
                FdoCard::ClubTen => 14,
                FdoCard::ClubKing => 15,
                FdoCard::ClubNine => 16,

                FdoCard::SpadeAce => 17,
                FdoCard::SpadeTen => 18,
                FdoCard::SpadeKing => 19,
                FdoCard::SpadeNine => 20,

                FdoCard::HeartAce => 21,
                FdoCard::HeartKing => 22,
                FdoCard::HeartNine => 23
            };

            if (card_was_played_by_player[index * 2] == zero_player) {
                // Platz 1 belegen
                card_was_played_by_player[index * 2] = encode_player_or_none(Some(player), current_player)[0];
                card_was_played_at_position[index * 2] = card_position;
            }

            // 2. Platz belegen
            else {
                card_was_played_by_player[index * 2 + 1] = encode_player_or_none(Some(player), current_player)[0];
                card_was_played_at_position[index * 2 + 1] = card_position;
            }

            card_position += 1;
        }
    }

    (card_was_played_by_player, card_was_played_at_position)
}