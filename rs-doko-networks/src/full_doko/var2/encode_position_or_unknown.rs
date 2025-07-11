use rs_full_doko::player::player::FdoPlayer;
use crate::full_doko::var1::player::PLAYER_OR_NONE_COUNT;

fn player_index_relative(
    player: FdoPlayer,
    relative_to_player: FdoPlayer,
) -> i64 {
    let index = player.index() as i64;
    let relative_to_index = relative_to_player.index() as i64;

    (index - relative_to_index).rem_euclid(4)
}

pub fn encode_position_or_unknown_hand(
    player: Option<FdoPlayer>,

    relative_to_player: FdoPlayer
) -> [i64; 1] {
    match player {
        None => return [0],
        Some(player) => [
            return [player_index_relative(player, relative_to_player) + 1 + 52]
        ]
    }
}

pub fn encode_position_or_unknown_int(
    position: usize
) -> [i64; 1] {
    assert!(position >= 0 && position < 52);

    let position = position + 1;

    [position as i64]
}

#[cfg(test)]
mod tests {
    use rs_full_doko::player::player::FdoPlayer;
    use crate::full_doko::var2::encode_position_or_unknown::{encode_position_or_unknown_hand, encode_position_or_unknown_int};

    #[cfg(test)]
    mod tests {
        use super::*;
        use rs_full_doko::player::player::FdoPlayer;
        use crate::full_doko::var2::encode_position_or_unknown::player_index_relative;

        #[test]
        fn test_player_index_relative() {
            let p0 = FdoPlayer::from_index(0);
            let p1 = FdoPlayer::from_index(1);
            let p2 = FdoPlayer::from_index(2);
            let p3 = FdoPlayer::from_index(3);

            assert_eq!(player_index_relative(p0, p0), 0);
            assert_eq!(player_index_relative(p1, p0), 1);
            assert_eq!(player_index_relative(p2, p0), 2);
            assert_eq!(player_index_relative(p3, p0), 3);

            assert_eq!(player_index_relative(p0, p1), 3);
            assert_eq!(player_index_relative(p1, p1), 0);
            assert_eq!(player_index_relative(p2, p1), 1);
            assert_eq!(player_index_relative(p3, p1), 2);

            assert_eq!(player_index_relative(p0, p2), 2);
            assert_eq!(player_index_relative(p1, p2), 3);
            assert_eq!(player_index_relative(p2, p2), 0);
            assert_eq!(player_index_relative(p3, p2), 1);

            assert_eq!(player_index_relative(p0, p3), 1);
            assert_eq!(player_index_relative(p1, p3), 2);
            assert_eq!(player_index_relative(p2, p3), 3);
            assert_eq!(player_index_relative(p3, p3), 0);
        }
    }

    #[test]
    fn test_encode_position_or_unknown_hand() {
        let p0 = FdoPlayer::from_index(0);
        let p1 = FdoPlayer::from_index(1);
        let p2 = FdoPlayer::from_index(2);
        let p3 = FdoPlayer::from_index(3);

        // Test mit bekanntem Spieler
        assert_eq!(encode_position_or_unknown_hand(Some(p0), p0), [53]); // 0 + 1 + 52
        assert_eq!(encode_position_or_unknown_hand(Some(p1), p0), [54]); // 1 + 1 + 52
        assert_eq!(encode_position_or_unknown_hand(Some(p2), p0), [55]); // 2 + 1 + 52
        assert_eq!(encode_position_or_unknown_hand(Some(p3), p0), [56]); // 3 + 1 + 52

        // Test mit unbekanntem Spieler
        assert_eq!(encode_position_or_unknown_hand(None, p0), [0]); // Kein Spieler
    }

    #[test]
    fn test_encode_position_or_unknown_int() {
        for pos in 0..52 {
            assert_eq!(encode_position_or_unknown_int(pos), [(pos + 1) as i64]);
        }
    }

    #[test]
    #[should_panic]
    fn test_encode_position_or_unknown_int_out_of_bounds_low() {
        encode_position_or_unknown_int(usize::MIN);
    }

    #[test]
    #[should_panic]
    fn test_encode_position_or_unknown_int_out_of_bounds_high() {
        encode_position_or_unknown_int(52); 
    }

}