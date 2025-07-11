use rs_full_doko::player::player::FdoPlayer;

/// Anzahl der Spieler im Spiel (für die Embedding-Größe).
pub const PLAYER_OR_NONE_COUNT: i64 = 5;

fn player_index_relative(
    player: FdoPlayer,
    relative_to_player: FdoPlayer,
) -> i64 {
    let index = player.index() as i64;
    let relative_to_index = relative_to_player.index() as i64;

    (index - relative_to_index).rem_euclid(4)
}

/// Kodiert den Spieler (oder null). Wird
/// als Embedding innerhalb des neuronalen
/// Netzwerkes verwendet.
pub fn encode_player_or_none(
    player: Option<FdoPlayer>,

    relative_to_player: FdoPlayer,
) -> [i64; 1] {
    let player_num = player
        .map_or(0, |p| player_index_relative(p, relative_to_player) + 1);

    debug_assert!(player_num < PLAYER_OR_NONE_COUNT as i64);
    debug_assert!(player_num >= 0);

    [player_num as i64]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_index_relative() {
        // Bezug: Bottom
        assert_eq!(player_index_relative(FdoPlayer::BOTTOM, FdoPlayer::BOTTOM), 0);
        assert_eq!(player_index_relative(FdoPlayer::LEFT, FdoPlayer::BOTTOM), 1);
        assert_eq!(player_index_relative(FdoPlayer::TOP, FdoPlayer::BOTTOM), 2);
        assert_eq!(player_index_relative(FdoPlayer::RIGHT, FdoPlayer::BOTTOM), 3);

        // Bezug: Right
        assert_eq!(player_index_relative(FdoPlayer::LEFT, FdoPlayer::LEFT), 0);
        assert_eq!(player_index_relative(FdoPlayer::TOP, FdoPlayer::LEFT), 1);
        assert_eq!(player_index_relative(FdoPlayer::RIGHT, FdoPlayer::LEFT), 2);
        assert_eq!(player_index_relative(FdoPlayer::BOTTOM, FdoPlayer::LEFT), 3);

        // Bezug: Top
        assert_eq!(player_index_relative(FdoPlayer::TOP, FdoPlayer::TOP), 0);
        assert_eq!(player_index_relative(FdoPlayer::RIGHT, FdoPlayer::TOP), 1);
        assert_eq!(player_index_relative(FdoPlayer::BOTTOM, FdoPlayer::TOP), 2);
        assert_eq!(player_index_relative(FdoPlayer::LEFT, FdoPlayer::TOP), 3);

        // Bezug: Left
        assert_eq!(player_index_relative(FdoPlayer::RIGHT, FdoPlayer::RIGHT), 0);
        assert_eq!(player_index_relative(FdoPlayer::BOTTOM, FdoPlayer::RIGHT), 1);
        assert_eq!(player_index_relative(FdoPlayer::LEFT, FdoPlayer::RIGHT), 2);
        assert_eq!(player_index_relative(FdoPlayer::TOP, FdoPlayer::RIGHT), 3);
    }

    #[test]
    fn test_encode_player_or_none() {
        assert_eq!(encode_player_or_none(None, FdoPlayer::BOTTOM), [0]);

        assert_eq!(encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::BOTTOM), [1]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::BOTTOM), [2]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::BOTTOM), [3]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::BOTTOM), [4]);

        assert_eq!(encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::LEFT), [1]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::LEFT), [2]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::LEFT), [3]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::LEFT), [4]);

        assert_eq!(encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::TOP), [1]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::TOP), [2]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::TOP), [3]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::TOP), [4]);

        assert_eq!(encode_player_or_none(Some(FdoPlayer::RIGHT), FdoPlayer::RIGHT), [1]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::BOTTOM), FdoPlayer::RIGHT), [2]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::LEFT), FdoPlayer::RIGHT), [3]);
        assert_eq!(encode_player_or_none(Some(FdoPlayer::TOP), FdoPlayer::RIGHT), [4]);
    }
}