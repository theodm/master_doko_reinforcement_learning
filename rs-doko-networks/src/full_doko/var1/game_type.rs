use rs_full_doko::game_type::game_type::FdoGameType;

/// Anzahl der Spieltypen (oder None) im Spiel (für die Embedding-Größe).
pub const GAME_TYPE_OR_NONE_COUNT: i64 = 10;

pub fn encode_game_type_or_none(
    game_type: Option<FdoGameType>
) -> [i64; 1] {
    fn map_game_type(
        game_type: Option<FdoGameType>
    ) -> i64 {
        match game_type {
            None => 0,
            Some(game_type) => {
                match game_type {
                    FdoGameType::Normal => 1,
                    FdoGameType::Wedding => 2,
                    FdoGameType::TrumplessSolo => 3,
                    FdoGameType::QueensSolo => 4,
                    FdoGameType::JacksSolo => 5,
                    FdoGameType::DiamondsSolo => 6,
                    FdoGameType::HeartsSolo => 7,
                    FdoGameType::SpadesSolo => 8,
                    FdoGameType::ClubsSolo => 9
                }
            }
        }
    }

    let game_type_num = map_game_type(game_type);

    debug_assert!(game_type_num < GAME_TYPE_OR_NONE_COUNT);
    debug_assert!(game_type_num >= 0);

    [game_type_num]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_game_type_or_none() {
        assert_eq!(encode_game_type_or_none(None), [0]);

        assert_eq!(encode_game_type_or_none(Some(FdoGameType::Normal)), [1]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::Wedding)), [2]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::TrumplessSolo)), [3]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::QueensSolo)), [4]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::JacksSolo)), [5]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::DiamondsSolo)), [6]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::HeartsSolo)), [7]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::SpadesSolo)), [8]);
        assert_eq!(encode_game_type_or_none(Some(FdoGameType::ClubsSolo)), [9]);
    }
}