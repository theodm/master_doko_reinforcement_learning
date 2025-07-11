

pub type DoPlayer = usize;

pub const PLAYER_BOTTOM: DoPlayer = 0;
pub const PLAYER_LEFT: DoPlayer = 1;
pub const PLAYER_TOP: DoPlayer = 2;
pub const PLAYER_RIGHT: DoPlayer = 3;

/// Wandelt einen potenziell überlaufenden Spielerindex in einen gültigen Spielerindex um.
///
/// Bsp.:
///    player_wraparound(4) => 0
///    player_wraparound(5) => 1
///
/// Vorsicht: Kann nur für Überlauf, aber nicht für Unterlauf verwendet werden.
pub fn player_wraparound(player: DoPlayer) -> DoPlayer {
    return player % 4;
}

pub fn player_increase(player: DoPlayer) -> DoPlayer {
    return player_wraparound(player + 1);
}


#[macro_export]
macro_rules! debug_assert_valid_player {
    ($player:expr) => {
        debug_assert!($player < 4, "The player {} is not a valid player. There are only 4 players.", $player);
    };
}