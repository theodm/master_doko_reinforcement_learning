use crate::card::cards::DoCard;
use crate::hand::hand::DoHand;
use crate::player::player::{DoPlayer};
use crate::util::bitflag::bitflag::{bitflag_add, bitflag_contains, bitflag_number_of_ones};

pub type DoPlayerSet = usize;

pub fn player_set_add(player_set: DoPlayerSet, player_index: DoPlayer) -> DoPlayerSet {
    return bitflag_add::<DoPlayerSet, 4>(player_set, 1 << player_index);
}

pub fn player_set_contains(player_set: DoPlayerSet, player_index: DoPlayer) -> bool {
    return bitflag_contains::<DoPlayerSet, 4>(player_set, 1 << player_index);
}

pub fn player_set_len(player_set: DoPlayerSet) -> u32 {
    return bitflag_number_of_ones::<DoPlayerSet, 4>(player_set);
}

pub fn player_set_inverse(player_set: DoPlayerSet) -> DoPlayerSet {
    return !player_set & 0b1111;
}

/// Für Testzwecke
pub fn player_set_to_vec(player_set: DoPlayerSet) -> Vec<DoPlayer> {
    let mut players: Vec<DoPlayer> = Vec::new();

    for player_index in 0..4 {
        if player_set_contains(player_set, player_index) {
            players.push(player_index);
        }
    }

    return players;
}

/// Für Testzwecke.
pub fn player_set_create(players: Vec<DoPlayer>) -> DoPlayerSet {
    let mut player_set: DoPlayerSet = 0;

    for player in players {
        player_set = player_set_add(player_set, player);
    }

    return player_set;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::player::PLAYER_BOTTOM;
    use crate::player::player::PLAYER_LEFT;
    use crate::player::player::PLAYER_RIGHT;
    use crate::player::player::PLAYER_TOP;

    #[test]
    fn test_player_set_add() {
        let player_set = player_set_add(0, PLAYER_BOTTOM);
        assert!(player_set_contains(player_set, PLAYER_BOTTOM));

        let player_set = player_set_add(player_set, PLAYER_LEFT);
        assert!(player_set_contains(player_set, PLAYER_BOTTOM));
        assert!(player_set_contains(player_set, PLAYER_LEFT));

        let player_set = player_set_add(player_set, PLAYER_RIGHT);
        assert!(player_set_contains(player_set, PLAYER_BOTTOM));
        assert!(player_set_contains(player_set, PLAYER_LEFT));
        assert!(player_set_contains(player_set, PLAYER_RIGHT));

        let player_set = player_set_add(player_set, PLAYER_TOP);
        assert!(player_set_contains(player_set, PLAYER_BOTTOM));
        assert!(player_set_contains(player_set, PLAYER_LEFT));
        assert!(player_set_contains(player_set, PLAYER_RIGHT));
        assert!(player_set_contains(player_set, PLAYER_TOP));
    }

    #[test]
    fn test_player_set_contains() {
        let player_set = player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP]);

        assert!(player_set_contains(player_set, PLAYER_BOTTOM));
        assert!(player_set_contains(player_set, PLAYER_LEFT));
        assert!(player_set_contains(player_set, PLAYER_RIGHT));
        assert!(player_set_contains(player_set, PLAYER_TOP));
    }

    #[test]
    fn test_player_set_len() {
        let player_set = player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP]);

        assert_eq!(player_set_len(player_set), 4);

        let player_set = player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT]);

        assert_eq!(player_set_len(player_set), 2);
    }

    #[test]
    fn test_player_set_inverse() {
        let player_set = player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP]);

        let inverse = player_set_inverse(player_set);

        assert!(!player_set_contains(inverse, PLAYER_BOTTOM));
        assert!(!player_set_contains(inverse, PLAYER_LEFT));
        assert!(!player_set_contains(inverse, PLAYER_RIGHT));
        assert!(!player_set_contains(inverse, PLAYER_TOP));

        let player_set = player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT]);

        let inverse = player_set_inverse(player_set);

        assert!(!player_set_contains(inverse, PLAYER_BOTTOM));
        assert!(!player_set_contains(inverse, PLAYER_LEFT));
        assert!(player_set_contains(inverse, PLAYER_RIGHT));
        assert!(player_set_contains(inverse, PLAYER_TOP));
    }
}