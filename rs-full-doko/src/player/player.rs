use std::fmt::{Debug, Display, Formatter};
use std::ops::Add;
use enumset::EnumSetType;
use rand::Rng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use crate::basic::team::FdoTeam;
use crate::player::player_set::FdoPlayerSet;

#[derive(EnumSetType, Hash, Serialize, Deserialize)]
pub enum FdoPlayer {
    BOTTOM,
    LEFT,
    TOP,
    RIGHT,
}


impl Debug for FdoPlayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoPlayer::BOTTOM => write!(f, "FdoPlayer::BOTTOM"),
            FdoPlayer::LEFT => write!(f, "FdoPlayer::LEFT"),
            FdoPlayer::TOP => write!(f, "FdoPlayer::TOP"),
            FdoPlayer::RIGHT => write!(f, "FdoPlayer::RIGHT"),
        }
    }
}

impl Display for FdoPlayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoPlayer::BOTTOM => write!(f, "Unten"),
            FdoPlayer::LEFT => write!(f, "Links"),
            FdoPlayer::TOP => write!(f, "Oben"),
            FdoPlayer::RIGHT => write!(f, "Rechts"),
        }
    }
}

impl FdoPlayer {
    pub fn from_rng(
        rng: &mut SmallRng
    ) -> FdoPlayer {
        FdoPlayer::from_index(rng.random_range(0..4))
    }

    pub fn from_index(index: usize) -> FdoPlayer {
        match index {
            0 => FdoPlayer::BOTTOM,
            1 => FdoPlayer::LEFT,
            2 => FdoPlayer::TOP,
            3 => FdoPlayer::RIGHT,
            _ => panic!("Ungültiger Spielerindex: {}", index),
        }
    }

    pub fn index(&self) -> usize {
        match self {
            FdoPlayer::BOTTOM => 0,
            FdoPlayer::LEFT => 1,
            FdoPlayer::TOP => 2,
            FdoPlayer::RIGHT => 3,
        }
    }

    pub fn next(
        self,
        i: usize
    ) -> FdoPlayer {
        FdoPlayer::from_index((self.index() + i) % 4)
    }

    pub fn team(
        &self,
        re_players: FdoPlayerSet
    ) -> FdoTeam {
        if re_players.contains(*self) {
            FdoTeam::Re
        } else {
            FdoTeam::Kontra
        }
    }
}

impl Add<usize> for FdoPlayer {
    type Output = FdoPlayer;

    fn add(self, rhs: usize) -> Self::Output {
        self.next(rhs)
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    fn test_next() {
        assert_eq!(FdoPlayer::BOTTOM.next(0), FdoPlayer::BOTTOM);
        assert_eq!(FdoPlayer::BOTTOM.next(1), FdoPlayer::LEFT);
        assert_eq!(FdoPlayer::BOTTOM.next(2), FdoPlayer::TOP);
        assert_eq!(FdoPlayer::BOTTOM.next(3), FdoPlayer::RIGHT);
        assert_eq!(FdoPlayer::BOTTOM.next(4), FdoPlayer::BOTTOM);
        assert_eq!(FdoPlayer::BOTTOM.next(5), FdoPlayer::LEFT);

        assert_eq!(FdoPlayer::LEFT.next(0), FdoPlayer::LEFT);
        assert_eq!(FdoPlayer::LEFT.next(1), FdoPlayer::TOP);
        assert_eq!(FdoPlayer::LEFT.next(2), FdoPlayer::RIGHT);
        assert_eq!(FdoPlayer::LEFT.next(3), FdoPlayer::BOTTOM);
        assert_eq!(FdoPlayer::LEFT.next(4), FdoPlayer::LEFT);
        assert_eq!(FdoPlayer::LEFT.next(5), FdoPlayer::TOP);

        assert_eq!(FdoPlayer::TOP.next(0), FdoPlayer::TOP);
        assert_eq!(FdoPlayer::TOP.next(1), FdoPlayer::RIGHT);
        assert_eq!(FdoPlayer::TOP.next(2), FdoPlayer::BOTTOM);
        assert_eq!(FdoPlayer::TOP.next(3), FdoPlayer::LEFT);
        assert_eq!(FdoPlayer::TOP.next(4), FdoPlayer::TOP);

        assert_eq!(FdoPlayer::RIGHT.next(0), FdoPlayer::RIGHT);
        assert_eq!(FdoPlayer::RIGHT.next(1), FdoPlayer::BOTTOM);
        assert_eq!(FdoPlayer::RIGHT.next(2), FdoPlayer::LEFT);
        assert_eq!(FdoPlayer::RIGHT.next(3), FdoPlayer::TOP);
        assert_eq!(FdoPlayer::RIGHT.next(4), FdoPlayer::RIGHT);
        assert_eq!(FdoPlayer::RIGHT.next(5), FdoPlayer::BOTTOM);
    }

}
