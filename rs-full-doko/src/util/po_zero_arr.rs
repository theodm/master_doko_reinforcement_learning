use std::fmt::{Debug, Display, Formatter};
use std::ops::{Index, IndexMut};
use serde::{Deserialize, Serialize};
use crate::player::player::FdoPlayer;
use crate::util::po_arr::PlayerOrientedArr;

#[derive(Debug,  Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerZeroOrientedArr<T: Debug + Send> {
    pub storage: [T; 4]
}

impl<T: Send + Debug> PlayerZeroOrientedArr<T> {
    pub fn from_full(
        storage: [T; 4]
    ) -> PlayerZeroOrientedArr<T> {
        PlayerZeroOrientedArr {
            storage
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self
            .storage
            .iter()
    }

    pub fn iter_with_player(&self) -> impl Iterator<Item = (FdoPlayer, &T)> {
        self
            .storage
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (FdoPlayer::BOTTOM.next(index), value)
            })
    }

    pub fn map<U: Send + Debug, F: Fn(&T) -> U>(
        &self,
        f: F
    ) -> PlayerZeroOrientedArr<U> {
        PlayerZeroOrientedArr::from_full(
            [
                f(&self[FdoPlayer::BOTTOM]),
                f(&self[FdoPlayer::LEFT]),
                f(&self[FdoPlayer::TOP]),
                f(&self[FdoPlayer::RIGHT])
            ]
        )
    }

    pub fn to_oriented_arr(
        self
    ) -> PlayerOrientedArr<T> {
        PlayerOrientedArr::from_full(
            FdoPlayer::BOTTOM,
            self.storage
        )
    }

}

impl<T: Send + Debug + Clone> PlayerZeroOrientedArr<T> {

    pub fn rotate_to(
        self,
        pov: FdoPlayer
    ) -> PlayerOrientedArr<T> {
        PlayerOrientedArr::from_full(
            pov,
            [
                self[pov].clone(),
                self[pov + 1].clone(),
                self[pov + 2].clone(),
                self[pov + 3].clone()
            ]
        )
    }
}

impl<T: Display + Debug + Send> Display for PlayerZeroOrientedArr<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unten: {}\nLinks: {}\nOben:{}\nRechts: {})", self.storage[0], self.storage[1], self.storage[2], self.storage[3])
    }
}


/// Berechnet den Index, der i in der Perspektive pov_i
/// hat.
pub fn index_for_i(
    pov_i: usize,
    i: usize,
    N: usize
) -> usize {
    ((N - pov_i) + i) % N
}

impl<T: Send + Debug> Index<FdoPlayer> for PlayerZeroOrientedArr<T> {
    type Output = T;

    fn index(&self, index: FdoPlayer) -> &Self::Output {
        &self.storage[index_for_i(FdoPlayer::BOTTOM.index(), index.index(), 4)]
    }
}

impl <T: Send + Debug> IndexMut<FdoPlayer> for PlayerZeroOrientedArr<T> {
    fn index_mut(&mut self, index: FdoPlayer) -> &mut Self::Output {
        &mut self.storage[index_for_i(FdoPlayer::BOTTOM.index(), index.index(), 4)]
    }
}
