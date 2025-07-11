use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use crate::player::player::FdoPlayer;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerOrientedArr<T: Debug> {
    pub storage: [T; 4],

    pub starting_player: FdoPlayer
}


impl<T: Debug> PlayerOrientedArr<T> {
    pub fn from_full(
        starting_player: FdoPlayer,
        storage: [T; 4]
    ) -> PlayerOrientedArr<T> {
        PlayerOrientedArr {
            storage,
            starting_player
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
                (self.starting_player.next(index), value)
            })
    }

    pub fn iter_with_player_and_index(&self) -> impl Iterator<Item = (usize, FdoPlayer,  &T)> {
        self
            .storage
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (index, self.starting_player.next(index), value)
            })
    }

    pub fn get_raw(&self, index: usize) -> &T {
        &self
            .storage[index]
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

impl<T: Debug> Index<FdoPlayer> for PlayerOrientedArr<T> {
    type Output = T;

    fn index(&self, index: FdoPlayer) -> &Self::Output {
        &self.storage[index_for_i(self.starting_player.index(), index.index(), 4)]
    }
}

impl <T: Debug> IndexMut<FdoPlayer> for PlayerOrientedArr<T> {
    fn index_mut(&mut self, index: FdoPlayer) -> &mut Self::Output {
        &mut self.storage[index_for_i(self.starting_player.index(), index.index(), 4)]
    }
}

impl<T: Send + Debug> PlayerOrientedArr<Option<T>> {
    pub fn all_present(&self) -> impl Iterator<Item = &T> {
        self.iter().filter_map(|opt| opt.as_ref())
    }
}