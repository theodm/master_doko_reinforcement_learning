use std::fmt::Debug;
use std::ops::Index;
use serde::{Deserialize, Serialize};
use crate::card::cards::FdoCard;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation::FdoReservation;
use crate::util::po_arr::PlayerOrientedArr;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerOrientedVec<T: Debug> {
    pub storage: heapless::Vec<T, 4>,

    pub starting_player: FdoPlayer
}

impl<T: Debug> PlayerOrientedVec<T> {
    pub fn empty(starting_player: FdoPlayer) -> PlayerOrientedVec<T> {
        PlayerOrientedVec {
            storage: heapless::Vec::new(),
            starting_player
        }
    }

    pub fn from_full(
        starting_player: FdoPlayer,
        cards: Vec<T>
    ) -> PlayerOrientedVec<T> {
        let mut storage = heapless::Vec::new();

        for card in cards {
            storage
                .push(card)
                .unwrap();
        }

        PlayerOrientedVec {
            storage,
            starting_player
        }
    }

    pub fn push(&mut self, value: T) {
        self
            .storage
            .push(value)
            .unwrap();
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn next_empty_player(&self) -> Option<FdoPlayer> {
        if self.storage.len() == 4 {
            return None;
        }

        return Some(self.starting_player.next(self.storage.len()));
    }

    pub fn is_full(&self) -> bool {
        self.storage.len() == 4
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.storage.iter()
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


    pub fn get(&self, player: FdoPlayer) -> Option<&T> {
        self.storage.get(index_for_i(self.starting_player.index(), player.index(), 4))
    }

    pub fn get_raw(&self, index: usize) -> Option<&T> {
        self.storage.get(index)
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

impl<T: Debug> Index<FdoPlayer> for PlayerOrientedVec<T> {
    type Output = T;

    fn index(&self, index: FdoPlayer) -> &Self::Output {
        &self.storage[index_for_i(self.starting_player.index(), index.index(), 4)]
    }
}

impl<T: Debug + Send + Copy + Clone> PlayerOrientedVec<T> {
    pub fn to_array(
        &self
    ) -> PlayerOrientedArr<T> {
        PlayerOrientedArr::from_full(
            self.starting_player,
            self.storage.as_slice().try_into().unwrap()
        )
    }

    pub fn to_array_remaining_option(
        &self
    ) -> PlayerOrientedArr<Option<T>> {
        let slice = self
            .storage
            .as_slice();

        // fill up to 4 with None
        let mut x: [Option<T>; 4] = [None; 4];

        for i in 0..slice.len() {
            x[i] = Some(slice[i]);
        }

        PlayerOrientedArr::from_full(
            self.starting_player,
            x.try_into().unwrap()
        )
    }

    pub fn to_zero_array(&self) -> PlayerZeroOrientedArr<T> {
        let x: [T; 4] = FdoPlayerSet::all()
            .iter()
            .map(|player| {
                self[player]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        PlayerZeroOrientedArr::from_full(x)
    }

    pub fn to_zero_array_remaining_option(&self) -> PlayerZeroOrientedArr<Option<T>> {
        let x: [Option<T>; 4] = FdoPlayerSet::all()
            .iter()
            .map(|player| {
                self.get(player).copied()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        PlayerZeroOrientedArr::from_full(x)
    }

    pub fn map<U: Debug, F: FnMut(&T) -> U>(&self, mut f: F) -> PlayerOrientedVec<U> {
        let mut new_storage = heapless::Vec::new();

        for value in self.storage.iter() {
            new_storage
                .push(f(value))
                .unwrap();
        }

        PlayerOrientedVec {
            storage: new_storage,
            starting_player: self.starting_player
        }
    }


}

#[cfg(test)]
mod tests {
    use crate::card::cards::FdoCard::{ClubJack, DiamondAce, HeartQueen, SpadeTen};
    use super::*;

    #[test]
    fn test_empty() {
        let pov_vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::empty(FdoPlayer::BOTTOM);

        assert_eq!(pov_vec.len(), 0);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::BOTTOM));

        let pov_vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::empty(FdoPlayer::LEFT);

        assert_eq!(pov_vec.len(), 0);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::LEFT));

    }

    /// Testet, ob `push`, `len`, `next_empty_player` und `is_full` korrekt arbeiten.
    #[test]
    fn test_push_and_len() {
        let starting_player = FdoPlayer::TOP;
        let mut pov_vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::empty(starting_player);
        assert_eq!(pov_vec.len(), 0);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::TOP));

        pov_vec.push(FdoCard::ClubQueen);
        assert_eq!(pov_vec.len(), 1);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::RIGHT));

        pov_vec.push(FdoCard::ClubQueen);
        assert_eq!(pov_vec.len(), 2);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::BOTTOM));

        pov_vec.push(FdoCard::ClubAce);
        assert_eq!(pov_vec.len(), 3);
        assert_eq!(pov_vec.next_empty_player(), Some(FdoPlayer::LEFT));

        pov_vec.push(FdoCard::ClubJack);
        assert_eq!(pov_vec.len(), 4);
        assert_eq!(pov_vec.next_empty_player(), None);
        assert!(pov_vec.is_full());
    }

    /// Testet den Iterator, der jeweils das Element und den zugehörigen Spieler liefert.
    #[test]
    fn test_iter_with_player() {
        let pov_vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::ClubQueen,
            FdoCard::ClubKing,
            FdoCard::ClubAce
        ]);

        let mut iter = pov_vec
            .iter_with_player();

        assert_eq!(iter.next(), Some((FdoPlayer::RIGHT, &FdoCard::ClubJack)));
        assert_eq!(iter.next(), Some((FdoPlayer::BOTTOM, &FdoCard::ClubQueen)));
        assert_eq!(iter.next(), Some((FdoPlayer::LEFT, &FdoCard::ClubKing)));
        assert_eq!(iter.next(), Some((FdoPlayer::TOP, &FdoCard::ClubAce)));
    }

    /// Testet die Hilfsmethode `index_for_i` mit verschiedenen Parametern.
    #[test]
    fn test_index_for_i() {
        assert_eq!(index_for_i(0, 0, 4), 0);
        assert_eq!(index_for_i(0, 1, 4), 1);
        assert_eq!(index_for_i(0, 2, 4), 2);
        assert_eq!(index_for_i(0, 3, 4), 3);

        assert_eq!(index_for_i(1, 0, 4), 3);
        assert_eq!(index_for_i(1, 1, 4), 0);
        assert_eq!(index_for_i(1, 2, 4), 1);
        assert_eq!(index_for_i(1, 3, 4), 2);

        assert_eq!(index_for_i(2, 0, 4), 2);
        assert_eq!(index_for_i(2, 1, 4), 3);
        assert_eq!(index_for_i(2, 2, 4), 0);
        assert_eq!(index_for_i(2, 3, 4), 1);

        assert_eq!(index_for_i(3, 0, 4), 1);
        assert_eq!(index_for_i(3, 1, 4), 2);
        assert_eq!(index_for_i(3, 2, 4), 3);
        assert_eq!(index_for_i(3, 3, 4), 0);
    }

    /// Testet die Indexierung mittels FdoPlayer.
    #[test]
    fn test_indexing_by_player() {
        let pov_vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::DiamondAce,
            FdoCard::HeartQueen,
            FdoCard::SpadeTen
        ]);

        assert_eq!(pov_vec[FdoPlayer::TOP], SpadeTen);
        assert_eq!(pov_vec[FdoPlayer::RIGHT], ClubJack);
        assert_eq!(pov_vec[FdoPlayer::BOTTOM], DiamondAce);
        assert_eq!(pov_vec[FdoPlayer::LEFT], HeartQueen);
    }
    #[test]
    fn to_zero_array() {
        let vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::DiamondAce,
            FdoCard::HeartQueen,
            FdoCard::SpadeTen,
        ]);

        let zero_arr = vec.to_zero_array();

        assert_eq!(zero_arr[FdoPlayer::BOTTOM], DiamondAce);
        assert_eq!(zero_arr[FdoPlayer::LEFT], HeartQueen);
        assert_eq!(zero_arr[FdoPlayer::TOP], SpadeTen);
        assert_eq!(zero_arr[FdoPlayer::RIGHT], ClubJack);
    }
    #[test]
    fn to_zero_array_remaining_option() {
        let vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
        ]);

        let zero_arr = vec.to_zero_array_remaining_option();

        assert_eq!(zero_arr[FdoPlayer::BOTTOM], None);
        assert_eq!(zero_arr[FdoPlayer::LEFT], None);
        assert_eq!(zero_arr[FdoPlayer::TOP], None);
        assert_eq!(zero_arr[FdoPlayer::RIGHT], Some(ClubJack));

        let vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::DiamondAce,
        ]);

        let zero_arr = vec.to_zero_array_remaining_option();

        assert_eq!(zero_arr[FdoPlayer::BOTTOM], Some(DiamondAce));
        assert_eq!(zero_arr[FdoPlayer::LEFT], None);
        assert_eq!(zero_arr[FdoPlayer::TOP], None);
        assert_eq!(zero_arr[FdoPlayer::RIGHT], Some(ClubJack));

        let vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::DiamondAce,
            FdoCard::HeartQueen,
        ]);

        let zero_arr = vec.to_zero_array_remaining_option();

        assert_eq!(zero_arr[FdoPlayer::BOTTOM], Some(DiamondAce));
        assert_eq!(zero_arr[FdoPlayer::LEFT], Some(HeartQueen));
        assert_eq!(zero_arr[FdoPlayer::TOP], None);
        assert_eq!(zero_arr[FdoPlayer::RIGHT], Some(ClubJack));

        let vec: PlayerOrientedVec<FdoCard> = PlayerOrientedVec::from_full(FdoPlayer::RIGHT, vec![
            FdoCard::ClubJack,
            FdoCard::DiamondAce,
            FdoCard::HeartQueen,
            FdoCard::SpadeTen,
        ]);

        let zero_arr = vec.to_zero_array_remaining_option();

        assert_eq!(zero_arr[FdoPlayer::BOTTOM], Some(DiamondAce));
        assert_eq!(zero_arr[FdoPlayer::LEFT], Some(HeartQueen));
        assert_eq!(zero_arr[FdoPlayer::TOP], Some(SpadeTen));
        assert_eq!(zero_arr[FdoPlayer::RIGHT], Some(ClubJack));
    }


}

