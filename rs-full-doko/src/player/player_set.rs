use std::fmt::{Debug};
use enumset::EnumSet;
use serde::{Deserialize, Serialize};
use crate::player::player::FdoPlayer;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FdoPlayerSet(
    EnumSet<FdoPlayer>
);


impl FdoPlayerSet {
    pub fn all() -> Self {
        FdoPlayerSet(EnumSet::all())
    }

    /// Erstellt eine leere Menge von Spielern.
    pub fn empty() -> Self {
        FdoPlayerSet(EnumSet::empty())
    }

    /// Fügt einen Spieler zur Menge hinzu.
    pub fn insert(&mut self, player: FdoPlayer) {
        self.0.insert(player);
    }

    /// Prüft, ob ein Spieler in der Menge enthalten ist.
    pub fn contains(&self, player: FdoPlayer) -> bool {
        self.0.contains(player)
    }

    /// Entfernt einen Spieler aus der Menge.
    pub fn remove(&mut self, player: FdoPlayer) {
        self.0.remove(player);
    }

    /// Gibt die Anzahl der Spieler in der Menge zurück.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Gibt die inverse Menge von Spielern zurück (alle Spieler, die nicht in der aktuellen Menge sind).
    pub fn complement(&self) -> Self {
        FdoPlayerSet(self.0.complement())
    }

    pub fn iter(
        &self
    ) -> impl Iterator<Item = FdoPlayer> {
        self.0.iter()
    }

    /// Erstellt eine Menge von Spielern aus einem Vektor von Spielern.
    pub fn from_vec(players: Vec<FdoPlayer>) -> Self {
        let mut player_set = FdoPlayerSet::empty();

        for player in players {
            player_set.insert(player);
        }
        player_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_contains() {
        let mut player_set = FdoPlayerSet::empty();
        let player = FdoPlayer::BOTTOM;

        player_set.insert(player);
        assert!(player_set.contains(player));
    }

    #[test]
    fn test_remove() {
        let mut player_set = FdoPlayerSet::empty();
        let player = FdoPlayer::BOTTOM;

        player_set.insert(player);
        player_set.remove(player);
        assert!(!player_set.contains(player));
    }

    #[test]
    fn test_len() {
        let mut player_set = FdoPlayerSet::empty();
        player_set.insert(FdoPlayer::BOTTOM);
        player_set.insert(FdoPlayer::LEFT);

        assert_eq!(player_set.len(), 2);
    }

    #[test]
    fn test_inverse() {
        let mut player_set = FdoPlayerSet::empty();
        player_set.insert(FdoPlayer::BOTTOM);
        player_set.insert(FdoPlayer::LEFT);

        println!("{:?}", player_set);
        let inverse = player_set.complement();

        println!("{:?}", inverse);

        assert!(!inverse.contains(FdoPlayer::BOTTOM));
        assert!(!inverse.contains(FdoPlayer::LEFT));
        assert!(inverse.contains(FdoPlayer::TOP));
        assert!(inverse.contains(FdoPlayer::RIGHT));
    }

    #[test]
    fn test_to_vec() {
        let mut player_set = FdoPlayerSet::empty();
        player_set.insert(FdoPlayer::BOTTOM);
        player_set.insert(FdoPlayer::LEFT);

        let vec = player_set.iter().collect::<Vec<_>>();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&FdoPlayer::BOTTOM));
        assert!(vec.contains(&FdoPlayer::LEFT));
    }

    #[test]
    fn test_from_vec() {
        let players = vec![FdoPlayer::BOTTOM, FdoPlayer::LEFT];
        let player_set = FdoPlayerSet::from_vec(players);

        assert!(player_set.contains(FdoPlayer::BOTTOM));
        assert!(player_set.contains(FdoPlayer::LEFT));
    }
}
