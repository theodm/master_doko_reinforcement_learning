use std::error::Error;
use rand::seq::IteratorRandom;
use rand::Rng;
use sled::Db;
use serde::{Serialize, Deserialize};
use bincode;
use std::path::Path;
use serde::de::DeserializeOwned;
use rs_full_doko::state::state::FdoState;

#[derive(Debug)]
pub struct SledStateDb {
    db: Db,
}

impl SledStateDb {
    /// Öffnet eine neue Datenbank am gegebenen Pfad.
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self { db: sled::open(path)? })
    }

    /// Fügt einen neuen Zustand hinzu.
    pub fn insert_state<T: Serialize>(&self, state: &T) -> Result<(), Box<dyn Error>> {
        let id = self.db.generate_id()?; 
        let serialized = bincode::serialize(state)?;
        self.db.insert(id.to_be_bytes(), serialized)?;
        Ok(())
    }

    pub fn merge_from_paths(&self, paths: &[&str]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut merged_count = 0;

        for &path in paths {
            let other_db = sled::open(path)?;
            for item in other_db.iter() {
                let (key, value) = item?;       
                self.db.insert(key, value)?;    
                merged_count += 1;
            }
        }

        self.db.flush()?;
        Ok(merged_count)
    }

    pub fn get_random_states<T: DeserializeOwned>(&self, n: usize) -> Result<Vec<T>, Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        let total_entries = self.db.len();

        if total_entries == 0 {
            return Ok(vec![]); 
        }

        let random_keys: Vec<_> = (0..total_entries)
            .choose_multiple(&mut rng, n)
            .into_iter()
            .map(|id| id.to_be_bytes())
            .collect();

        let random_states = random_keys
            .into_iter()
            .filter_map(|key| self.db.get(&key).ok().flatten())
            .filter_map(|val| bincode::deserialize::<T>(&val).ok())
            .collect();

        Ok(random_states)
    }


    pub fn close(&self) -> Result<(), Box<dyn Error>> {
        self.db.flush()?;
        Ok(())
    }

    pub fn clear(&self) -> Result<(), Box<dyn Error>> {
        self.db.clear()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use super::*;
    use tempfile::TempDir;

    fn sample_state(id: u64) -> FdoState {
        let mut rng = SmallRng::from_os_rng();
        FdoState::new_game(&mut rng)
    }

    #[test]
    fn test_insert_and_get_random_states() {
        let tmp_dir = TempDir::new().unwrap();
        let db = SledStateDb::new(tmp_dir.path().to_str().unwrap()).unwrap();

        db.clear().unwrap();

        let mut states = vec![];

        for i in 0..10 {
            let state1 = sample_state(i);
            db.insert_state(&state1).unwrap();
            states.push(state1);
        }

        let random_states: Vec<FdoState> = db.get_random_states(3).unwrap();
        assert_eq!(random_states.len(), 3);

        for s in &random_states {
            assert!(states.iter().any(|state| *state == *s), "State not found in sample");
        }
    }

    #[test]
    fn test_random_state_diversity() {
        let tmp_dir = TempDir::new().unwrap();
        let db = SledStateDb::new(tmp_dir.path().to_str().unwrap()).unwrap();
        db.clear().unwrap();

        for _ in 0..10 {
            let state = sample_state(0); 
            db.insert_state(&state).unwrap();
        }

        let mut unique_combinations = std::collections::HashSet::new();

        for _ in 0..100 {
            let sample: Vec<FdoState> = db.get_random_states(5).unwrap();
            let mut sample_ids: Vec<String> = sample.iter().map(|s| format!("{:?}", s)).collect();
            sample_ids.sort();
            unique_combinations.insert(sample_ids);
        }

        println!(
            "Anzahl einzigartiger Kombinationen bei 100 Samples: {}",
            unique_combinations.len()
        );

        assert!(unique_combinations.len() > 10, "Zu geringe Diversität in Zufalls-Stichproben!");
    }

    #[test]
    fn test_merge_from_paths() {
        let merged_db = SledStateDb::new("/home/theo/Desktop/played_games_db_comp/merged_games_db")
            .unwrap();

        let sources = [
            "/home/theo/Desktop/played_games_db_comp/played_games_db",
            "/home/theo/Desktop/played_games_db_comp/played_games_db_2",
            "/home/theo/Desktop/played_games_db_comp/played_games_db_3"
        ];

        let count = merged_db.merge_from_paths(&sources).unwrap();
        println!("Insgesamt {} Einträge gemerged.", count);

    }
}
