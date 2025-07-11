use std::marker::PhantomData;
use std::ops::DerefMut;
use rand::prelude::{SliceRandom, SmallRng};
use rand_distr::num_traits::Zero;
use std::sync::PoisonError;
use rand::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use rs_doko_networks::full_doko::ipi_network::{FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};

#[derive(Serialize, Deserialize)]
struct DBRecord {
    state: heapless::Vec<i64, { FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE }>,
    result: heapless::Vec<f32, { FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE }>,
}


pub(crate) struct ImpiExperienceReplayBuffer {
    pub db: sled::Db
}


impl ImpiExperienceReplayBuffer {
    pub(crate) fn new() -> Self {
        let db = sled::open("./temporaryimpi.sled")
            .expect("Could not open Sled DB");

        db.clear().unwrap();

        ImpiExperienceReplayBuffer {
            db
        }
    }

    pub fn append_single(
        &self,

        state: [i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE],
        result: [f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE],
    ) {
        self.append_slice(&[state], &[result]);
    }

    pub fn append_slice(
        &self,

        state: &[[i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]],
        result: &[[f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]],
    ) {
        // println!("state.len(): {}", state.len());

        let mut rng = SmallRng::from_os_rng();

        for i in 0..state.len() {
            let record = DBRecord {
                state: heapless::Vec::from_slice(&state[i]).unwrap(),
                result: heapless::Vec::from_slice(&result[i]).unwrap(),
            };

            let serialized_data = bincode::serialize(&record)
                .expect("Failed to serialize DBRecord");

            let key = rng.next_u64().to_be_bytes();

            self
                .db
                .insert(key, serialized_data)
                .unwrap();
        }
    }

    pub fn clear(&self) {
        println!("Clearing database");
        self.db.clear().unwrap()
    }

    pub fn load_clear(
        &self,
        rng: &mut rand::rngs::SmallRng,
    ) -> (Vec<[i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]>,
        Vec<[f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]>) {
        let mut state_memory: Vec<[i64; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE]> = Vec::new();
        let mut result_memory: Vec<[f32; FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE]> = Vec::new();

        self.db.iter()
            .map(|x| x.expect("Failed to read from Sled"))
            .for_each(|(key, value)| {
                let record: DBRecord = bincode::deserialize(&value)
                    .expect("Failed to deserialize DBRecord");

                let (state, result) = (
                    record.state,
                    record.result,
                );

                state_memory.push(state.as_slice().try_into().unwrap());
                result_memory.push(result.as_slice().try_into().unwrap());
            });

        let mut indices: Vec<usize> = (0..state_memory.len()).collect();
        indices.shuffle(rng);

        state_memory = indices.iter().map(|i| state_memory[*i]).collect();
        result_memory = indices.iter().map(|i| result_memory[*i]).collect();

        self.clear();

        (state_memory, result_memory)
    }
}
