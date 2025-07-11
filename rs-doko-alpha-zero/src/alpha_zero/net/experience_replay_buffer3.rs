use std::marker::PhantomData;
use std::ops::DerefMut;
use rand::prelude::{SliceRandom, SmallRng};
use rand_distr::num_traits::Zero;
use std::sync::PoisonError;
use bytemuck::Pod;
use rand::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

#[derive(Serialize, Deserialize)]
struct DBRecord<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    state: heapless::Vec<i64, TStateDim>,
    value: heapless::Vec<f32, TNumberOfPlayers>,
    policy: heapless::Vec<f32, TNumberOfActions>,
}

unsafe impl<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>
Send for ExperienceReplayBuffer3<
    TStateDim,
    TNumberOfPlayers,
    TNumberOfActions,
> {}

unsafe impl<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
>
Sync for ExperienceReplayBuffer3<
    TStateDim,
    TNumberOfPlayers,
    TNumberOfActions,
> {}

pub(crate) struct ExperienceReplayBuffer3<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> {
    pub db: sled::Db,

    pub(crate) state_dim: usize,
    pub(crate) value_dim: usize,
    pub(crate) policy_dim: usize,

}


impl<
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize,
> ExperienceReplayBuffer3<
    TStateDim,
    TNumberOfPlayers,
    TNumberOfActions,
> {
    pub(crate) fn new(
        // Anzahl der Einträge, die der State des Modells ausgibt.
        state_dim: usize,

        // Anzahl der Werte, die der Value-Head des Modells ausgibt,
        // in der Regel soviele wie es Spieler gibt.
        value_dim: usize,

        // Anzahl der Werte, die der Policy-Head des Modells ausgibt,
        // in der Regel soviele wie es unterscheidbare Aktionen gibt.
        policy_dim: usize
    ) -> Self {
        let db = sled::open("./temporary.sled")
            .expect("Could not open Sled DB");

        db.clear().unwrap();

        let result = ExperienceReplayBuffer3::<TStateDim, TNumberOfPlayers, TNumberOfActions> {
            db,
            state_dim,
            value_dim,
            policy_dim
        };

        return result;
    }

    pub fn append_slice(
        &self,

        state: &[[i64; TStateDim]],
        value: &[[f32; TNumberOfPlayers]],
        policy: &[[f32; TNumberOfActions]],
    ) {
        // println!("state.len(): {}", state.len());

        let mut rng = SmallRng::from_os_rng();

        for i in 0..state.len() {
            let record = DBRecord {
                state: heapless::Vec::<i64, TStateDim>::from_slice(&state[i]).unwrap(),
                value: heapless::Vec::<f32, TNumberOfPlayers>::from_slice(&value[i]).unwrap(),
                policy: heapless::Vec::<f32, TNumberOfActions>::from_slice(&policy[i]).unwrap(),
            };

            let serialized_data = bincode::serialize(&record)
                .expect("Failed to serialize DBRecord");

            // 4. Generate a random key and insert + timestamp (unix)
            let key = rng.next_u64().to_be_bytes();

            self
                .db
                .insert(key, serialized_data)
                .unwrap();
        }

        // self
        //     .db
        //     .flush()
        //     .expect("Failed to flush Sled");
    }

    pub fn clear(&self) {
        // Clearing database
        println!("Clearing database");
        self.db.clear().unwrap()
    }

    pub fn load(
        &self,
        rng: &mut rand::rngs::SmallRng,
    ) -> (Vec<[i64; TStateDim]>, Vec<[f32; TNumberOfPlayers]>, Vec<[f32; TNumberOfActions]>) {
        let mut state_memory: Vec<[i64; TStateDim]> = Vec::new();
        let mut value_memory: Vec<[f32; TNumberOfPlayers]> = Vec::new();
        let mut policy_memory: Vec<[f32; TNumberOfActions]> = Vec::new();

        self.db.iter()
            .map(|x| x.expect("Failed to read from Sled"))
            .for_each(|(key, value)| {
                let record: DBRecord<TStateDim, TNumberOfPlayers, TNumberOfActions> = bincode::deserialize(&value)
                    .expect("Failed to deserialize DBRecord");

                let (state, value, policy) = (
                    record.state,
                    record.value,
                    record.policy,
                );

                state_memory.push(state.as_slice().try_into().unwrap());
                value_memory.push(value.as_slice().try_into().unwrap());
                policy_memory.push(policy.as_slice().try_into().unwrap());
            });


        let mut indices: Vec<usize> = (0..state_memory.len()).collect();
        indices.shuffle(rng);

        state_memory = indices.iter().map(|i| state_memory[*i]).collect();
        value_memory = indices.iter().map(|i| value_memory[*i]).collect();
        policy_memory = indices.iter().map(|i| policy_memory[*i]).collect();

        return (state_memory, value_memory, policy_memory);
    }
}
