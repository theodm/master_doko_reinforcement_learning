use std::fmt::Debug;
use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;
use fxhash::FxHasher64;
use quick_cache::sync::Cache;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
use crate::env::env_state::AzEnvState;
use bytemuck::{cast_slice, Pod};
use crate::alpha_zero::train::glob;
use crate::alpha_zero::train::glob::GlobalStats;

unsafe impl <
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> Send for CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions> {

}

unsafe impl <
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> Sync for CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions> {

}

fn hash_array<T: Pod, const N: usize>(array: &[T; N]) -> u64 {
    let mut hasher = DefaultHasher::default();
    // Cast `[T]` → `[u8]`
    let as_bytes: &[u8] = cast_slice(array);
    // Diese Bytes am Stück in den Hasher
    hasher.write(as_bytes);
    hasher.finish()
}

#[derive(Clone)]
// Ein NetworkBatchProcessor, der die Ergebnisse cacht. Im Rahmen des AlphaZero-Algorithmus
// wird das Netzwerk (in der Self-Play) sehr oft mit denselben Daten aufgerufen. Daher
// macht es Sinn, die Ergebnisse zu cachen.
pub struct CachedNetworkBatchProcessorSender<
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> {
    pub batch_processor: BatchProcessorSender<[i64; TStateDim], ([f32; TNumberOfPlayers], [f32; TNumberOfActions])>,
    pub cache: Arc<Cache<u64, ([f32; TNumberOfPlayers], [f32; TNumberOfActions])>>,
    pub glob: Arc<GlobalStats>,

    // Aus merkwürdigen Gründen will der Compiler diese Marker.
    pub _marker: std::marker::PhantomData<TActionType>,
    pub _marker2: std::marker::PhantomData<TGameState>,
}

impl <
    TGameState: AzEnvState<TActionType, TNumberOfPlayers, TNumberOfActions>,
    TActionType: Copy + Send + Debug,
    const TStateDim: usize,
    const TNumberOfPlayers: usize,
    const TNumberOfActions: usize
> CachedNetworkBatchProcessorSender<TGameState, TActionType, TStateDim, TNumberOfPlayers, TNumberOfActions> {
    pub fn new(
        batch_processor: BatchProcessorSender<[i64; TStateDim], ([f32; TNumberOfPlayers], [f32; TNumberOfActions])>,
        batch_size: usize,
        glob_stats: Arc<GlobalStats>
    ) -> Self {
        CachedNetworkBatchProcessorSender {
            batch_processor,
            cache: Arc::new(Cache::new(batch_size)),
            glob: glob_stats,

            _marker: std::marker::PhantomData,
            _marker2: std::marker::PhantomData,
        }
    }

    pub async fn process(&self, state: TGameState) -> ([f32; TNumberOfPlayers], [f32; TNumberOfActions]) {
        let mut encoded = [0; TStateDim];
        state.encode_into_memory(&mut encoded);

        let hash_key = hash_array(&encoded);

        if let Some(result) = self.cache.get(&hash_key) {
            self.glob.number_of_cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            return result;
        }

        self.glob.number_of_cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let result = self
            .batch_processor
            .process(encoded)
            .await;

        self.cache.insert(hash_key, result);

        result
    }
}
