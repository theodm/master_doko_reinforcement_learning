use std::sync::atomic::AtomicUsize;
pub struct GlobalStats {
    pub(crate) number_of_experiences_added_in_epoch: AtomicUsize,
    pub(crate) number_of_cache_misses: AtomicUsize,
    pub(crate) number_of_cache_hits: AtomicUsize,
    pub(crate) time_of_batch_processor: AtomicUsize,
    pub(crate) number_of_batch_processor_hits: AtomicUsize,

    pub number_of_turns_in_epoch: AtomicUsize,
    pub number_of_batch_processed_entries: AtomicUsize,
}

impl GlobalStats {
    pub fn new() -> Self {
        Self {
            number_of_experiences_added_in_epoch: AtomicUsize::new(0),
            number_of_cache_misses: AtomicUsize::new(0),
            number_of_cache_hits: AtomicUsize::new(0),
            time_of_batch_processor: AtomicUsize::new(0),
            number_of_batch_processor_hits: AtomicUsize::new(0),

            number_of_turns_in_epoch: AtomicUsize::new(0),
            number_of_batch_processed_entries: AtomicUsize::new(0),
        }
    }
}