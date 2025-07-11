use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTarget {
    Default,
    Greedy,
    Avg
}

#[derive(Debug, Clone, Copy)]
pub struct AlphaZeroTrainOptions {
    pub number_of_batch_receivers: usize,
    pub batch_nn_max_delay: Duration,
    pub batch_nn_buffer_size: usize,

    pub checkpoint_every_n_epochs: usize,
    pub probability_of_keeping_experience: f32,

    pub games_per_epoch: usize,
    pub max_concurrent_games: usize,

    pub mcts_iterations: usize,

    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,

    pub puct_exploration_constant: f32,

    pub temperature: Option<f32>,
    // ToDo: Das muss ich nochmal besser machen
    pub min_or_max_value: f32,

    pub value_target: ValueTarget,

    pub node_arena_capacity: usize,
    pub state_arena_capacity: usize,
    pub cache_size_batch_processor: usize,

    pub mcts_workers: usize,
    pub max_concurrent_games_in_evaluation: usize,

    pub epoch_start: Option<usize>,
    pub evaluation_every_n_epochs: usize,
    pub skip_evaluation: bool
}
