use std::time::Duration;

#[derive(Clone, Debug)]
pub struct AlphaZeroEvaluationOptions {
    pub iterations: usize,

    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,

    pub puct_exploration_constant: f32,

    pub min_or_max_value: f32,
    pub par_iterations: usize,
    pub virtual_loss: f32
}


