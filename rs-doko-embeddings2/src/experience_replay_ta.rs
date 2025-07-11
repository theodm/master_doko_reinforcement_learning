
pub(crate) struct ExperienceReplayWithThrowAway {
    pub states_memory: Vec<i64>,
    pub value_targets_memory: Vec<f32>,
    pub policy_targets_memory: Vec<f32>,

    pub(crate) max_memory: usize,

    pub current_index: usize,
    pub(crate) memory_full: bool,

    pub(crate) minibatch_size: usize,
    pub(crate) n_minibatches: usize,

    pub state_dim: usize,
    pub value_dim: usize,
    pub policy_dim: usize,


}

impl ExperienceReplayWithThrowAway {
    pub fn new(
        state_dim: usize,
        value_dim: usize,
        policy_dim: usize,

        minibatch_size: usize,
        n_minibatches: usize
    ) -> Self {
        let max_memory = minibatch_size * n_minibatches;
        Self {
            states_memory: vec![0; state_dim * max_memory],
            value_targets_memory: vec![0.0; value_dim * max_memory],
            policy_targets_memory: vec![0.0; policy_dim * max_memory],

            max_memory: max_memory,

            current_index: 0,
            memory_full: false,

            minibatch_size,
            n_minibatches,

            state_dim,
            value_dim,
            policy_dim
        }
    }

    pub(crate) fn append(
        &mut self,

        state_memory: &[i64],
        value_target_memory: &[f32],
        policy_target_memory: &[f32],
    ) {
        assert!(!self.memory_full, "Memory is full");

        let i = self
            .current_index;

        self.states_memory[i * self.state_dim..(i + 1) * self.state_dim]
            .copy_from_slice(state_memory);
        self.value_targets_memory[i * self.value_dim..(i + 1) * self.value_dim]
            .copy_from_slice(value_target_memory);
        self.policy_targets_memory[i * self.policy_dim..(i + 1) * self.policy_dim]
            .copy_from_slice(policy_target_memory);

        self.current_index = (self.current_index + 1) % self.max_memory;

        if self.current_index == 0 {
            self.memory_full = true;
        }
    }

    pub fn len(&self) -> usize {
        self.states_memory.len()
    }
}