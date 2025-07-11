
pub mod impi_experience_replay_buffer;
pub mod training;
pub mod train_impi;
pub mod network;
pub mod forward;
pub mod get_weighted_action;
mod tensor_test;
pub mod modified_random_full_doko_policy;
pub mod mcts_full_doko_policy;
pub mod save_log_maybe;
pub mod tensorboard;
pub mod csv_writer_thread;
pub mod next_consistent;

pub mod eval {
    pub mod single_eval_net_fn;
    pub mod eval_net_fn;
    pub mod batch_processor;
    pub mod batch_eval_net_fn;

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}