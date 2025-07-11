pub mod alpha_zero {
    pub mod batch_processor {
        pub mod batch_processor;
        pub mod cached_network_batch_processor;
        pub mod network_batch_processor;
    }

    pub mod eval {
        pub mod evaluate;
    }

    pub mod mcts {
        pub mod node;
        pub mod mcts;

        pub mod async_mcts;
    }

    pub mod net {
        pub mod experience_replay_buffer3;

        pub mod network;
    }

    pub mod shared {
        pub mod net_eval;
        pub mod single_net_eval;
    }

    pub mod train {
        pub mod train_loop;
        pub mod self_play;
        pub mod glob;
    }

    pub mod utils {
        pub mod float_array_utils;
        pub mod float_vec_utils;
        pub mod pretty_print_utils;
        pub mod vec_utils;
        pub mod rot_arr;

    }

    pub mod net_trainer {
        pub mod net_trainer;
        pub mod train_network;
    }

    pub mod tensorboard {
        pub mod tensorboard_sender;
    }

    pub mod alpha_zero_evaluation_options;
    pub mod alpha_zero_train_options;

    pub mod alpha_zero_train;

    pub mod alpha_zero_evaluate;
}

pub mod env {
    pub mod env_state;

    pub mod envs {
        pub mod full_doko {
            pub mod full_doko;
        }

        pub mod tic_tac_toe {
            pub mod tic_tac_toe_impl;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}