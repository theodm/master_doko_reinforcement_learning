pub mod env {
    pub mod env_state;

    pub mod envs {
        pub mod env_state_doko;
        pub mod env_state_full_doko;
    }
}
pub mod example {
    pub mod tic_tac_toe;
    pub mod tic_tac_toe_impl;
    pub mod tic_tac_toe_player;
}

pub mod mcts {
    pub mod node;
    pub mod mcts;
}

pub mod save_log_maybe;