pub mod doko {
    pub mod policy {
        pub mod policy;
        pub mod random_policy;
    }

    pub mod evaluate_single_game;
}


pub mod full_doko {
    pub mod evaluate_single_game;

    pub mod policy {
        pub mod policy;
        pub mod random_policy;
        pub mod mcts_policy;
        pub mod az_policy;
    }

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}