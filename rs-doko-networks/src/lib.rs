
#![recursion_limit = "1024"]

pub mod doko {
    pub mod var1 {
        pub mod encoder;
        pub mod encode_for_impi_gan;
    }

}

pub mod full_doko {
    pub mod var1 {
        pub mod ipi_output;
        pub mod encode_pi;
        pub mod encode_ipi;
        pub mod phase;
        pub mod player;
        pub mod reservation;
        pub mod card;
        pub mod announcement;

        pub mod trick_method2;

        pub mod game_type;
        pub mod visible_reservation;
        pub mod bool;
    }

    pub mod var2 {
        pub mod encode_reservation_or_card_or_none;

        pub mod encode_position_or_unknown;
        pub mod encode_subposition;
        pub mod encode_annoucnement_team;
        pub mod encode_action;
    }
    pub mod pi_network;
    pub mod ipi_network;
    mod utils;
    mod res_block;
}