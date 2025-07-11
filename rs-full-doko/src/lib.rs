pub mod card {
    pub mod cards;
    pub mod card_to_eyes;
    pub mod card_to_color;
    pub mod card_in_trick_logic;
    pub mod card_color_masks;
}

pub mod matching {
    pub mod card_matching;

    pub mod is_consistent;
    pub mod gather_impossible_colors;

}

pub mod announcement {
    pub mod announcement;
    pub mod announcement_set;
    pub mod calc_announcement;
    pub mod announcement_round;
}
pub mod stats {
    pub mod stats;
    pub mod basic_points {
        pub mod basic_draw_points;
        pub mod basic_winning_points;
    }
    pub mod win_conditions {
        pub mod re_won;
        pub mod kontra_won;
    }
    pub mod additional_points {
        pub mod additional_points;
        pub mod fuchs_gefangen;
        pub mod doppelkopf;
        pub mod last_trick_karlchen;
    }
}
pub mod action {
    pub mod action;
    pub mod allowed_actions;
}
pub mod game_type {
    pub mod game_type;
}

pub mod basic {
    pub mod color;
    pub mod phase;
    pub mod team;
}

pub mod state {
    pub mod state;
}

pub mod hand {
    pub mod hand;

    pub mod hand_iter;
}

pub mod trick {
    pub mod trick;
    pub mod trick_winning_player_logic;
}

pub mod player {
    pub mod player;
    pub mod player_set;
}

pub mod reservation {
    pub mod reservation;
    pub mod reservation_allowed_action_logic;
    pub mod reservation_round;
    pub mod reservation_winning_logic;
    pub mod visible_reservations_logic;

}

pub mod team {
    pub mod team_logic;
}

pub mod observation {
    pub mod observation;
}

pub mod display {
    pub mod display;
}

pub mod util {
    pub mod rot_arr;

    pub mod po_vec;

    pub mod po_arr;

    pub mod po_zero_arr;
}