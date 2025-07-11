use std::fmt::{Display, Formatter};
use rand::prelude::SmallRng;
use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::{allowed_actions_to_vec};
use rs_doko::basic::phase::DoPhase;
use rs_doko::observation::observation::DoObservation;
use rs_doko::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
use rs_doko::player::player_set::player_set_contains;
use rs_doko::state::state::DoState;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::action::allowed_actions::FdoAllowedActions;
use crate::env::env_state::McEnvState;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct McDokoEnvState {
    pub doko: DoState,
    pub observation: DoObservation,

    pub last_played_action: Option<DoAction>
}

impl McDokoEnvState {
    pub fn new(
        doko: DoState,
        last_played_action: Option<DoAction>
    ) -> Self {
        let observation = doko.observation_for_current_player();

        McDokoEnvState {
            doko,
            observation,
            last_played_action
        }
    }

}

fn rewards_from_obs(
    observation: &DoObservation
) -> Option<[f64; 4]> {
    let eog_stats = &observation.finished_observation;

    return match eog_stats {
        None => {
            None
        }
        Some(eog_stats) => {
            let player_bottom_points = if player_set_contains(
                eog_stats.re_players,
                PLAYER_BOTTOM
            ) {
                eog_stats.re_points
            } else {
                eog_stats.kontra_points
            };

            let player_left_points = if player_set_contains(
                eog_stats.re_players,
                PLAYER_LEFT
            ) {
                eog_stats.re_points
            } else {
                eog_stats.kontra_points
            };

            let player_top_points = if player_set_contains(
                eog_stats.re_players,
                PLAYER_TOP
            ) {
                eog_stats.re_points
            } else {
                eog_stats.kontra_points
            };

            let player_right_points = if player_set_contains(
                eog_stats.re_players,
                PLAYER_RIGHT
            ) {
                eog_stats.re_points
            } else {
                eog_stats.kontra_points
            };

            Some([
                player_bottom_points as f64,
                player_left_points as f64,
                player_top_points as f64,
                player_right_points as f64
            ])
        }
    }
}

impl Display for McDokoEnvState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{:?}", self.last_played_action);
    }
}

impl McEnvState<DoAction, 4, 26> for McDokoEnvState {
    fn current_player(&self) -> usize {
        self
            .observation
            .current_player
            .unwrap_or(PLAYER_BOTTOM)
    }

    fn is_terminal(&self) -> bool {
        self
            .observation
            .phase == DoPhase::Finished
    }

    fn last_action(&self) -> Option<DoAction> {
        self
            .last_played_action
    }

    fn possible_states(&self, first_expansion: bool) -> heapless::Vec<Self, 26> {
        let observation = &self.observation;

        let allowed_actions = allowed_actions_to_vec(
            observation.allowed_actions_current_player
        );

        let mut possible_states = heapless::Vec::new();
        for action in allowed_actions {
            let mut new_doko = self.doko.clone();

            new_doko.play_action(action);

            possible_states.push(McDokoEnvState::new(new_doko, Some(action)));
        }

        possible_states
    }

    fn allowed_actions(&self, first_expansion: bool) -> FdoAllowedActions {
        todo!()
    }

    fn by_action(&self, action: FdoAction) -> Self {
        todo!()
    }

    fn rewards_or_none(&self) -> Option<[f64; 4]> {
        return rewards_from_obs(&self.observation);
    }

    fn random_rollout(
        &self,
        rng: &mut SmallRng
    ) -> [f64; 4] {
        let mut rollout_doko = self
            .doko
            .clone();

        loop {
            let is_finished = rollout_doko.random_action_for_current_player(rng);

            if is_finished {
                break;
            }
        }

        let eog_stats = rollout_doko
            .observation_for_current_player();

        return rewards_from_obs(&eog_stats).unwrap();
    }
}