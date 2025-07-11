use std::fmt::{Display, Formatter};
use rand::prelude::SmallRng;
use strum::EnumCount;
use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::allowed_actions_to_vec;
use rs_doko::basic::phase::DoPhase;
use rs_doko::observation::observation::DoObservation;
use rs_doko::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
use rs_doko::player::player_set::player_set_contains;
use rs_doko::state::state::DoState;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::action::allowed_actions::FdoAllowedActions;
use rs_full_doko::announcement::announcement::FdoAnnouncement;
use rs_full_doko::basic::phase::FdoPhase;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::reservation::reservation::FdoReservation;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
use crate::env::env_state::McEnvState;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct McFullDokoEnvState {
    pub doko: FdoState,

    pub last_played_action: Option<FdoAction>
}

impl McFullDokoEnvState {
    pub fn new(
        doko: FdoState,
        last_played_action: Option<FdoAction>
    ) -> Self {
        McFullDokoEnvState {
            doko,
            last_played_action
        }
    }

}

fn rewards_from_obs(
    observation: &FdoObservation
) -> Option<PlayerZeroOrientedArr<f64>> {
    let eog_stats = &observation.finished_stats;

    return match eog_stats {
        None => {
            None
        }
        Some(eog_stats) => {
            Some(eog_stats.player_points.map(|points| { (*points).into() }))
        }
    }
}

impl Display for McFullDokoEnvState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{:?}", self.last_played_action);
    }
}

impl McEnvState<FdoAction, 4, { FdoAction::COUNT }> for McFullDokoEnvState {
    fn current_player(&self) -> usize {
        self
            .doko
            .current_player
            .map(|it| it.index())
            .unwrap_or(FdoPlayer::BOTTOM.index())
    }

    fn is_terminal(&self) -> bool {
        self
            .doko
            .current_phase == FdoPhase::Finished
    }

    fn last_action(&self) -> Option<FdoAction> {
        self
            .last_played_action
    }

    fn possible_states(&self, first_expansion: bool) -> heapless::Vec<Self, 39> {
        let observation = self.doko.observation_for_current_player();

        let mut allowed_actions =
            observation
                .allowed_actions_current_player
                .clone();

        if !first_expansion {
            for obs in observation
                .phi_real_reservations
                .reservations
                .storage
                .iter() {
                let obs = *obs;

                if obs == FdoReservation::ClubsSolo
                    || obs == FdoReservation::SpadesSolo
                    || obs == FdoReservation::DiamondsSolo
                    || obs == FdoReservation::HeartsSolo
                    || obs == FdoReservation::TrumplessSolo
                    || obs == FdoReservation::JacksSolo
                    || obs == FdoReservation::QueensSolo {
                    allowed_actions.remove(FdoAction::ReservationClubsSolo);
                    allowed_actions.remove(FdoAction::ReservationSpadesSolo);
                    allowed_actions.remove(FdoAction::ReservationDiamondsSolo);
                    allowed_actions.remove(FdoAction::ReservationHeartsSolo);
                    allowed_actions.remove(FdoAction::ReservationTrumplessSolo);
                    allowed_actions.remove(FdoAction::ReservationJacksSolo);
                    allowed_actions.remove(FdoAction::ReservationQueensSolo);
                    allowed_actions.remove(FdoAction::ReservationWedding);
                }
            }

            allowed_actions.remove(FdoAction::AnnouncementReContra);
            allowed_actions.remove(FdoAction::AnnouncementNo90);
            allowed_actions.remove(FdoAction::AnnouncementNo60);
            allowed_actions.remove(FdoAction::AnnouncementNo30);
            allowed_actions.remove(FdoAction::AnnouncementBlack);
        }

        let mut possible_states = heapless::Vec::new();
        for action in allowed_actions.to_vec() {
            let mut new_doko = self.doko.clone();

            new_doko.play_action(action);

            possible_states.push(McFullDokoEnvState::new(new_doko, Some(action)));
        }

        possible_states
    }

    fn allowed_actions(
        &self,
        first_expansion: bool
    ) -> FdoAllowedActions {
        let observation = self.doko.observation_for_current_player();

        let mut allowed_actions =
            observation
                .allowed_actions_current_player
                .clone();

        if !first_expansion {
            for obs in observation
                .phi_real_reservations
                .reservations
                .storage
                .iter() {
                let obs = *obs;

                if obs == FdoReservation::ClubsSolo
                    || obs == FdoReservation::SpadesSolo
                    || obs == FdoReservation::DiamondsSolo
                    || obs == FdoReservation::HeartsSolo
                    || obs == FdoReservation::TrumplessSolo
                    || obs == FdoReservation::JacksSolo
                    || obs == FdoReservation::QueensSolo {
                    allowed_actions.remove(FdoAction::ReservationClubsSolo);
                    allowed_actions.remove(FdoAction::ReservationSpadesSolo);
                    allowed_actions.remove(FdoAction::ReservationDiamondsSolo);
                    allowed_actions.remove(FdoAction::ReservationHeartsSolo);
                    allowed_actions.remove(FdoAction::ReservationTrumplessSolo);
                    allowed_actions.remove(FdoAction::ReservationJacksSolo);
                    allowed_actions.remove(FdoAction::ReservationQueensSolo);
                    allowed_actions.remove(FdoAction::ReservationWedding);
                }
            }

            allowed_actions.remove(FdoAction::AnnouncementReContra);
            allowed_actions.remove(FdoAction::AnnouncementNo90);
            allowed_actions.remove(FdoAction::AnnouncementNo60);
            allowed_actions.remove(FdoAction::AnnouncementNo30);
            allowed_actions.remove(FdoAction::AnnouncementBlack);
        }

        return allowed_actions;
    }

    fn by_action(&self, action: FdoAction) -> Self {
        let mut new_doko = self.doko.clone();

        new_doko.play_action(action);

        return McFullDokoEnvState::new(new_doko, Some(action));
    }

    fn rewards_or_none(&self) -> Option<[f64; 4]> {
        let observation = self.doko.observation_for_current_player();

        return rewards_from_obs(&observation)
            .map(|it| it.storage);
    }

    fn random_rollout(
        &self,
        rng: &mut SmallRng
    ) -> [f64; 4] {
        let mut rollout_doko = self
            .doko
            .clone();

        loop {
            let is_finished = rollout_doko.random_action_for_current_player_no_announcement(rng);

            if is_finished {
                break;
            }
        }

        let eog_stats = rollout_doko
            .observation_for_current_player();

        return rewards_from_obs(&eog_stats)
            .map(|it| it.storage)
            .unwrap();
    }
}