use rs_doko::action::action::DoAction;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use fxhash::FxHasher64;
use strum::EnumCount;
use rs_doko_networks::full_doko::var1::encode_pi::encode_state_pi;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::basic::phase::FdoPhase;
use rs_full_doko::display::display::display_game;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::state::state::FdoState;
use crate::env::env_state::AzEnvState;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FdoAzEnvState {
    pub doko: FdoState,
    pub observation: FdoObservation,

    pub last_played_action: Option<FdoAction>,
}

const MIN_EPOCH: usize = 10;

impl FdoAzEnvState {
    pub fn new(doko: FdoState, last_played_action: Option<FdoAction>) -> Self {
        let observation = doko.observation_for_current_player();

        FdoAzEnvState {
            doko,
            observation,
            last_played_action,
        }
    }
}

impl Display for FdoAzEnvState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.last_played_action)
    }
}

impl AzEnvState<FdoAction, 4, {FdoAction::COUNT}> for FdoAzEnvState {
    const GAME_NAME: &'static str = "full_doko";

    fn current_player(&self) -> usize {
        self.observation.current_player.unwrap_or(FdoPlayer::BOTTOM).index()
    }

    fn is_terminal(&self) -> bool {
        self.observation.phase == FdoPhase::Finished
    }

    fn rewards_or_none(&self) -> Option<[f32; 4]> {
        let fstats = &self.observation.finished_stats;

        match fstats {
            None => None,
            Some(fstats) => {
                fstats
                    .player_points
                    .iter()
                    .map(|&x| x as f32 / 8f32)
                    .collect::<Vec<f32>>()
                    .try_into()
                    .ok()
            }
        }
    }

    fn encode_into_memory(&self, memory: &mut [i64]) {
        let memory_real = encode_state_pi(
            &self.doko,
            &self.observation
        );

        memory.copy_from_slice(&memory_real);
    }

    fn allowed_actions_by_action_index(
        &self,
        is_secondary: bool,
        _epoch: usize
    ) -> heapless::Vec<usize, 39> {
        let observation = &self.observation;

        let mut allowed_actions = observation
            .allowed_actions_current_player
            .clone();

        if is_secondary || _epoch < MIN_EPOCH {
            allowed_actions.remove(FdoAction::AnnouncementReContra);
            allowed_actions.remove(FdoAction::AnnouncementNo90);
            allowed_actions.remove(FdoAction::AnnouncementNo60);
            allowed_actions.remove(FdoAction::AnnouncementNo30);
            allowed_actions.remove(FdoAction::AnnouncementBlack);
        }

        allowed_actions.to_vec().iter().map(|x| (*x).to_index()).collect()
    }

    fn number_of_allowed_actions(
        &self,
        epoch: usize
    ) -> usize {
        let mut allowed_actions = self
            .observation.allowed_actions_current_player
            .clone();

        if epoch < MIN_EPOCH {
            allowed_actions.remove(FdoAction::AnnouncementReContra);
            allowed_actions.remove(FdoAction::AnnouncementNo90);
            allowed_actions.remove(FdoAction::AnnouncementNo60);
            allowed_actions.remove(FdoAction::AnnouncementNo30);
            allowed_actions.remove(FdoAction::AnnouncementBlack);
        }

        allowed_actions.len()
    }

    fn take_action_by_action_index(&self, action: usize, skip_single: bool, epoch: usize) -> Self {
        let action = FdoAction::from_index(action);

        let mut new_doko = self.doko.clone();
        new_doko.play_action(action);

        if skip_single {
            loop {
                if new_doko.observation_for_current_player()
                    .phase == FdoPhase::Finished {
                    break;
                }

                let mut allowed_actions = new_doko
                    .observation_for_current_player()
                    .allowed_actions_current_player
                    .clone();

                allowed_actions.remove(FdoAction::AnnouncementReContra);
                allowed_actions.remove(FdoAction::AnnouncementNo90);
                allowed_actions.remove(FdoAction::AnnouncementNo60);
                allowed_actions.remove(FdoAction::AnnouncementNo30);
                allowed_actions.remove(FdoAction::AnnouncementBlack);

                if allowed_actions.len() == 1 {
                    new_doko.play_action(allowed_actions.to_vec()[0]);
                } else {
                    break;
                }
            }
        }

        FdoAzEnvState::new(new_doko, Some(action))
    }

    fn id(&self) -> u64 {
        // ToDo: Käse und falsch :(
        let mut hasher = FxHasher64::default();

        self.hash(&mut hasher);

        hasher.finish()
    }

    fn last_action(&self) -> Option<FdoAction> {
        self.last_played_action
    }

    fn display_game(&self) -> String {
        display_game(self.doko.observation_for_current_player())
    }
}
