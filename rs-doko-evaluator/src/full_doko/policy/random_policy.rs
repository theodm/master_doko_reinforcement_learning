use async_trait::async_trait;
use strum::EnumCount;
use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::random_action;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use crate::full_doko::policy::policy::EvFullDokoPolicy;


#[derive(Clone)]
pub struct EvFullDokoRandomPolicy {

}

#[async_trait]
impl EvFullDokoPolicy for EvFullDokoRandomPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut rand::rngs::SmallRng,
    ) -> (FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }]) {
        let mut allowed_actions = observation.allowed_actions_current_player.clone();

        allowed_actions.remove(FdoAction::AnnouncementReContra);
        allowed_actions.remove(FdoAction::AnnouncementNo90);
        allowed_actions.remove(FdoAction::AnnouncementNo60);
        allowed_actions.remove(FdoAction::AnnouncementNo30);
        allowed_actions.remove(FdoAction::AnnouncementBlack);

        (
            allowed_actions.random(rng),
            [0; { FdoAction::COUNT }],
            [0.0f32; { FdoAction::COUNT }]
        )
    }
}