use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::random_action;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;

pub fn doko_random_policy(
    _: &DoState,
    observation: &DoObservation,
    rng: &mut rand::rngs::SmallRng,

    _: &mut CachedMCTS<
        McDokoEnvState,
        DoAction,
        4,
        26
    >
) -> DoAction {
    let actions = observation.allowed_actions_current_player;

    random_action(actions, rng)
}