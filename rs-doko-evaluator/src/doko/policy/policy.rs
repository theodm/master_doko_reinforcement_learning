use rs_doko::action::action::DoAction;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;

/// Eine Strategie, die sowohl den Zustand als auch die Beobachtung berücksichtigt und
/// damit für die Betrachtung von Spielen mit perfekter Information als auch
/// für Spiele mit unvollständiger Information geeignet ist. Handelt es sich um eine
/// unvollständige Strategie, soll sie netterweise :-) nur die Beobachtung berücksichtigen.
pub type EvDokoPolicy<'a> = dyn Fn(
    &DoState,
    &DoObservation,
    &mut rand::rngs::SmallRng,

    // Das gehört eigentlich nicht hier hin, ich bin aber auch nicht schlau genug,
    // das entsprechend abzukapseln :-/. Zieht eine eigene Dependency mit sich.
    &mut CachedMCTS<
        McDokoEnvState,
        DoAction,
        4,
        26
    >
) -> DoAction + 'a;

