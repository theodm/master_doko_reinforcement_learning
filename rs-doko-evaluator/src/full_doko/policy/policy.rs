use std::sync::Arc;
use async_trait::async_trait;
use strum::EnumCount;
use rs_doko::action::action::DoAction;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use rs_doko_alpha_zero::alpha_zero::batch_processor::batch_processor::BatchProcessorSender;
use rs_doko_alpha_zero::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;

/// Eine Strategie, die sowohl den Zustand als auch die Beobachtung berücksichtigt und
/// damit für die Betrachtung von Spielen mit perfekter Information als auch
/// für Spiele mit unvollständiger Information geeignet ist. Handelt es sich um eine
/// unvollständige Strategie, soll sie netterweise :-) nur die Beobachtung berücksichtigen.
#[async_trait]
pub trait EvFullDokoPolicy: Send + Sync {
    async fn evaluate(
        &self,

        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut rand::rngs::SmallRng,
    ) -> (FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }]);
}