use async_trait::async_trait;
use rand::prelude::SmallRng;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use crate::get_weighted_action;
use crate::train_impi::FullDokoPolicy;

#[derive(Debug, Clone)]
pub struct ModifiedRandomFullDokoPolicy {
}


impl ModifiedRandomFullDokoPolicy {
    pub fn new() -> ModifiedRandomFullDokoPolicy {
        ModifiedRandomFullDokoPolicy {
        }
    }
}

#[async_trait]
impl FullDokoPolicy for ModifiedRandomFullDokoPolicy {
    async fn execute_policy(
        &self,

        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> FdoAction {
        let action = get_weighted_action::get_weighted_action(observation, rng);

        action
    }
}