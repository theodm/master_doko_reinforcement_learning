use std::sync::Arc;
use std::thread::available_parallelism;
use async_trait::async_trait;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use strum::{EnumCount, IntoEnumIterator};
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSFactory;
use rs_doko_impi::mcts_full_doko_policy::MCTSFullDokoPolicy;
use rs_doko_impi::train_impi::FullDokoPolicy;
use rs_full_doko::matching::card_matching::card_matching_full;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::state::state::FdoState;
use rs_doko_evaluator::full_doko::policy::policy;
use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
use rs_doko_impi::eval::eval_net_fn::EvaluateNetFn;
use rs_doko_impi::forward::{forward_pred_process_with_mask, ForwardPredResult};
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::matching::is_consistent::NotConsistentReason;
use crate::compare_impi::compare_impi::SamplignAndMCTSStepResult::{Failed, Successful};
use crate::compare_impi::policy_fusion::{PolicyFusionFn, PolicyFusionMaxN};
use std::hash::{Hash, Hasher};
use std::fmt;
use itertools::all;

impl PartialEq for DefaultImpiPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for DefaultImpiPolicy {}

impl Hash for DefaultImpiPolicy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl fmt::Debug for DefaultImpiPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefaultImpiPolicy")
            .field("name", &self.name)
            .finish()
    }
}


#[async_trait]
pub trait SamplingFn: Send + Sync {
    async fn sample(
        &self,
        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> ForwardPredResult;
}

pub(crate) struct CAPSampling {

}

#[async_trait]
impl SamplingFn for CAPSampling {
    async fn sample(
        &self,
        state: &FdoState,
        observation: &FdoObservation,

        rng: &mut SmallRng
    ) -> ForwardPredResult {
        let sampled_state = card_matching_full(
            state,
            observation,

            rng
        );

        return ForwardPredResult::Consistent(sampled_state);
    }

}

pub(crate) struct ARSampling {
    pub(crate) temperature: f32,
    pub(crate) evaluate_net_fn: Arc<dyn EvaluateNetFn>
}

#[async_trait]
impl SamplingFn for ARSampling {
    async fn sample(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> ForwardPredResult {
        forward_pred_process_with_mask(
            state,
            self.temperature,
            1f32/30000f32,
            &self.evaluate_net_fn,
            rng
        ).await
    }
}

#[async_trait]
pub trait EvImpiPolicy : Send + Sync {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> (FdoAction, usize, Vec<NotConsistentReason>, bool);

    fn name(&self) -> String;
}


pub struct FullPolicy {
    pub name: String,

    full_policy: Arc<dyn EvFullDokoPolicy>,
}

impl PartialEq for FullPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for FullPolicy {

}

impl Hash for FullPolicy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

#[async_trait]
impl EvImpiPolicy for FullPolicy {

    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> (FdoAction, usize, Vec<NotConsistentReason>, bool) {
        let (action, visits, values) = self
            .full_policy
            .evaluate(state, observation, rng).await;

        (action, 0, vec![], false)
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

impl FullPolicy {
    pub fn new(
        name: String,
        full_policy: Arc<dyn EvFullDokoPolicy>,
    ) -> Self {
        FullPolicy {
            name,
            full_policy,
        }
    }

}

#[derive(Clone)]
pub struct DefaultImpiPolicy {
    pub name: String,

    full_policy: Arc<dyn EvFullDokoPolicy>,
    sampling_fn: Arc<dyn SamplingFn>,
    policy_fusion_fn: Arc<dyn PolicyFusionFn>,

    num_samples: usize,
}


enum SamplignAndMCTSStepResult {
    Successful((FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }])),
    Failed(NotConsistentReason),
}
impl DefaultImpiPolicy {
    pub fn new(
        name: String,

        full_policy: Arc<dyn EvFullDokoPolicy>,
        sampling_fn: Arc<dyn SamplingFn>,
        policy_fusion_fn: Arc<dyn PolicyFusionFn>,

        num_samples: usize,

    ) -> Self {
        DefaultImpiPolicy {
            name,
            full_policy,
            sampling_fn,
            policy_fusion_fn,
            num_samples,
        }
    }

    pub async fn execute(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
    ) -> (FdoAction, usize, Vec<NotConsistentReason>, bool) {
        async fn run_batch(
            sampling_fn: Arc<dyn SamplingFn>,
            full_policy: Arc<dyn EvFullDokoPolicy>,
            state: FdoState,
            observation: FdoObservation,
            num_samples: usize,
        ) -> (
            Vec<([usize; FdoAction::COUNT], [f32; FdoAction::COUNT])>,
            Vec<NotConsistentReason>,
        ) {
            let mut joinset = tokio::task::JoinSet::new();

            for _ in 0..num_samples {
                let sampling_fn = sampling_fn.clone();
                let full_policy = full_policy.clone();
                let state = state.clone();
                let observation = observation.clone();

                joinset.spawn(async move {
                    let mut rng = SmallRng::from_os_rng();

                    match sampling_fn.sample(&state, &observation, &mut rng).await {
                        ForwardPredResult::Consistent(sampled_state) => {
                            let sampled_obs = sampled_state
                                .observation_for_current_player();

                            let (action, visits, values) = full_policy
                                .evaluate(&sampled_state, &sampled_obs, &mut rng)
                                .await;

                            let allowed_actions = observation
                                .allowed_actions_current_player
                                .clone();

                            let allowed_actions_sampled = observation
                                .allowed_actions_current_player
                                .clone();

                            if allowed_actions_sampled != allowed_actions {
                                println!("Warning: allowed actions differ between sampled and original observation");
                            }

                            for a in FdoAction::iter() {
                                if visits[a.to_index()] > 0 && !allowed_actions.contains(a) {
                                    println!("Warning: action {:?} has {} visits but is not allowed", a, visits[a.to_index()]);
                                }
                            }

                            Successful((action, visits, values))
                        }
                        ForwardPredResult::NotConsistent(reason) => Failed(reason),
                    }
                });
            }

            let mut succ = Vec::new();
            let mut fail = Vec::new();

            // Verarbeite jede Task, sobald sie fertig ist:
            while let Some(join_res) = joinset.join_next().await {
                match join_res {
                    Ok(Successful((_act, visits, values))) => {
                        succ.push((visits, values));
                    }
                    Ok(Failed(reason)) => {
                        fail.push(reason);
                    }
                    Err(join_err) => {
                        eprintln!("Join error in run_batch: {}", join_err);
                    }
                }
            }

            (succ, fail)
        }

        let mut all_succ = Vec::new();
        let mut all_unsucc = Vec::new();

        for _ in 0..1 {
            let (succ, unsucc) = run_batch(
                self.sampling_fn.clone(),
                self.full_policy.clone(),
                state.clone(),
                observation.clone(),
                self.num_samples,
            )
                .await;

            all_succ.extend(succ);
            all_unsucc.extend(unsucc);

            if !all_succ.is_empty() {
                break;
            }
        }

        if !all_succ.is_empty() {
            let num_success = all_succ.len();
            let mut visits_vec = Vec::with_capacity(num_success);
            let mut values_vec = Vec::with_capacity(num_success);

            for (visits, values) in all_succ {
                visits_vec.push(visits);
                values_vec.push(values);
            }

            let allowed_actions = observation.allowed_actions_current_player.clone();

            let mut best_actions = vec![];
            for i in 0..num_success {
                let visits = visits_vec[i];

                let max_visits = visits
                    .iter()
                    .max()
                    .unwrap_or(&0);

                let max_index = visits
                    .iter()
                    .position(|&x| x == *max_visits)
                    .unwrap_or(0);

                best_actions.push(FdoAction::from_index(max_index));
            }

            let action = self.policy_fusion_fn
                .fuse(visits_vec.clone(), values_vec.clone(), allowed_actions.clone()).await;

            // println!("Best actions before fusion: {:?} after: {:?}", best_actions, action);

            if allowed_actions.contains(action) {
                // println!("Action {:?} is allowed", action);
            } else {
                println!("visits: {:?}", visits_vec);
                println!("values: {:?}", values_vec);
                println!("allowed actions: {:?}", allowed_actions);
                println!("Warning: action {:?} is not allowed", action);
            }

            (action, num_success, all_unsucc, false)
        } else {
            // Nach 5 Durchgängen noch immer kein Erfolg → Zufalls-Fallback
            let mut allowed = observation.allowed_actions_current_player.clone();
            allowed.remove(FdoAction::AnnouncementReContra);
            allowed.remove(FdoAction::AnnouncementNo90);
            allowed.remove(FdoAction::AnnouncementNo60);
            allowed.remove(FdoAction::AnnouncementNo30);
            allowed.remove(FdoAction::AnnouncementBlack);

            let mut rng = SmallRng::from_os_rng();
            let action = allowed.random(&mut rng);
            (action, 0, all_unsucc, true)
        }
    }
}


// impl DefaultImpiPolicy {
//     pub fn new(
//         name: String,
//
//         full_policy: Arc<dyn EvFullDokoPolicy>,
//         sampling_fn: Arc<dyn SamplingFn>,
//         policy_fusion_fn: Arc<dyn PolicyFusionFn>,
//
//         num_samples: usize,
//
//     ) -> Self {
//         DefaultImpiPolicy {
//             name,
//             full_policy,
//             sampling_fn,
//             policy_fusion_fn,
//             num_samples,
//         }
//     }
//
//     pub async fn execute(
//         &self,
//         state: &FdoState,
//         observation: &FdoObservation,
//     ) -> (FdoAction, Vec<NotConsistentReason>, bool) {
//
//         let mut joinset =
//             tokio::task::JoinSet::new();
//
//         for i in 0..self.num_samples {
//             let sampling_fn = self.sampling_fn.clone();
//             let full_policy = self.full_policy.clone();
//             let state = state.clone();
//             let observation = observation.clone();
//
//             joinset.spawn(async move {
//                 let mut rng = SmallRng::from_os_rng();
//
//                 let sampled_state = sampling_fn.sample(
//                     &state,
//                     &observation,
//                     &mut rng
//                 ).await;
//
//                 return match sampled_state {
//                     ForwardPredResult::Consistent(sampled_state) => {
//                         let policy_results = full_policy.evaluate(
//                             &sampled_state,
//                             &observation,
//                             &mut rng
//                         ).await;
//
//                         Successful(policy_results)
//                     }
//                     ForwardPredResult::NotConsistent(not_consistent_reason) => {
//                         return Failed(not_consistent_reason);
//                     }
//                 }
//             });
//         }
//
//         let results = joinset
//             .join_all()
//             .await;
//
//         let succesful = results
//             .iter()
//             .filter_map(|x| {
//                 match x {
//                     Successful(res) => Some(res),
//                     Failed(_) => None
//                 }
//             })
//             .collect::<Vec<_>>();
//
//         let unsuccesful = results
//             .iter()
//             .filter_map(|x| {
//                 match x {
//                     Successful(_) => None,
//                     Failed(reason) => Some(reason.clone())
//                 }
//             })
//             .collect::<Vec<_>>();
//
//         if succesful.len() > 0 {
//             let mut moves_to_visits = vec![];
//             let mut moves_to_values = vec![];
//
//             for r in succesful {
//                 moves_to_visits.push(
//                     r.1
//                 );
//                 moves_to_values.push(
//                     r.2
//                 );
//             }
//
//             let action = self
//                 .policy_fusion_fn
//                 .fuse(moves_to_visits, moves_to_values)
//                 .await;
//
//             (action, unsuccesful, false)
//         } else {
//             // Wenn alle Samples nicht konsistent sind, dann
//             // geben wir eine zufällige Aktion zurück.
//             let mut allowed_actions = observation.allowed_actions_current_player.clone();
//
//             allowed_actions.remove(FdoAction::AnnouncementReContra);
//             allowed_actions.remove(FdoAction::AnnouncementNo90);
//             allowed_actions.remove(FdoAction::AnnouncementNo60);
//             allowed_actions.remove(FdoAction::AnnouncementNo30);
//             allowed_actions.remove(FdoAction::AnnouncementBlack);
//
//             let mut rng = SmallRng::from_os_rng();
//             let action = allowed_actions.random(&mut rng);
//
//             (action, unsuccesful, true)
//         }
//     }
// }

#[async_trait]
impl EvImpiPolicy for DefaultImpiPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> (FdoAction, usize, Vec<NotConsistentReason>, bool) {
        let action = self.execute(
            state,
            observation
        ).await;

        return action;
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

pub struct RandomImpiPolicy {
}

#[async_trait]
impl EvFullDokoPolicy for RandomImpiPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> (FdoAction, [usize; { FdoAction::COUNT }], [f32; { FdoAction::COUNT }]) {
        let mut allowed_actions = observation.allowed_actions_current_player.clone();

        allowed_actions.remove(FdoAction::AnnouncementReContra);
        allowed_actions.remove(FdoAction::AnnouncementNo90);
        allowed_actions.remove(FdoAction::AnnouncementNo60);
        allowed_actions.remove(FdoAction::AnnouncementNo30);
        allowed_actions.remove(FdoAction::AnnouncementBlack);

        return (allowed_actions.random(rng), [0; { FdoAction::COUNT }], [0.0; { FdoAction::COUNT }]);
    }
}

impl RandomImpiPolicy {
    pub fn new() -> Self {
        RandomImpiPolicy {}
    }
}

impl PartialEq for RandomImpiPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for RandomImpiPolicy {}

impl Hash for RandomImpiPolicy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

#[async_trait]
impl EvImpiPolicy for RandomImpiPolicy {
    async fn evaluate(
        &self,
        state: &FdoState,
        observation: &FdoObservation,
        rng: &mut SmallRng
    ) -> (FdoAction, usize, Vec<NotConsistentReason>, bool) {
        let mut allowed_actions = observation.allowed_actions_current_player.clone();

        allowed_actions.remove(FdoAction::AnnouncementReContra);
        allowed_actions.remove(FdoAction::AnnouncementNo90);
        allowed_actions.remove(FdoAction::AnnouncementNo60);
        allowed_actions.remove(FdoAction::AnnouncementNo30);
        allowed_actions.remove(FdoAction::AnnouncementBlack);

        return (allowed_actions.random(rng), 0, vec![], false);
    }

    fn name(&self) -> String {
        "RandomImpiPolicy".to_string()
    }
}

