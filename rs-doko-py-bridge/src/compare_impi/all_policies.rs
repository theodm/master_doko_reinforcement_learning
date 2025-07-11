use async_trait::async_trait;
use rs_full_doko::action::action::FdoAction;
use strum::EnumCount;
use std::sync::Arc;
use rs_doko_alpha_zero::alpha_zero::alpha_zero_evaluation_options::AlphaZeroEvaluationOptions;
use rs_doko_evaluator::full_doko::policy::az_policy::AZPolicyFactory;
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSFactory;
use rs_doko_impi::eval::eval_net_fn::EvaluateNetFn;
use crate::compare_impi::compare_impi::{ARSampling, CAPSampling, DefaultImpiPolicy, FullPolicy, RandomImpiPolicy};
use crate::compare_impi::policy_fusion::{PolicyFusionAverageStrategy, PolicyFusionMaxN};

pub fn az_ar_maxn_policy(
    mut az_factory: AZPolicyFactory,
    iterations: usize,
    puct: f32,
    num_samples: usize,
    evaluate_net_fn: Arc<dyn EvaluateNetFn>,
) -> DefaultImpiPolicy {
    let base_policy = az_factory.policy(AlphaZeroEvaluationOptions {
        iterations,
        dirichlet_alpha: 0.5,
        dirichlet_epsilon: 0.0,
        puct_exploration_constant: puct,
        min_or_max_value: 0.0,
        par_iterations: 0,
        virtual_loss: 0.0,
    });

    let sampling_fn = ARSampling {
        temperature: 0.4,
        evaluate_net_fn: evaluate_net_fn.clone(),
    };

    let fusion_fn = PolicyFusionMaxN {};

    DefaultImpiPolicy::new(
        "az_ar_maxn_policy".to_string(),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples,
    )
}

pub fn az_ar_avg_policy(
    mut az_factory: AZPolicyFactory,
    iterations: usize,
    puct: f32,
    num_samples: usize,
    evaluate_net_fn: Arc<dyn EvaluateNetFn>,
) -> DefaultImpiPolicy {
    let base_policy = az_factory.policy(AlphaZeroEvaluationOptions {
        iterations,
        dirichlet_alpha: 0.5,
        dirichlet_epsilon: 0.0,
        puct_exploration_constant: puct,
        min_or_max_value: 0.0,
        par_iterations: 0,
        virtual_loss: 0.0,
    });

    let sampling_fn = ARSampling {
        temperature: 0.4,
        evaluate_net_fn: evaluate_net_fn.clone(),
    };

    let fusion_fn = PolicyFusionAverageStrategy {};

    DefaultImpiPolicy::new(
        "az_ar_avg_policy".to_string(),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples,
    )
}

pub fn mcts_ar_maxn_policy(
    mcts_factory: MCTSFactory,
    iterations: usize,
    uct: f32,
    num_samples: usize,
    evaluate_net_fn: Arc<dyn EvaluateNetFn>,
) -> DefaultImpiPolicy {
    let base_policy = mcts_factory.policy(uct, iterations);

    let sampling_fn = ARSampling {
        temperature: 0.4,
        evaluate_net_fn: evaluate_net_fn.clone(),
    };

    let fusion_fn = PolicyFusionMaxN {};

    DefaultImpiPolicy::new(
        "mcts_ar_maxn_policy".to_string(),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples,
    )
}

pub fn mcts_ar_avg_policy(
    mcts_factory: MCTSFactory,
    iterations: usize,
    uct: f32,
    num_samples: usize,
    evaluate_net_fn: Arc<dyn EvaluateNetFn>,
) -> DefaultImpiPolicy {
    let base_policy = mcts_factory.policy(uct, iterations);

    let sampling_fn = ARSampling {
        temperature: 0.4,
        evaluate_net_fn: evaluate_net_fn.clone(),
    };

    let fusion_fn = PolicyFusionAverageStrategy {};

    DefaultImpiPolicy::new(
        format!(
            "mcts_ar_avg_policy_{}_{}_{}",
            iterations,
            uct,
            num_samples
        ),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples,
    )
}

pub fn mcts_cap_maxn_policy(
    mcts_factory: MCTSFactory,
    iterations: usize,
    uct: f32,
    num_samples: usize 
) -> DefaultImpiPolicy {
    let base_policy = mcts_factory.policy(uct, iterations);

    let sampling_fn = CAPSampling {};

    let fusion_fn = PolicyFusionMaxN {};

    DefaultImpiPolicy::new(
        "mcts_cap_maxn_policy".to_string(),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples, // Verwendung des Parameters
    )
}

pub fn mcts_cap_avg_policy(
    mcts_factory: MCTSFactory,
    iterations: usize,
    uct: f32,
    num_samples: usize, 
) -> DefaultImpiPolicy {
    let base_policy = mcts_factory.policy(uct, iterations);

    let sampling_fn = CAPSampling {};

    let fusion_fn = PolicyFusionAverageStrategy {};

    DefaultImpiPolicy::new(
        "mcts_cap_avg_policy".to_string(),
        Arc::new(base_policy),
        Arc::new(sampling_fn),
        Arc::new(fusion_fn),
        num_samples, 
    )
}

pub fn mcts_pi_policy(
    mcts_factory: MCTSFactory,
    iterations: usize,
    uct: f32,
) -> FullPolicy {
    let base_policy = mcts_factory.policy(uct, iterations);

    FullPolicy::new(
        format!("mcts_pi_{}_{}", iterations, uct),
        Arc::new(base_policy)
    )
}

pub fn az_pi_policy(
    mut az_factory: AZPolicyFactory
) -> FullPolicy {
    FullPolicy::new(
        format!("az_pi_{}_{}", 200, 4),
        Arc::new(az_factory.policy(AlphaZeroEvaluationOptions {
            iterations: 200,
            dirichlet_alpha: 0.5,
            dirichlet_epsilon: 0.0,
            puct_exploration_constant: 4.0,
            min_or_max_value: 0.0,
            par_iterations: 0,
            virtual_loss: 0.0,
        }))
    )
}

pub fn random_policy() -> FullPolicy {
    FullPolicy::new(
        "random_policy".to_string(),
        Arc::new(RandomImpiPolicy {})
    )

}