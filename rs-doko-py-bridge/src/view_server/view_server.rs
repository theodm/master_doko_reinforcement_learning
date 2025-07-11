use std::sync::Arc;
use std::thread::available_parallelism;
use rand::{Rng, SeedableRng};
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSWorkers;
use rs_full_doko::state::state::FdoState;
use crate::compare_impi::all_policies::{mcts_cap_avg_policy, mcts_cap_maxn_policy};
use crate::compare_impi::compare_impi::{EvImpiPolicy};
use crate::view_server::api::{ApGameState, ApPlayedGame};

pub async fn create_game_for_web_view(
    seed: u64,

    policies: [Arc<impl EvImpiPolicy>; 4],
) -> String {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let mut state = FdoState::new_game(&mut rng);

    let mut results = ApPlayedGame {
        players: vec!["Bottom".to_string(), "Left".to_string(), "Top".to_string(), "Right".to_string()],
        states: vec![],
    };

    let mut last_action = None;
    loop {
        let obs = state.observation_for_current_player();

        results.states.push(ApGameState::create_from(&state, &obs, last_action));

        if obs.finished_stats.is_some() {
            break;
        }

        let (action, numcons, not_consistent_reasons, was_random)  = policies[state.current_player.unwrap().index()]
            .evaluate(&state, &obs, &mut rng)
            .await;

        state.play_action(action);

        last_action = Some(action);
    }

    return serde_json::to_string_pretty(&results).unwrap();
}

pub fn create_game_for_web_view_exec2(
    seed: Option<u64>
) -> String {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .enable_time()
        .build()
        .unwrap();

    let available_parallelism = available_parallelism().unwrap().get();

    return rt.block_on(async {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            rng.gen_range(0..u64::MAX)
        });

        let (mcts_workers, mcts_factory) = MCTSWorkers::create_and_run(
            available_parallelism,
            200001,
            200001
        );

        let mcts_policy = Arc::new(mcts_cap_avg_policy(
            mcts_factory,

            200000,
            5.5,
            10
        ));

        let result = create_game_for_web_view(
            seed,
            [
                mcts_policy.clone(),
                mcts_policy.clone(),
                mcts_policy.clone(),
                mcts_policy.clone(),
            ]
        ).await;

        mcts_workers.abort();

        result
    });
}