use std::sync::Arc;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use strum::EnumCount;
use rs_doko_evaluator::full_doko::evaluate_single_game::full_doko_evaluate_single_game;
use rs_doko_evaluator::full_doko::policy::mcts_policy::MCTSWorkers;
use rs_doko_evaluator::full_doko::policy::policy::EvFullDokoPolicy;
use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::display::display::display_game;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;

pub async fn doko_mcts_simple_tree() {
    let mut rng = SmallRng::from_os_rng();

    let seed = 1011;

    let mut game_creation_rng = SmallRng::seed_from_u64(seed);

    let (workers, factory) = MCTSWorkers::create_and_run(
        1,
        200000*10,
        200000*10
    );

    let mut state = FdoState::new_game(&mut game_creation_rng);

    println!("{}", display_game(state.observation_for_current_player()));

    let mut game_creation_rng = SmallRng::seed_from_u64(seed);

    let policy = Arc::new(factory.policy(20f32 * 2f32.sqrt(), 20000));

    let policies: [Arc<dyn EvFullDokoPolicy>; 4] = [
        policy.clone(),
        policy.clone(),
        policy.clone(),
        policy.clone(),
    ];

    let result =
        full_doko_evaluate_single_game(policies, &mut game_creation_rng, &mut rng).await;

    workers.abort();

}
