
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::available_parallelism,
};

use rand::{prelude::SliceRandom, rngs::SmallRng, SeedableRng};
use tokio::{runtime::Builder, sync::Mutex};

use rs_doko_evaluator::full_doko::policy::mcts_policy::{MCTSFactory, MCTSWorkers};
use rs_doko_impi::csv_writer_thread::CSVWriterThread;

use crate::compare_impi::{
    all_policies::{mcts_pi_policy, random_policy},
    compare_impi::EvImpiPolicy,
    evaluate_single_game_impi::full_doko_evaluate_single_game_impi,
};

fn repeat_policy_remaining(
    n_mcts: usize,
    cur: &Arc<dyn EvImpiPolicy>,
    remaining: &Arc<dyn EvImpiPolicy>,
) -> [Arc<dyn EvImpiPolicy>; 4] {
    match n_mcts {
        1 => [cur.clone(), remaining.clone(), remaining.clone(), remaining.clone()],
        2 => [cur.clone(), cur.clone(), remaining.clone(), remaining.clone()],
        3 => [cur.clone(), cur.clone(), cur.clone(), remaining.clone()],
        _ => panic!("n_mcts muss 1–3 sein"),
    }
}

struct GameToPlay {
    game_id:          usize,
    iteration:        usize,
    uct_param:        f32,
    uct_key:          String,  
    prev_iteration:   usize,
    prev_best_uct:    f32,
    num_mcts_players: usize,

    policies:         [Arc<dyn EvImpiPolicy>; 4],
    player_1_name:    String,
    player_2_name:    String,
    player_3_name:    String,
    player_4_name:    String,
}

pub fn run_mcts_vs_previous(games_per_matchup: usize) {
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();

    rt.block_on(async {
        let cores = available_parallelism().unwrap().get();
        let (_workers, mut mcts_factory) =
            MCTSWorkers::create_and_run(cores, 825_000, 824_999);

        let uct_values: [f32; 4] = [
            5f32.sqrt(),
            10f32 * 2f32.sqrt(),
            20f32 * 2f32.sqrt(),
            20f32 * 5f32.sqrt(),
        ];

        let iterations: [usize; 7] = [10, 20, 100, 200, 1_000, 10_000, 100_000];
        let n_mcts_grid: [usize; 3] = [3, 2, 1];

        let total_games =
            games_per_matchup * uct_values.len() * iterations.len() * n_mcts_grid.len();
        let csv_writer = CSVWriterThread::new(
            PathBuf::from("mcts_vs_previous_full.csv"),
            &[
                "iteration","uct_param","prev_iteration","prev_best_uct","num_mcts_players",
                "game_id","player_1_name","player_2_name","player_3_name","player_4_name",
                "player_1_points","player_2_points","player_3_points","player_4_points",
                "player_1_number_executed_actions","player_2_number_executed_actions",
                "player_3_number_executed_actions","player_4_number_executed_actions",
                "number_of_actions","played_game_mode",
                "lowest_announcement_re","lowest_announcement_contra","branching_factor",
                "player_1_reservation_made","player_2_reservation_made",
                "player_3_reservation_made","player_4_reservation_made",
                "player_1_lowest_announcement_made","player_2_lowest_announcement_made",
                "player_3_lowest_announcement_made","player_4_lowest_announcement_made",
                "player_1_number_consistent","player_2_number_consistent",
                "player_3_number_consistent","player_4_number_consistent",
                "player_1_number_not_consistent","player_2_number_not_consistent",
                "player_3_number_not_consistent","player_4_number_not_consistent",
                "player_1_number_was_random_because_not_consistent",
                "player_2_number_was_random_because_not_consistent",
                "player_3_number_was_random_because_not_consistent",
                "player_4_number_was_random_because_not_consistent",
                "player_1_number_not_consistentHandSizeMismatch",
                "player_2_number_not_consistentHandSizeMismatch",
                "player_3_number_not_consistentHandSizeMismatch",
                "player_4_number_not_consistentHandSizeMismatch",
                "player_1_number_not_consistentRemainingCardsLeft",
                "player_2_number_not_consistentRemainingCardsLeft",
                "player_3_number_not_consistentRemainingCardsLeft",
                "player_4_number_not_consistentRemainingCardsLeft",
                "player_1_number_not_consistentNotInRemainingCards",
                "player_2_number_not_consistentNotInRemainingCards",
                "player_3_number_not_consistentNotInRemainingCards",
                "player_4_number_not_consistentNotInRemainingCards",
                "player_1_number_not_consistentAlreadyDiscardedColor",
                "player_2_number_not_consistentAlreadyDiscardedColor",
                "player_3_number_not_consistentAlreadyDiscardedColor",
                "player_4_number_not_consistentAlreadyDiscardedColor",
                "player_1_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_2_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_3_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_4_number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding",
                "player_1_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_2_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_3_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_4_number_not_consistentHasNoClubQueenButAnnouncedRe",
                "player_1_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_2_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_3_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_4_number_not_consistentHasClubQueenButAnnouncedKontra",
                "player_1_number_not_consistentWrongReservation",
                "player_2_number_not_consistentWrongReservation",
                "player_3_number_not_consistentWrongReservation",
                "player_4_number_not_consistentWrongReservation",
                "player_1_number_not_consistentWrongReservationClubQ",
                "player_2_number_not_consistentWrongReservationClubQ",
                "player_3_number_not_consistentWrongReservationClubQ",
                "player_4_number_not_consistentWrongReservationClubQ",
                "player_1_was_re","player_2_was_re","player_3_was_re","player_4_was_re",
            ],
            Some(total_games as u64),
            false,
        );

        let mut prev_iteration = 0usize;
        let mut prev_best_uct = 0.0f32;
        let random_pol: Arc<dyn EvImpiPolicy> = Arc::new(random_policy());

        for &iteration in &iterations {
            println!("╭─ Iteration {iteration}");

            let sums: Arc<Mutex<HashMap<String, i32>>> =
                Arc::new(Mutex::new(HashMap::new()));
                
            let mut games = Vec::<GameToPlay>::new();
            let mut id_counter = 0usize;

            for &uct in &uct_values {
                let uct_key = format!("{:.6}", uct); 

                let cur_pol: Arc<dyn EvImpiPolicy> =
                    Arc::new(mcts_pi_policy(mcts_factory.clone(), iteration, uct));

                let prev_pol: Arc<dyn EvImpiPolicy> = if prev_iteration == 0 {
                    random_pol.clone()
                } else {
                    Arc::new(mcts_pi_policy(
                        mcts_factory.clone(),
                        prev_iteration,
                        prev_best_uct,
                    ))
                };

                for &n_mcts in &n_mcts_grid {
                    let pol_arr = repeat_policy_remaining(n_mcts, &cur_pol, &prev_pol);

                    let p1 = pol_arr[0].name().clone();
                    let p2 = pol_arr[1].name().clone();
                    let p3 = pol_arr[2].name().clone();
                    let p4 = pol_arr[3].name().clone();

                    for _ in 0..games_per_matchup {
                        games.push(GameToPlay {
                            game_id: id_counter,
                            iteration,
                            uct_param: uct,
                            uct_key: uct_key.clone(),
                            prev_iteration,
                            prev_best_uct,
                            num_mcts_players: n_mcts,
                            policies: pol_arr.clone(),
                            player_1_name: p1.clone(),
                            player_2_name: p2.clone(),
                            player_3_name: p3.clone(),
                            player_4_name: p4.clone(),
                        });
                        id_counter += 1;
                    }
                }
            }

            games.shuffle(&mut SmallRng::from_os_rng());

            let queue = Arc::new(Mutex::new(VecDeque::from(games)));
            let finished = Arc::new(AtomicUsize::new(0));
            let mut join_set = tokio::task::JoinSet::new();

            for _ in 0..(cores+2) {
                let queue = Arc::clone(&queue);
                let sums = Arc::clone(&sums);
                let csv = csv_writer.clone();
                let finished = Arc::clone(&finished);

                join_set.spawn(async move {
                    let mut rng_local = SmallRng::from_os_rng();
                    let mut rng_game = SmallRng::from_os_rng();

                    while let Some(game) = {
                        let mut lock = queue.lock().await;
                        lock.pop_front()
                    } {
                        let result = full_doko_evaluate_single_game_impi(
                            game.policies.clone(),
                            &mut rng_local,
                            &mut rng_game,
                            false,
                            false
                        )
                            .await;

                        {
                            let mut map = sums.lock().await;
                            if game.num_mcts_players == 1 {
                                *map.entry(game.uct_key.clone())
                                    .or_insert(0) += result.points[1];
                            } else if game.num_mcts_players == 2 {
                                *map.entry(game.uct_key.clone())
                                    .or_insert(0) += result.points[2];
                            } else if game.num_mcts_players == 3 {
                                *map.entry(game.uct_key.clone())
                                    .or_insert(0) += result.points[3];
                            }
                        }

                        csv.write_row(vec![
                            game.iteration.to_string(),
                            game.uct_param.to_string(),
                            game.prev_iteration.to_string(),
                            game.prev_best_uct.to_string(),
                            game.num_mcts_players.to_string(),
                            game.game_id.to_string(),
                            game.player_1_name.clone(), game.player_2_name.clone(),
                            game.player_3_name.clone(), game.player_4_name.clone(),
                            result.points[0].to_string(), result.points[1].to_string(),
                            result.points[2].to_string(), result.points[3].to_string(),
                            result.number_executed_actions[0].to_string(),
                            result.number_executed_actions[1].to_string(),
                            result.number_executed_actions[2].to_string(),
                            result.number_executed_actions[3].to_string(),
                            result.number_of_actions.to_string(),
                            result.played_game_mode.to_string(),
                            result.lowest_announcement_re.map(|x| x.to_string()).unwrap_or_default(),
                            result.lowest_announcement_contra.map(|x| x.to_string()).unwrap_or_default(),
                            result.branching_factor.to_string(),
                            result.reservation_made[0].to_string(),
                            result.reservation_made[1].to_string(),
                            result.reservation_made[2].to_string(),
                            result.reservation_made[3].to_string(),
                            result.lowest_announcement_made[0].map(|x| x.to_string()).unwrap_or_default(),
                            result.lowest_announcement_made[1].map(|x| x.to_string()).unwrap_or_default(),
                            result.lowest_announcement_made[2].map(|x| x.to_string()).unwrap_or_default(),
                            result.lowest_announcement_made[3].map(|x| x.to_string()).unwrap_or_default(),
                            result.number_consistent[0].to_string(),
                            result.number_consistent[1].to_string(),
                            result.number_consistent[2].to_string(),
                            result.number_consistent[3].to_string(),
                            result.number_not_consistent[0].to_string(),
                            result.number_not_consistent[1].to_string(),
                            result.number_not_consistent[2].to_string(),
                            result.number_not_consistent[3].to_string(),
                            result.number_was_random_because_not_consistent[0].to_string(),
                            result.number_was_random_because_not_consistent[1].to_string(),
                            result.number_was_random_because_not_consistent[2].to_string(),
                            result.number_was_random_because_not_consistent[3].to_string(),
                            result.number_not_consistentHandSizeMismatch[0].to_string(),
                            result.number_not_consistentHandSizeMismatch[1].to_string(),
                            result.number_not_consistentHandSizeMismatch[2].to_string(),
                            result.number_not_consistentHandSizeMismatch[3].to_string(),
                            result.number_not_consistentRemainingCardsLeft[0].to_string(),
                            result.number_not_consistentRemainingCardsLeft[1].to_string(),
                            result.number_not_consistentRemainingCardsLeft[2].to_string(),
                            result.number_not_consistentRemainingCardsLeft[3].to_string(),
                            result.number_not_consistentNotInRemainingCards[0].to_string(),
                            result.number_not_consistentNotInRemainingCards[1].to_string(),
                            result.number_not_consistentNotInRemainingCards[2].to_string(),
                            result.number_not_consistentNotInRemainingCards[3].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[0].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[1].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[2].to_string(),
                            result.number_not_consistentAlreadyDiscardedColor[3].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[0].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[1].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[2].to_string(),
                            result.number_not_consistentHasClubQueenButSomeoneElseAnnouncedWedding[3].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[0].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[1].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[2].to_string(),
                            result.number_not_consistentHasNoClubQueenButAnnouncedRe[3].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[0].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[1].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[2].to_string(),
                            result.number_not_consistentHasClubQueenButAnnouncedKontra[3].to_string(),
                            result.number_not_consistentWrongReservation[0].to_string(),
                            result.number_not_consistentWrongReservation[1].to_string(),
                            result.number_not_consistentWrongReservation[2].to_string(),
                            result.number_not_consistentWrongReservation[3].to_string(),
                            result.number_not_consistentWrongReservationClubQ[0].to_string(),
                            result.number_not_consistentWrongReservationClubQ[1].to_string(),
                            result.number_not_consistentWrongReservationClubQ[2].to_string(),
                            result.number_not_consistentWrongReservationClubQ[3].to_string(),
                            result.player_was_re[0].to_string(),
                            result.player_was_re[1].to_string(),
                            result.player_was_re[2].to_string(),
                            result.player_was_re[3].to_string(),
                        ]);

                        let finished_now = finished.fetch_add(1, Ordering::Relaxed) + 1;
                        if finished_now % 10_000 == 0 {
                            println!("   – {finished_now}/{total_games} Spiele erledigt");
                        }
                    }
                });
            }

            while let Some(res) = join_set.join_next().await {
                if let Err(err) = res {
                    eprintln!("Worker-Task Fehler: {err:?}");
                }
            }

            let best_uct_key = {
                let m = sums.lock().await;
                m.iter().max_by_key(|(_, pts)| *pts).map(|(k, _)| k.clone()).unwrap()
            };
            let best_uct: f32 = best_uct_key.parse().unwrap();

            println!("╰─ Iteration {iteration}: bester UCT = {best_uct_key}");

            prev_iteration = iteration;
            prev_best_uct = best_uct;
        }

        csv_writer.finish();
    });
}
