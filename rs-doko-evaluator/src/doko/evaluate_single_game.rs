use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::allowed_actions_len;
use rs_doko::basic::phase::DoPhase;
use rs_doko::state::state::DoState;
use std::intrinsics::transmute;
use rs_doko_mcts::env::envs::env_state_doko::McDokoEnvState;
use rs_doko_mcts::mcts::mcts::CachedMCTS;
use crate::doko::policy::policy::EvDokoPolicy;

#[derive(Debug)]
pub struct EvDokoSingleGameEvaluationResult {
    // Spielergebnisse
    pub points: [i32; 4],

    // Gesamte Ausführungszeit der Policy (meist unintressant, viel interessanter ist die durchschnittliche Ausführungszeit)
    pub total_execution_time: [f64; 4],

    // Durchschnittliche Ausführungszeit der Policy
    pub avg_execution_time: [f64; 4],

    // Wie oft wurde die Policy tatächlich ausgeführt. Wenn sie aufgrund Optimierungen nicht ausgeführt wurde, wird sie nicht gezählt.
    pub number_executed_actions: [i32; 4],

    // Wie oft wurden Aktionen ausgeführt (wieviele Züge gab es). Auch wenn die Policy nicht ausgeführt wurde, wird die Aktion gezählt.
    pub number_of_actions: i32
}

/// Führt ein einzelnes Spiel mit den gegebenen Policies aus und gibt das Ergebnis zurück.
///
/// Dabei werden folgende Werte zurückgegeben:
///
/// - Die Punkte der Spieler
/// - Die gesamte Ausführungszeit der Policies
/// - Die durchschnittliche Ausführungszeit der Policies
/// - Die Anzahl der ausgeführten Aktionen
pub fn doko_evaluate_single_game(
    policies: [&EvDokoPolicy; 4],

    create_game_rng: &mut rand::rngs::SmallRng,

    // Dieser RNG wird für alle anderen Zufallsentscheidungen, z.B. die der Policies
    // und deren internes Verhalten, verwendet.
    rng: &mut rand::rngs::SmallRng,

    // Gehört nicht hierhin. Aber der MCTS braucht eine UnsafeArena um unnötige Allokationen
    // zu vermeiden. Geht sicher auch schöner
    cached_mcts: &mut CachedMCTS<
        McDokoEnvState,
        DoAction,
        4,
        26
    >
) -> EvDokoSingleGameEvaluationResult {
    // Aktueller Zustand des Spiels
    let mut state = DoState::new_game(create_game_rng);

    let mut execution_times: [f64; 4] = [0.0; 4];
    let mut number_executed_actions: [i32; 4] = [0; 4];
    let mut number_of_actions = 0;

    loop {
        let current_observation = state
            .observation_for_current_player();

        // Bis das Spiel beendet ist.
        if current_observation.phase == DoPhase::Finished {
            break;
        }

        let current_player = current_observation
            .current_player
            .unwrap();

        // Optimierung:
        //
        // Wenn es nur eine mögliche Aktion des Spielers gibt,
        // dann brauchen wir nicht die Policy auszuwerten, sondern können sie einfach
        // ausführen. Dann zählt die Zeit nicht hinzu.
        let action: DoAction = //if allowed_actions_len(current_observation.allowed_actions_current_player) == 1 {
            //unsafe {
            //    transmute(current_observation.allowed_actions_current_player as usize)
         //   }
      //  } else {
        {
            // Wir messen die Zeit der Policy-Ausführung
            let start_time = std::time::Instant::now();

            let action = policies[current_player](
                &state,
                &current_observation,
                rng,
                cached_mcts
            );

            let end_time = start_time
                .elapsed()
                .as_secs_f64();

            execution_times[current_player] += end_time;
            number_executed_actions[current_player] += 1;

            action
        };

        number_of_actions+=1;
        state.play_action(action);
    }

    return EvDokoSingleGameEvaluationResult {
        points: state
            .observation_for_current_player()
            .finished_observation
            .unwrap()
            .player_points,
        total_execution_time: execution_times,
        avg_execution_time: [
            execution_times[0] / number_executed_actions[0] as f64,
            execution_times[1] / number_executed_actions[1] as f64,
            execution_times[2] / number_executed_actions[2] as f64,
            execution_times[3] / number_executed_actions[3] as f64
        ],
        number_executed_actions,
        number_of_actions
    };
}