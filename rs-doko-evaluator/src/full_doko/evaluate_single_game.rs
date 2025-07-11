use std::collections::HashMap;
use std::sync::Arc;
use strum::EnumCount;
use strum_macros::EnumIter;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::announcement::announcement::FdoAnnouncement;
use rs_full_doko::basic::phase::FdoPhase;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::game_type::game_type::FdoGameType;
use rs_full_doko::player::player_set::FdoPlayerSet;
use rs_full_doko::reservation::reservation::FdoReservation;
use rs_full_doko::state::state::FdoState;
use crate::full_doko::policy::policy::EvFullDokoPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum FdoPlayedGameType {
    Normal,

    QuietSolo,
    Wedding,

    DiamondsSolo,
    HeartsSolo,
    SpadesSolo,
    ClubsSolo,

    TrumplessSolo,
    QueensSolo,
    JacksSolo
}

impl ToString for FdoPlayedGameType {
    fn to_string(&self) -> String {
        match self {
            FdoPlayedGameType::Normal => "Normalspiel".to_string(),
            FdoPlayedGameType::QuietSolo => "Stilles Solo".to_string(),
            FdoPlayedGameType::Wedding => "Hochzeit".to_string(),
            FdoPlayedGameType::DiamondsSolo => "Diamond-Solo".to_string(),
            FdoPlayedGameType::HeartsSolo => "Heart-Solo".to_string(),
            FdoPlayedGameType::SpadesSolo => "Spade-Solo".to_string(),
            FdoPlayedGameType::ClubsSolo => "Club-Solo".to_string(),
            FdoPlayedGameType::TrumplessSolo => "Fleischloser".to_string(),
            FdoPlayedGameType::QueensSolo => "Damen-Solo".to_string(),
            FdoPlayedGameType::JacksSolo => "Buben-Solo".to_string()
        }
    }
}

#[derive(Debug)]
pub struct EvFullDokoSingleGameEvaluationResult {
    // Spielergebnisse
    pub points: [i32; 4],

    // Gesamte Ausführungszeit der Policy (meist unintressant, viel interessanter ist die durchschnittliche Ausführungszeit)
    pub total_execution_time: [f64; 4],

    // Durchschnittliche Ausführungszeit der Policy
    pub avg_execution_time: [f64; 4],

    // Wie oft wurde die Policy tatächlich ausgeführt. Wenn sie aufgrund Optimierungen nicht ausgeführt wurde, wird sie nicht gezählt.
    pub number_executed_actions: [i32; 4],

    // Wie oft wurden Aktionen ausgeführt (wieviele Züge gab es). Auch wenn die Policy nicht ausgeführt wurde, wird die Aktion gezählt.
    pub number_of_actions: i32,

    pub played_game_mode: FdoPlayedGameType,

    pub lowest_announcement_re: Option<FdoAnnouncement>,
    pub lowest_announcement_contra: Option<FdoAnnouncement>,

    pub reservation_made: [FdoReservation; 4],

    pub lowest_announcement_made: [Option<FdoAnnouncement>; 4],
    pub branching_factor: f64,
}

fn determine_lowest_announcement_made(state: &FdoState) -> [Option<FdoAnnouncement>; 4] {
    let mut result = [None; 4];

    for ann in state
        .announcements
        .clone()
        .announcements {
        result[ann.player.index()] = Some(ann.announcement);
    }

    result
}


fn determine_played_game_mode(
    state: &FdoState,

    player_has_both_club_queens: [bool; 4]
) -> FdoPlayedGameType {
    let game_type = state
        .observation_for_current_player()
        .game_type;

    match game_type.unwrap() {
        FdoGameType::Normal => {
            for player in FdoPlayerSet::all().iter() {
                if player_has_both_club_queens[player.index()] {
                    return FdoPlayedGameType::QuietSolo;
                }
            }

            FdoPlayedGameType::Normal
        },
        FdoGameType::DiamondsSolo => FdoPlayedGameType::DiamondsSolo,
        FdoGameType::HeartsSolo => FdoPlayedGameType::HeartsSolo,
        FdoGameType::SpadesSolo => FdoPlayedGameType::SpadesSolo,
        FdoGameType::ClubsSolo => FdoPlayedGameType::ClubsSolo,
        FdoGameType::TrumplessSolo => FdoPlayedGameType::TrumplessSolo,
        FdoGameType::QueensSolo => FdoPlayedGameType::QueensSolo,
        FdoGameType::JacksSolo => FdoPlayedGameType::JacksSolo,
        FdoGameType::Wedding => FdoPlayedGameType::Wedding
    }
}

/// Führt ein einzelnes Spiel mit den gegebenen Policies aus und gibt das Ergebnis zurück.
///
/// Dabei werden folgende Werte zurückgegeben:
///
/// - Die Punkte der Spieler
/// - Die gesamte Ausführungszeit der Policies
/// - Die durchschnittliche Ausführungszeit der Policies
/// - Die Anzahl der ausgeführten Aktionen
pub async fn full_doko_evaluate_single_game(
    policies: [Arc<dyn EvFullDokoPolicy>; 4],

    create_game_rng: &mut rand::rngs::SmallRng,

    // Dieser RNG wird für alle anderen Zufallsentscheidungen, z.B. die der Policies
    // und deren internes Verhalten, verwendet.
    rng: &mut rand::rngs::SmallRng
) -> EvFullDokoSingleGameEvaluationResult {
    // Aktueller Zustand des Spiels
    let mut state = FdoState::new_game(create_game_rng);

    let mut execution_times: [f64; 4] = [0.0; 4];
    let mut number_executed_actions: [i32; 4] = [0; 4];
    let mut number_of_actions = 0;

    let mut has_both_club_queens: [bool; 4] = [false; 4];

    for player in FdoPlayerSet::all().iter() {
        if state
            .hands[player]
            .contains_both(FdoCard::ClubQueen) {
            has_both_club_queens[player.index()] = true;
        }
    }

    let mut number_of_available_actions = 0;
    let mut number_of_available_actions_num = 0;

    loop {
        let current_observation = state
            .observation_for_current_player();

        // Bis das Spiel beendet ist.
        if current_observation.phase == FdoPhase::Finished {
            // display_game(current_observation);
            break;
        }

        let current_player = current_observation
            .current_player
            .unwrap();

        number_of_available_actions += current_observation
            .allowed_actions_current_player
            .len();
        number_of_available_actions_num += 1;

        // Optimierung:
        //
        // Wenn es nur eine mögliche Aktion des Spielers gibt,
        // dann brauchen wir nicht die Policy auszuwerten, sondern können sie einfach
        // ausführen. Dann zählt die Zeit nicht hinzu.
        let action: FdoAction = {
            // Wir messen die Zeit der Policy-Ausführung
            let start_time = std::time::Instant::now();

            let action = policies[current_player.index()].evaluate(
                &state,
                &current_observation,
                rng
            ).await;

            let end_time = start_time
                .elapsed()
                .as_secs_f64();

            execution_times[current_player.index()] += end_time;
            number_executed_actions[current_player.index()] += 1;

            action.0
        };

        number_of_actions += 1;
        state.play_action(action);
    }

    // println!("{}", display_game(state.observation_for_current_player()));
    // println!("{:#?}", state.observation_for_current_player().finished_stats.unwrap().basic_winning_point_details);
    // println!("{:#?}", state.observation_for_current_player().finished_stats.unwrap().basic_draw_points_details);
    // println!("{:#?}", state.observation_for_current_player().finished_stats.unwrap().additional_points_details);

    return EvFullDokoSingleGameEvaluationResult {
        points: state
            .observation_for_current_player()
            .finished_stats
            .unwrap()
            .player_points
            .storage,
        total_execution_time: execution_times,
        avg_execution_time: [
            execution_times[0] / number_executed_actions[0] as f64,
            execution_times[1] / number_executed_actions[1] as f64,
            execution_times[2] / number_executed_actions[2] as f64,
            execution_times[3] / number_executed_actions[3] as f64
        ],
        number_executed_actions,
        number_of_actions,

        played_game_mode: determine_played_game_mode(&state, has_both_club_queens),
        lowest_announcement_re: state.announcements.re_lowest_announcement,
        lowest_announcement_contra: state.announcements.contra_lowest_announcement,
        reservation_made: state
            .observation_for_current_player()
            .phi_real_reservations.reservations
            .to_zero_array()
            .storage,
        lowest_announcement_made: determine_lowest_announcement_made(&state),

        branching_factor: number_of_available_actions as f64 / number_of_available_actions_num as f64
    };
}