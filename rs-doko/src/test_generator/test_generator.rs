use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use crate::action::action::DoAction;
use crate::display::display::display_game;
use crate::action::allowed_actions::{allowed_actions_to_vec, DoAllowedActions};
use crate::basic::phase::DoPhase;
use crate::card::cards::DoCard;
use crate::hand::hand::{DoHand, hand_to_vec};
use crate::observation::observation::DoObservation;
use crate::player::player::{DoPlayer, PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
use crate::player::player_set::{DoPlayerSet, player_set_to_vec};
use crate::reservation::reservation::{DoReservation, DoVisibleReservation};
use crate::reservation::reservation_round::DoReservationRound;
use crate::state::state::DoState;
use crate::trick::trick::DoTrick;

fn append_to_file(file_name: &str, content: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_name)
        .unwrap();

    file.write_all(content.as_bytes()).unwrap();
}

fn phase_to_str(phase: DoPhase) -> &'static str {
    match phase {
        DoPhase::Reservation => "DoPhase::Reservation",
        DoPhase::PlayCard => "DoPhase::PlayCard",
        DoPhase::Finished => "DoPhase::Finished"
    }
}

fn player_to_str(player: DoPlayer) -> &'static str {
    match player {
        PLAYER_BOTTOM => "PLAYER_BOTTOM",
        PLAYER_LEFT => "PLAYER_LEFT",
        PLAYER_TOP => "PLAYER_TOP",
        PLAYER_RIGHT => "PLAYER_RIGHT",
        _ => panic!("should not happen")
    }
}

fn optional_player_to_str(player: Option<DoPlayer>) -> String {
    match player {
        Some(player) => player_to_str(player).to_string(),
        None => "None".to_string()
    }
}

fn card_to_str(card: DoCard) -> String {
    match card {
        DoCard::DiamondNine => "DoCard::DiamondNine".to_string(),
        DoCard::DiamondTen => "DoCard::DiamondTen".to_string(),
        DoCard::DiamondJack => "DoCard::DiamondJack".to_string(),
        DoCard::DiamondQueen => "DoCard::DiamondQueen".to_string(),
        DoCard::DiamondKing => "DoCard::DiamondKing".to_string(),
        DoCard::DiamondAce => "DoCard::DiamondAce".to_string(),

        DoCard::HeartNine => "DoCard::HeartNine".to_string(),
        DoCard::HeartTen => "DoCard::HeartTen".to_string(),
        DoCard::HeartJack => "DoCard::HeartJack".to_string(),
        DoCard::HeartQueen => "DoCard::HeartQueen".to_string(),
        DoCard::HeartKing => "DoCard::HeartKing".to_string(),
        DoCard::HeartAce => "DoCard::HeartAce".to_string(),

        DoCard::SpadeNine => "DoCard::SpadeNine".to_string(),
        DoCard::SpadeTen => "DoCard::SpadeTen".to_string(),
        DoCard::SpadeJack => "DoCard::SpadeJack".to_string(),
        DoCard::SpadeQueen => "DoCard::SpadeQueen".to_string(),
        DoCard::SpadeKing => "DoCard::SpadeKing".to_string(),
        DoCard::SpadeAce => "DoCard::SpadeAce".to_string(),

        DoCard::ClubNine => "DoCard::ClubNine".to_string(),
        DoCard::ClubTen => "DoCard::ClubTen".to_string(),
        DoCard::ClubJack => "DoCard::ClubJack".to_string(),
        DoCard::ClubQueen => "DoCard::ClubQueen".to_string(),
        DoCard::ClubKing => "DoCard::ClubKing".to_string(),
        DoCard::ClubAce => "DoCard::ClubAce".to_string(),
    }
}

fn action_to_str(action: DoAction) -> String {
    match action {
        DoAction::CardDiamondNine => "DoAction::CardDiamondNine".to_string(),
        DoAction::CardDiamondTen => "DoAction::CardDiamondTen".to_string(),
        DoAction::CardDiamondJack => "DoAction::CardDiamondJack".to_string(),
        DoAction::CardDiamondQueen => "DoAction::CardDiamondQueen".to_string(),
        DoAction::CardDiamondKing => "DoAction::CardDiamondKing".to_string(),
        DoAction::CardDiamondAce => "DoAction::CardDiamondAce".to_string(),

        DoAction::CardHeartNine => "DoAction::CardHeartNine".to_string(),
        DoAction::CardHeartTen => "DoAction::CardHeartTen".to_string(),
        DoAction::CardHeartJack => "DoAction::CardHeartJack".to_string(),
        DoAction::CardHeartQueen => "DoAction::CardHeartQueen".to_string(),
        DoAction::CardHeartKing => "DoAction::CardHeartKing".to_string(),
        DoAction::CardHeartAce => "DoAction::CardHeartAce".to_string(),

        DoAction::CardSpadeNine => "DoAction::CardSpadeNine".to_string(),
        DoAction::CardSpadeTen => "DoAction::CardSpadeTen".to_string(),
        DoAction::CardSpadeJack => "DoAction::CardSpadeJack".to_string(),
        DoAction::CardSpadeQueen => "DoAction::CardSpadeQueen".to_string(),
        DoAction::CardSpadeKing => "DoAction::CardSpadeKing".to_string(),
        DoAction::CardSpadeAce => "DoAction::CardSpadeAce".to_string(),

        DoAction::CardClubNine => "DoAction::CardClubNine".to_string(),
        DoAction::CardClubTen => "DoAction::CardClubTen".to_string(),
        DoAction::CardClubJack => "DoAction::CardClubJack".to_string(),
        DoAction::CardClubQueen => "DoAction::CardClubQueen".to_string(),
        DoAction::CardClubKing => "DoAction::CardClubKing".to_string(),
        DoAction::CardClubAce => "DoAction::CardClubAce".to_string(),

        DoAction::ReservationHealthy => "DoAction::ReservationHealthy".to_string(),
        DoAction::ReservationWedding => "DoAction::ReservationWedding".to_string()
    }
}

fn map_actions_to_str(actions: DoAllowedActions) -> String {
    let x = allowed_actions_to_vec(actions);

    let comma_seperated_actions: String = x
        .iter()
        .map(|&action| action_to_str(action))
        .collect::<Vec<String>>().join(", ");

    return format!("allowed_actions_from_vec(vec![{}])", comma_seperated_actions);
}

fn map_optional_card_to_str(card: Option<DoCard>) -> String {
    match card {
        Some(card) => card_to_str(card),
        None => "None".to_string()
    }
}

fn map_trick_to_str(trick: Option<DoTrick>) -> String {
    match trick {
        Some(trick) => {
            format!("DoTrick::existing({}, vec![{}])", player_to_str(trick.start_player), trick
                .cards
                .iter()
                .map(|&card| map_optional_card_to_str(card))
                .filter(|card| card != "None")
                .collect::<Vec<String>>().join(", "))
        }
        None => {
            "None".to_string()
        }
    }
}

fn map_tricks_to_str(tricks: [Option<DoTrick>; 12]) -> String {
    let tricks_with_comma = tricks
        .iter()
        .map(|&trick| map_trick_to_str(trick))
        .collect::<Vec<_>>()
        .join(",\n\t");

    return format!("[\n\t{}\n]", tricks_with_comma);
}

fn map_visible_reservation_to_str(
    visible_reservation: Option<DoVisibleReservation>
) -> String {
    match visible_reservation {
        Some(visible_reservation) => {
            match visible_reservation {
                DoVisibleReservation::Wedding => { "Some(DoVisibleReservation::Wedding)".to_string() }
                DoVisibleReservation::Healthy => { "Some(DoVisibleReservation::Healthy)".to_string() }
                DoVisibleReservation::NotRevealed => { "Some(DoVisibleReservation::NotRevealed)".to_string() }
            }
        }
        None => {
            "None".to_string()
        }
    }
}

fn map_visible_reservations_to_str(
    visible_reservations: [Option<DoVisibleReservation>; 4]
) -> String {
    let visible_reservations_with_comma = visible_reservations
        .iter()
        .map(|&visible_reservation| map_visible_reservation_to_str(visible_reservation))
        .collect::<Vec<_>>()
        .join(", ");

    return format!("[{}]", visible_reservations_with_comma);
}

fn map_player_eyes_to_str(player_eyes: [u32; 4]) -> String {
    format!("[{}, {}, {}, {}]", player_eyes[0], player_eyes[1], player_eyes[2], player_eyes[3])
}

fn map_player_hand(player_hand: DoHand) -> String {
    let cards = hand_to_vec(player_hand);

    format!("hand_from_vec(vec![{}])", cards
        .iter()
        .map(|&card| card_to_str(card))
        .collect::<Vec<String>>()
        .join(", "))
}

fn map_player_hands_to_str(player_hands: [DoHand; 4]) -> String {
    let player_hands_with_comma = player_hands
        .iter()
        .map(|&player_hand| map_player_hand(player_hand))
        .collect::<Vec<_>>()
        .join(",\n\t");

    return format!("[\n\t{}\n]", player_hands_with_comma);
}

fn map_player_set_to_str(player_set: DoPlayerSet) -> String {
    player_set_to_vec(player_set)
        .iter()
        .map(|&player| player_to_str(player))
        .collect::<Vec<_>>()
        .join(", ")
}

fn map_re_players_to_str(re_players: Option<DoPlayerSet>) -> String {
    match re_players {
        Some(re_players) => format!("Some(player_set_create(vec![{}]))", player_set_to_vec(re_players).iter().map(|&player| player_to_str(player)).collect::<Vec<_>>().join(", ")),
        None => "None".to_string()
    }
}

fn map_real_reservation_to_str(real_reservations: Option<DoReservation>) -> String {
    match real_reservations {
        None => {
            "None".to_string()
        }
        Some(reservation) => {
            match reservation {
                DoReservation::Healthy => { "Some(DoReservation::Healthy)".to_string() }
                DoReservation::Wedding => { "Some(DoReservation::Wedding)".to_string() }
            }
        }
    }
}

fn map_reservation_round(real_reservations: DoReservationRound) -> String {
    let start_player = player_to_str(real_reservations.start_player);
    let reservations = real_reservations
        .reservations
        .iter()
        .map(|&reservation| map_real_reservation_to_str(reservation))
        .filter(|reservation| reservation != "None")
        .collect::<Vec<_>>()
        .join(", ");

    format!("DoReservationRound::existing({}, vec![{}])", start_player, reservations)
}

pub fn sample_random_action(
    allowed_actions: DoAllowedActions,
    rng: &mut SmallRng,
) -> DoAction {
    let actions = allowed_actions_to_vec(allowed_actions);

    println!("{:?}", actions);

    let index = rng.gen_range(0..actions.len());

    actions[index]
}

fn print_obs(last_observation: &Option<DoObservation>, observation: &DoObservation) {
    match last_observation {
        Some(last_observation) => {
            if last_observation.phase != observation.phase {
                println!("phase: {}", phase_to_str(observation.phase));
            }
            if last_observation.observing_player != observation.observing_player {
                println!("observing_player: {}", player_to_str(observation.observing_player));
            }
            if last_observation.current_player != observation.current_player {
                println!("current_player: {}", optional_player_to_str(observation.current_player));
            }
            if last_observation.allowed_actions_current_player != observation.allowed_actions_current_player {
                println!("allowed_actions_current_player: {}", map_actions_to_str(observation.allowed_actions_current_player));
            }
            if last_observation.game_starting_player != observation.game_starting_player {
                println!("game_starting_player: {}", player_to_str(observation.game_starting_player));
            }
            if last_observation.tricks != observation.tricks {
                println!("tricks: {}", map_tricks_to_str(observation.tricks));
            }
            if last_observation.visible_reservations != observation.visible_reservations {
                println!("visible_reservations: {}", map_visible_reservations_to_str(observation.visible_reservations));
            }
            if last_observation.player_eyes != observation.player_eyes {
                println!("player_eyes: {}", map_player_eyes_to_str(observation.player_eyes));
            }
            if last_observation.observing_player_hand != observation.observing_player_hand {
                println!("observing_player_hand: {}", map_player_hand(observation.observing_player_hand));
            }
            if last_observation.finished_observation != observation.finished_observation {
                println!("finished_observation: {:?}", observation.finished_observation);
            }
            if last_observation.phi_re_players != observation.phi_re_players {
                println!("phi_re_players: {}", map_re_players_to_str(observation.phi_re_players));
            }
            if last_observation.phi_real_reservations != observation.phi_real_reservations {
                println!("phi_real_reservations: {}", map_reservation_round(observation.phi_real_reservations));
            }
            if last_observation.phi_real_hands != observation.phi_real_hands {
                println!("phi_real_hands: {}", map_player_hands_to_str(observation.phi_real_hands));
            }
            println!()
        }
        None => {
            println!("phase: {}", phase_to_str(observation.phase));
            println!("observing_player: {}", player_to_str(observation.observing_player));
            println!("current_player: {}", optional_player_to_str(observation.current_player));
            println!("allowed_actions_current_player: {}", map_actions_to_str(observation.allowed_actions_current_player));
            println!("game_starting_player: {}", player_to_str(observation.game_starting_player));
            println!("tricks: {}", map_tricks_to_str(observation.tricks));
            println!("visible_reservations: {}", map_visible_reservations_to_str(observation.visible_reservations));
            println!("player_eyes: {}", map_player_eyes_to_str(observation.player_eyes));
            println!("observing_player_hand: {}", map_player_hand(observation.observing_player_hand));
            println!("finished_observation: {:?}", observation.finished_observation);
            println!("phi_re_players: {}", map_re_players_to_str(observation.phi_re_players));
            println!("phi_real_reservations: {}", map_reservation_round(observation.phi_real_reservations));
            println!("phi_real_hands: {}", map_player_hands_to_str(observation.phi_real_hands));
            println!();
        }
    }
}


pub fn test_generator() {
    let mut rng = SmallRng::seed_from_u64(999);

    let mut state = DoState::new_game(&mut rng);


    let first_observation = state.observation_for_current_player();

    let mut last_observation: Option<DoObservation> = None;

    append_to_file("test_generator.txt", format!("\
let mut state = DoState::new_game_from_hand_and_start_player(
    [
        {},
        {},
        {},
        {}
    ],
{}
);\n\n",
                                                 map_player_hand(first_observation.phi_real_hands[0]),
                                                 map_player_hand(first_observation.phi_real_hands[1]),
                                                 map_player_hand(first_observation.phi_real_hands[2]),
                                                 map_player_hand(first_observation.phi_real_hands[3]),
                                                 player_to_str(first_observation.game_starting_player)).as_str());


    loop {
        let observation = state.observation_for_current_player();

        print_obs(&last_observation, &observation);

        append_to_file("test_generator.txt", format!("\
let observation = state.observation_for_current_player();

assert_eq!(observation.phase, {});
assert_eq!(observation.observing_player, {});
assert_eq!(observation.current_player, {});
assert_eq!(observation.allowed_actions_current_player, {});
assert_eq!(observation.game_starting_player, {});
assert_eq!(observation.tricks, {});
assert_eq!(observation.visible_reservations, {});
assert_eq!(observation.player_eyes, {});
assert_eq!(observation.observing_player_hand, {});
assert_eq!(observation.finished_observation, {});
assert_eq!(observation.phi_re_players, {});
assert_eq!(observation.phi_real_reservations, {});
assert_eq!(observation.phi_real_hands, {});\n\n",
        phase_to_str(observation.phase),
        player_to_str(observation.observing_player),
        optional_player_to_str(observation.current_player),
         map_actions_to_str(observation.allowed_actions_current_player),
        player_to_str(observation.game_starting_player),
        map_tricks_to_str(observation.tricks),
        map_visible_reservations_to_str(observation.visible_reservations),
        map_player_eyes_to_str(observation.player_eyes),
        map_player_hand(observation.observing_player_hand),
        format!("{:?}", observation.finished_observation),
        map_re_players_to_str(observation.phi_re_players),
        map_reservation_round(observation.phi_real_reservations),
        map_player_hands_to_str(observation.phi_real_hands)
    ).as_str());

        if observation.phase == DoPhase::Finished {
            break;
        }

        // read line to continue
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        last_observation = Some(observation.clone());
        let random_action = sample_random_action(observation.allowed_actions_current_player, &mut rng);

        append_to_file("test_generator.txt", format!("\
    state.play_action({});\
\n\n", action_to_str(random_action)).as_str());

        println!("{} -> play_action({});\n", player_to_str(observation.observing_player), action_to_str(random_action));
        state.play_action(random_action);
    }

    display_game(state.observation_for_current_player());
}
