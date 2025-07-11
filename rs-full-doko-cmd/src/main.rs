use std::fs::File;
use std::io::Write;
use dialoguer::Select;
use rand::prelude::{IndexedRandom, SmallRng};
use rand::{Rng, SeedableRng};
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::basic::color::FdoColor::Club;
use rs_full_doko::card::cards::FdoCard::{ClubAce, ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondJack, DiamondKing, DiamondNine, DiamondQueen, DiamondTen, HeartAce, HeartJack, HeartKing, HeartNine, HeartQueen, HeartTen, SpadeAce, SpadeJack, SpadeKing, SpadeNine, SpadeQueen, SpadeTen};
use rs_full_doko::hand::hand::{ FdoHand};
use rs_full_doko::matching::card_matching::card_matching;
use rs_full_doko::matching::is_consistent::is_consistent;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::state::state::FdoState;

fn main() {
    let mut rng = SmallRng::from_os_rng();

    // https://www.online-doppelkopf.com/spiele/99.585.453
    // let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartTen, DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
    //         // Top
    //         FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
    //         // Right
    //         FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );

    // Soli, viele Ansagen
    // https://www.online-doppelkopf.com/spiele/95.882.010
    // let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![SpadeAce, ClubKing, ClubTen, DiamondJack, SpadeQueen, DiamondNine, DiamondNine, DiamondTen, DiamondKing, SpadeNine, HeartKing, HeartAce]),
    //         // Left
    //         FdoHand::from_vec(vec![ClubKing, SpadeQueen, ClubQueen, ClubQueen, HeartTen, HeartQueen, HeartTen, DiamondQueen, ClubNine, SpadeJack, HeartJack, DiamondJack]),
    //         // Top
    //         FdoHand::from_vec(vec![SpadeTen, HeartJack, SpadeJack, ClubJack, ClubTen, ClubJack, ClubAce, HeartQueen, DiamondAce, SpadeTen, SpadeAce, DiamondAce]),
    //         // Right
    //         FdoHand::from_vec(vec![SpadeNine, ClubAce, ClubNine, DiamondQueen, HeartNine, HeartNine, DiamondKing, DiamondTen, SpadeKing, SpadeKing, HeartKing, HeartAce]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );

    // Hochzeit
    // https://www.online-doppelkopf.com/spiele/99.589.890
    // let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartTen, DiamondJack, ClubKing, SpadeKing, HeartNine, HeartKing, SpadeTen, HeartQueen, SpadeTen, DiamondTen, DiamondQueen, ClubJack]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondKing, DiamondKing, ClubTen, SpadeAce, ClubKing, DiamondTen, ClubNine, HeartJack, SpadeJack, ClubJack, DiamondQueen, HeartQueen]),
    //         // Top
    //         FdoHand::from_vec(vec![DiamondNine, HeartTen, ClubAce, SpadeAce, HeartAce, HeartKing, ClubTen, SpadeQueen, SpadeKing, DiamondNine, DiamondAce, SpadeNine]),
    //         // Right
    //         FdoHand::from_vec(vec![DiamondJack, HeartJack, ClubNine, SpadeNine, HeartNine, HeartAce, ClubAce, SpadeJack, SpadeQueen, ClubQueen, ClubQueen, DiamondAce]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );

    // Stilles Solo
    // https://www.online-doppelkopf.com/spiele/99.514.292
    // let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartNine, DiamondAce, DiamondJack, HeartJack, SpadeJack, DiamondQueen, ClubJack, HeartAce, SpadeQueen, HeartKing, HeartTen, HeartNine]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondTen, SpadeTen, DiamondTen, ClubNine, DiamondNine, SpadeAce, HeartJack, ClubAce, ClubJack, ClubTen, ClubKing, HeartQueen]),
    //         // Top
    //         FdoHand::from_vec(vec![HeartKing, SpadeNine, ClubQueen, ClubNine, DiamondJack, SpadeNine, ClubQueen, HeartAce, DiamondAce, SpadeTen, ClubAce, SpadeKing]),
    //         // Right
    //         FdoHand::from_vec(vec![SpadeJack, SpadeAce, HeartTen, ClubKing, DiamondQueen, SpadeKing, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, ClubTen, HeartQueen]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );
    //
    // let mut file = File::create("output.txt").unwrap();
    //
    // use std::io::Write;

    // https://www.online-doppelkopf.com/spiele/99.585.453
    // file.write_all(b"let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartTen, DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
    //         // Top
    //         FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
    //         // Right
    //         FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
    //     ],
    //     FdoPlayer::TOP,
    // );\n").unwrap();

    // Soli, viele Ansagen
    // https://www.online-doppelkopf.com/spiele/95.882.010
    // file.write_all(b"let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![SpadeAce, ClubKing, ClubTen, DiamondJack, SpadeQueen, DiamondNine, DiamondNine, DiamondTen, DiamondKing, SpadeNine, HeartKing, HeartAce]),
    //         // Left
    //         FdoHand::from_vec(vec![ClubKing, SpadeQueen, ClubQueen, ClubQueen, HeartTen, HeartQueen, HeartTen, DiamondQueen, ClubNine, SpadeJack, HeartJack, DiamondJack]),
    //         // Top
    //         FdoHand::from_vec(vec![SpadeTen, HeartJack, SpadeJack, ClubJack, ClubTen, ClubJack, ClubAce, HeartQueen, DiamondAce, SpadeTen, SpadeAce, DiamondAce]),
    //         // Right
    //         FdoHand::from_vec(vec![SpadeNine, ClubAce, ClubNine, DiamondQueen, HeartNine, HeartNine, DiamondKing, DiamondTen, SpadeKing, SpadeKing, HeartKing, HeartAce]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );\n").unwrap();

    // Hochzeit
    // https://www.online-doppelkopf.com/spiele/99.589.890
    // file.write_all(b"let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartTen, DiamondJack, ClubKing, SpadeKing, HeartNine, HeartKing, SpadeTen, HeartQueen, SpadeTen, DiamondTen, DiamondQueen, ClubJack]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondKing, DiamondKing, ClubTen, SpadeAce, ClubKing, DiamondTen, ClubNine, HeartJack, SpadeJack, ClubJack, DiamondQueen, HeartQueen]),
    //         // Top
    //         FdoHand::from_vec(vec![DiamondNine, HeartTen, ClubAce, SpadeAce, HeartAce, HeartKing, ClubTen, SpadeQueen, SpadeKing, DiamondNine, DiamondAce, SpadeNine]),
    //         // Right
    //         FdoHand::from_vec(vec![DiamondJack, HeartJack, ClubNine, SpadeNine, HeartNine, HeartAce, ClubAce, SpadeJack, SpadeQueen, ClubQueen, ClubQueen, DiamondAce]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );\n").unwrap();

    // Stilles Solo
    // https://www.online-doppelkopf.com/spiele/99.514.292
    // file.write_all(b"let mut state = FdoState::new_game_from_hand_and_start_player(
    //     [
    //         // Bottom
    //         FdoHand::from_vec(vec![HeartNine, DiamondAce, DiamondJack, HeartJack, SpadeJack, DiamondQueen, ClubJack, HeartAce, SpadeQueen, HeartKing, HeartTen, HeartNine]),
    //         // Left
    //         FdoHand::from_vec(vec![DiamondTen, SpadeTen, DiamondTen, ClubNine, DiamondNine, SpadeAce, HeartJack, ClubAce, ClubJack, ClubTen, ClubKing, HeartQueen]),
    //         // Top
    //         FdoHand::from_vec(vec![HeartKing, SpadeNine, ClubQueen, ClubNine, DiamondJack, SpadeNine, ClubQueen, HeartAce, DiamondAce, SpadeTen, ClubAce, SpadeKing]),
    //         // Right
    //         FdoHand::from_vec(vec![SpadeJack, SpadeAce, HeartTen, ClubKing, DiamondQueen, SpadeKing, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, ClubTen, HeartQueen]),
    //     ],
    //     FdoPlayer::BOTTOM,
    // );\n").unwrap();
    //
    // loop {
    //     let obs = state
    //         .observation_for_current_player();
    //
    //     display_game(obs.clone());
    //
    //     let obs_str = format!("{:?}", obs);
    //
    //     // replace "tricks: [" with "tricks: heapless::Vec::from_slice(&["
    //     let obs_str = obs_str.replace("tricks: [", "tricks: heapless::Vec::from_slice(&[");
    //     // replace "], visible_reservations" with "]), visible_reservations"
    //     let obs_str = obs_str.replace("], visible_reservations", "]).unwrap(), visible_reservations");
    //     // replace "announcements: [" with "announcements: heapless::Vec::from_slice(&["
    //     let obs_str = obs_str.replace("announcements: [", "announcements: heapless::Vec::from_slice(&[");
    //     // replace "], player_eyes" with "]), player_eyes"
    //     let obs_str = obs_str.replace("], player_eyes", "]).unwrap(), player_eyes");
    //
    //     // Write observation to file
    //     file.write_all(format!("assert_eq!(state.observation_for_current_player(), {});\n\n", obs_str).as_bytes()).unwrap();
    //
    //     if obs.finished_stats.is_some() {
    //         println!("Game is over!");
    //         break;
    //     }
    //
    //     println!("Card Index: {}", obs.tricks.iter().map(|trick| trick.len()).sum::<usize>());
    //     println!("Current player: {:?}", obs.current_player);
    //
    //     let allowed_actions = obs.allowed_actions_current_player;
    //
    //     let selection = Select::new()
    //         .with_prompt("What do you choose?")
    //         .items(&allowed_actions.to_vec())
    //         .interact()
    //         .unwrap();
    //
    //     file.write_all(format!("state.play_action({:?});\n", allowed_actions.to_vec()[selection]).as_bytes()).unwrap();
    //
    //     state
    //         .play_action(allowed_actions.to_vec()[selection]);
    // }

    println!("Testing card matching...");
    let mut rng = rand::prelude::SmallRng::seed_from_u64(1711);

    // Wir testen einfach zufällig, ob
    // das Card-Matching immer konsistente Ergebnisse liefert.
    for i in 0..1500 {
        println!("Iteration: {}", i);

        let mut state = FdoState::new_game(&mut rng);

        loop {
            let obs = state.observation_for_current_player();

            if obs.finished_stats.is_some() {
                break;
            }

            for j in 0..10 {
                let c_state = card_matching(
                    &state,
                    &obs,
                    &mut rng
                );

                let _is_consistent = is_consistent(&state, &obs, c_state.0, c_state.1.to_oriented_arr());

                if !_is_consistent {
                    println!("Iteration: {}", i);
                    println!("State: {:?}", &c_state.0);
                    println!("Reservations: {:?}", c_state.1);

                    println!("Real Hands: {:?}", &obs.phi_real_hands);
                    println!("Game Start Player: {:?}", obs.game_starting_player);
                    println!("Real reservations: {:?}", obs.phi_real_reservations);
                    println!("Visible reservations: {:?}", obs.visible_reservations);
                }

                assert!(is_consistent(&state, &obs, c_state.0, c_state.1.to_oriented_arr()));
            }

            // Randomly sample an action
            let action = *obs
                .allowed_actions_current_player
                .to_vec()
                .choose_weighted(&mut rng, |action| {
                    match action {
                        FdoAction::CardDiamondNine => 20,
                        FdoAction::CardDiamondTen => 20,
                        FdoAction::CardDiamondJack => 20,
                        FdoAction::CardDiamondQueen => 20,
                        FdoAction::CardDiamondKing => 20,
                        FdoAction::CardDiamondAce => 20,
                        FdoAction::CardHeartNine => 20,
                        FdoAction::CardHeartTen => 20,
                        FdoAction::CardHeartJack => 20,
                        FdoAction::CardHeartQueen => 20,
                        FdoAction::CardHeartKing => 20,
                        FdoAction::CardHeartAce => 20,
                        FdoAction::CardClubNine => 20,
                        FdoAction::CardClubTen => 20,
                        FdoAction::CardClubJack => 20,
                        FdoAction::CardClubQueen => 20,
                        FdoAction::CardClubKing => 20,
                        FdoAction::CardClubAce => 20,
                        FdoAction::CardSpadeNine => 20,
                        FdoAction::CardSpadeTen => 20,
                        FdoAction::CardSpadeJack => 20,
                        FdoAction::CardSpadeQueen => 20,
                        FdoAction::CardSpadeKing => 20,
                        FdoAction::CardSpadeAce => 20,
                        FdoAction::ReservationHealthy => 20,
                        FdoAction::ReservationWedding => 4,
                        FdoAction::ReservationDiamondsSolo => 2,
                        FdoAction::ReservationHeartsSolo => 2,
                        FdoAction::ReservationSpadesSolo => 2,
                        FdoAction::ReservationClubsSolo => 2,
                        FdoAction::ReservationTrumplessSolo => 2,
                        FdoAction::ReservationQueensSolo => 2,
                        FdoAction::ReservationJacksSolo => 2,
                        FdoAction::AnnouncementReContra => 1,
                        FdoAction::AnnouncementNo90 => 1,
                        FdoAction::AnnouncementNo60 => 1,
                        FdoAction::AnnouncementNo30 => 1,
                        FdoAction::AnnouncementBlack => 1,
                        FdoAction::NoAnnouncement => 20
                    }
                })
                .unwrap();

            state.play_action(action.clone());
        }


    }
}
