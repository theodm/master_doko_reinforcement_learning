use enumset::EnumSet;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::game_type::game_type::FdoGameType;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::matching::gather_impossible_colors::gather_impossible_colors;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::player::player_set::FdoPlayerSet;
use rs_full_doko::reservation::reservation::FdoReservation;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::util::po_arr::PlayerOrientedArr;
use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;

macro_rules! debug_println {
    ($($arg:tt)*) => {
        // if cfg!(debug_assertions) {
        //     println!($($arg)*);
        // }
    };
}


/// Gibt an, welche Karten der Spieler fdo_player noch in
/// seiner Hand haben kann.
pub fn next_possible_card(
    _state: &FdoState,
    obs: &FdoObservation,

    assumed_hands: PlayerZeroOrientedArr<FdoHand>,
    assumed_reservations: PlayerOrientedArr<Option<FdoReservation>>,

    for_player: FdoPlayer
) -> FdoHand {
    let real_hands = obs.phi_real_hands;
    let game_type = obs.game_type;
    let re_players = obs.phi_re_players;
    let previous_tricks = obs.tricks.clone();
    let visible_reservations = obs.visible_reservations;
    let game_start_player = obs.game_starting_player;

    assert!(real_hands[obs.observing_player] == assumed_hands[obs.observing_player]);
    assert!(for_player != obs.observing_player);

    // Verbleibende Karten sind alle Karten, die die Spieler noch auf der Hand haben
    // abzüglich der eigenen Karten sowie der Karten, die wir für die anderen Spieler
    // annehmen.
    let mut remaining_cards = FdoHand::empty();

    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::BOTTOM]);
    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::LEFT]);
    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::TOP]);
    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::RIGHT]);

    remaining_cards = remaining_cards.minus_hand(assumed_hands[FdoPlayer::BOTTOM]);
    remaining_cards = remaining_cards.minus_hand(assumed_hands[FdoPlayer::LEFT]);
    remaining_cards = remaining_cards.minus_hand(assumed_hands[FdoPlayer::TOP]);
    remaining_cards = remaining_cards.minus_hand(assumed_hands[FdoPlayer::RIGHT]);

    let mut possible_cards = remaining_cards;

    // Karten von einer Farbe, die der Spieler bereits abgeworfen hat,
    // können nicht mehr auf der Hand sein.
    if let Some(game_type) = game_type {
        let impossible_colors = gather_impossible_colors(
            &previous_tricks,
            Some(game_type)
        );

        for imp_color in impossible_colors[for_player] {
            possible_cards.remove_color(imp_color, game_type);
        }
    }

    // Wenn jemand anderes eine Hochzeit angesagt (und der Spieler davon weiß) hat,
    // dann kann der Spieler keine Kreuz-Damen mehr haben.
    for player in FdoPlayerSet::all().iter() {
        if player == for_player {
            continue;
        }

        if assumed_reservations[player] == Some(FdoReservation::Wedding) {
            possible_cards.remove_ignore(FdoCard::ClubQueen);
            possible_cards.remove_ignore(FdoCard::ClubQueen);
        }
    }

    if let Some(re_players) = re_players {
        match game_type {
            Some(FdoGameType::Normal) => {
                let mut player_already_played_q_club = PlayerZeroOrientedArr::from_full([
                    false,
                    false,
                    false,
                    false
                ]);

                for trick in previous_tricks.iter() {
                    for (player, card) in trick.iter_with_player() {
                        if *card == FdoCard::ClubQueen {
                            player_already_played_q_club[player] = true;
                        }
                    }
                }

                let mut removed_q_club_for_player = PlayerZeroOrientedArr::from_full([
                    false,
                    false,
                    false,
                    false
                ]);

                for announcement in obs.announcements.iter() {
                    // Ein anderer Spieler hat Re angesagt, dann müssen wir seine
                    // Kreuz-Dame entfernen, falls er sie noch nicht gespielt hat.
                    if re_players.contains(announcement.player)
                        && announcement.player != for_player
                        && !assumed_hands[announcement.player].contains(FdoCard::ClubQueen)
                        && !removed_q_club_for_player[announcement.player]
                    {
                        if !player_already_played_q_club[announcement.player] {
                            possible_cards.remove_ignore(FdoCard::ClubQueen);
                            removed_q_club_for_player[announcement.player] = true;
                        }
                    }

                    // Der Spieler hat Kontra angesagt, dann hat der Spieler
                    // keine Kreuz-Dame.
                    if !re_players.contains(announcement.player) && announcement.player == for_player {
                        possible_cards.remove_ignore(FdoCard::ClubQueen);
                        possible_cards.remove_ignore(FdoCard::ClubQueen);
                    }
                }
            }
            _ => {}
        }
    }

    return possible_cards;
}

pub fn next_possible_reservation(
    _state: &FdoState,
    obs: &FdoObservation,

    assumed_hands: PlayerZeroOrientedArr<FdoHand>,

    for_player: FdoPlayer
) -> EnumSet<FdoReservation> {
    let previous_tricks = obs.tricks.clone();

    // Wir können eine Hochzeit nur annehmen, wenn der Spieler
    // beide Kreuz-Damen hat oder wenigstens hatte.
    let mut player_already_played_q_club = PlayerZeroOrientedArr::from_full([
        0,
        0,
        0,
        0
    ]);

    for trick in previous_tricks.iter() {
        for (player, card) in trick.iter_with_player() {
            if *card == FdoCard::ClubQueen {
                player_already_played_q_club[player] += 1;
            }
        }
    }

    let mut wedding_allowed = false;

    if assumed_hands[for_player].contains_both(FdoCard::ClubQueen) {
        wedding_allowed = true;
    } else if assumed_hands[for_player].contains(FdoCard::ClubQueen) && player_already_played_q_club[for_player] == 1 {
        wedding_allowed = true;
    } else if player_already_played_q_club[for_player] == 2 {
        wedding_allowed = true;
    }

    let mut allowed_reservations = EnumSet::empty();

    if wedding_allowed {
        allowed_reservations.insert(FdoReservation::Wedding);
    }
    allowed_reservations.insert(FdoReservation::DiamondsSolo);
    allowed_reservations.insert(FdoReservation::HeartsSolo);
    allowed_reservations.insert(FdoReservation::SpadesSolo);
    allowed_reservations.insert(FdoReservation::ClubsSolo);
    allowed_reservations.insert(FdoReservation::TrumplessSolo);
    allowed_reservations.insert(FdoReservation::QueensSolo);
    allowed_reservations.insert(FdoReservation::JacksSolo);
    allowed_reservations.insert(FdoReservation::Healthy);

    return allowed_reservations;
}

#[cfg(test)]
mod tests {
    use rand::prelude::{IndexedRandom, SmallRng};
    use rand::{Rng, SeedableRng};
    use rs_doko_networks::full_doko::var1::encode_ipi::encode_state_ipi;
    use rs_doko_networks::full_doko::var1::ipi_output::ImperfectInformationOutput;
    use rs_full_doko::action::action::FdoAction;
    use rs_full_doko::state::state::FdoState;
    use rs_full_doko::display::display::display_game;
    use rs_full_doko::matching::card_matching::card_matching;
    use rs_full_doko::matching::is_consistent::is_consistent;
    use rs_full_doko::reservation::reservation::FdoVisibleReservation;
    use crate::doko_ext::next_consistent::{next_possible_card, next_possible_reservation};

    pub fn reverse_pred_process(
        state: &FdoState,

        rng: &mut SmallRng
    ) {
        let obs = state
            .observation_for_current_player();
        let current_player = obs
            .current_player
            .unwrap();

        let mut hands = obs
            .phi_real_hands
            .clone();

        let mut reservations = obs
            .phi_real_reservations
            .reservations
            .to_zero_array_remaining_option()
            .clone();

        for i in 0..3 {
            let i = 3 - i;

            let player_to_guess = current_player + i;

            // Das Target, welches das neuronale Netzwerk ausgeben muss. Also
            // der tatsächlich zu erratende Vorbehalt.
            let target = reservations[player_to_guess];
            // Damit es was zu erraten gibt, setzen wir den Wert auf None :)
            reservations[player_to_guess] = None;

            let npo = next_possible_reservation(
                state,
                &obs,

                hands,

                player_to_guess
            );

            if target.is_some() {
                assert!(npo.contains(target.unwrap()));
            }
        }

        for hand_card_index in 0..12 {
            for i in 0..3 {
                let i = 3 - i;

                let player_to_guess = current_player + i;

                if hands[player_to_guess].len() == 0 {
                    continue;
                }

                // Das Target, welches das neuronale Netzwerk ausgeben  muss. Also
                // der tatsächlich zu erratende Vorbehalt.
                let target: Vec<_> = hands[player_to_guess]
                    .iter()
                    .collect();

                let target = target
                    .choose(rng)
                    .unwrap();

                hands[player_to_guess].remove(*target);

                let remaining_cards = obs
                    .phi_real_hands[player_to_guess]
                    .minus_hand(hands[player_to_guess]);

                let npo = next_possible_card(
                    state,
                    &obs,

                    hands.clone(),
                    reservations.to_oriented_arr().clone(),

                    player_to_guess
                );

                if !npo.contains(*target) {
                    println!("Für Spieler {:?}", player_to_guess);
                    println!("Folgende Karten können laut Methode noch vorhanden sein: {:?}", npo);
                    println!("Die Karte {:?} ist nicht in der Liste der möglichen Karten.", target);
                    println!("Spiel: {}", display_game(state.observation_for_current_player()));
                }

                assert!(npo.contains(*target));
                for card in remaining_cards.iter() {
                    assert!(npo.contains(card));
                }

            }
        }
    }

    #[test]
    fn test_next() {
        let mut rng = rand::prelude::SmallRng::from_os_rng();

        for i in 0..1000 {
            let mut state = FdoState::new_game(&mut rng);

            loop {
                let obs = state.observation_for_current_player();

                if obs.finished_stats.is_some() {
                    break;
                }

                reverse_pred_process(
                    &state,
                    &mut rng
                );

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
}