use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use crate::hand::hand::{FdoHand};
use crate::observation::observation::FdoObservation;
use crate::player::player::FdoPlayer;
use crate::reservation::reservation::{FdoReservation, FdoVisibleReservation};
use crate::state::state::FdoState;
use crate::trick::trick::FdoTrick;
use enumset::EnumSet;
use rand::prelude::{IteratorRandom, SliceRandom, SmallRng};
use std::cmp::PartialEq;
use std::thread::sleep;
use std::time::Duration;
use log::debug;
use crate::matching::gather_impossible_colors;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation_round::FdoReservationRound;
use crate::util::po_arr::PlayerOrientedArr;
use crate::util::po_vec::PlayerOrientedVec;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

macro_rules! debug_println {
    ($($arg:tt)*) => {
        // if cfg!(debug_assertions) {
        //     println!($($arg)*);
        // }
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CardMatchingState {
    observing_player: FdoPlayer,

    /// Die Karten, die noch verteilt werden müssen.
    available_cards: FdoHand,

    /// Gibt an, ob dem Spieler noch eine Kreuz-Dame zugeordnet werden müssen.
    must_have_q_club: PlayerZeroOrientedArr<bool>,
    /// Verbleibende Anzahl an Karten, die noch zugeordnet werden müssen.
    remaining_card_slots: PlayerZeroOrientedArr<usize>,
    /// Wurde dem Spieler zugeordnet.
    card_assignments: PlayerZeroOrientedArr<FdoHand>,
    /// Hat der Spieler möglicherweise.
    possible_cards: PlayerZeroOrientedArr<FdoHand>,
}

impl CardMatchingState {
    pub fn assign_card(
        &mut self,
        player: FdoPlayer,
        card: FdoCard
    ) {
        self.card_assignments[player].add(card);
        self.remaining_card_slots[player] -= 1;

        // Wir entfernen die Karte aus den noch verfügbaren Karten.
        self.available_cards.remove(card);

        // Wir entfernen die Karte aus den möglichen Karten des Spielers.
        self.possible_cards[player].remove(card);

        if self.remaining_card_slots[player] == 0 {
            self.possible_cards[player] = FdoHand::empty();
        }

        // Wir entfernen die Karten aus den möglichen Karten der anderen Spieler.
        for other_player in FdoPlayerSet::all().iter() {
            if other_player != player {
                self.possible_cards[other_player].remove_ignore(card);
            }
        }

        if card == FdoCard::ClubQueen {
            self.must_have_q_club[player] = false;
        }
    }

    fn rule1(
        &mut self
    ) {
        // Wir weisen zuerst die Karten zu, die nur einem Spieler explizit zugeordnet werden können
        // (und keinem anderen). (1. Fall)
        for card in self
            .available_cards
            .iter() {
            // Spieler, welche die Karte haben können.
            let mut single_player = None;

            for player in FdoPlayerSet::all().iter() {
                if player == self.observing_player {
                    continue;
                }

                if self
                    .possible_cards[player]
                    .contains(card) {
                    if single_player.is_some() {
                        // Es gibt mehr als einen Spieler, der die Karte haben kann.
                        single_player = None;
                        break;
                    }

                    single_player = Some(player);
                }
            }

            if let Some(player) = single_player {
                self.assign_card(
                    player, card
                );

                debug_println!("Karte {} wurde Spieler {:?} zugeordnet. (Regel 1)", card, player);
            }
        }
    }

    pub fn rule2(
        &mut self,
    ) -> bool {
        let mut something_changed = false;

        // Falls es einen Spieler gibt, bei dem die Anzahl der verbleibenden
        // Kartenslots der Anzahl der noch ihm zuordenbaren Karten entspricht,
        // dann werden ihm diese Karten zugeordnet. (2. Fall)
        for player in FdoPlayerSet::all().iter() {
            if player == self.observing_player {
                continue;
            }

            if self.remaining_card_slots[player] > 0
                && self.remaining_card_slots[player] == self.possible_cards[player].len()
            {
                something_changed = true;

                for card in self.possible_cards[player].clone().iter() {
                    self.assign_card(
                        player, card
                    );

                    debug_println!("Karte {} wurde Spieler {:?} zugeordnet. (Regel 2)", card, player);
                }

            }
        }

        return something_changed;
    }

    fn rule3(
        &mut self
    ) -> bool {
        let mut something_changed = false;

        // Falls einem Spieler eine Kreuz-Dame zwingend zugeordnet werden muss (wenn er Re ist)
        // dann ordnen wir sie ihm zu. (3. Fall)
        for player in FdoPlayerSet::all().iter() {
            let player = player;

            if player == self.observing_player {
                continue;
            }

            if self.must_have_q_club[player] && self.possible_cards[player].contains(FdoCard::ClubQueen) {
                something_changed = true;

                self.assign_card(
                    player, FdoCard::ClubQueen
                );

                debug_println!("Kreuz-Dame wurde Spieler {:?} zugeordnet. (Regel 3)", player);
            }
        }

        return something_changed;
    }

    fn rule4(
        &mut self,
        rng: &mut SmallRng,
    ) {
        // Ansonsten wählen wir eine verbleibende Karte aus und ordnen sie einem zufälligen Spieler
        // der sie noch brauchen kann, zu. (4. Fall)
        let chosen_card = self
            .available_cards
            .iter()
            .choose(rng)
            .unwrap();

        let player = *FdoPlayerSet::all()
            .iter()
            .filter(|&player| player != self.observing_player)
            .filter(|&player| self.possible_cards[player].contains(chosen_card))
            .collect::<Vec<_>>()
            .first()
            .unwrap();

        if self.possible_cards[player].contains(chosen_card) {
            self.assign_card(
                player, chosen_card
            );

            debug_println!("Karte {} wurde Spieler {:?} zugeordnet. (Regel 4)", chosen_card, player);
        }

    }

    pub fn execute(
        &mut self,
        rng: &mut SmallRng,
    ) {
        loop {
            // Solange es noch Karten zu verteilen gibt:
            if self
                .available_cards
                .len() == 0 {
                break;
            }

            self.rule1();

            if self.rule2() {
                continue;
            }

            if self.rule3() {
                continue;
            }

            if self
                .available_cards
                .len() == 0 {
                break;
            }

            self.rule4(rng);
        }

    }
}

pub fn card_matching(
    // Nimmt tatsächlich einen State entgegen,
    // aber tatsächlich wird nur auf Informationen der Beobachtung
    // zurückgegriffen.
    state: &FdoState,

    obs: &FdoObservation,

    rng: &mut SmallRng,
) -> (PlayerZeroOrientedArr<FdoHand>, PlayerZeroOrientedArr<Option<FdoReservation>>) {
    let current_player = state
        .current_player
        .expect("Das Spiel ist beendet. Ein Card-Matching macht keinen Sinn mehr.");

    debug_println!("Observing Player: {:?}", current_player);

    // Wir berechnen alle noch verfügbaren Karten
    let mut available_cards = FdoHand::empty();

    // Das sind die Karten, die die Spieler in der Hand haben,
    // außer dem Spieler, der gerade an der Reihe ist.
    for player in FdoPlayerSet::all().iter() {
        if player == current_player {
            continue;
        }

        available_cards = available_cards
            .plus_hand(state.hands[player]);
    }

    debug_println!("Verfügbare Karten (zum Verteilen): {}", available_cards);

    // Soviele Kartenslots müssen noch gefüllt werden.
    let mut remaining_card_slots = obs
        .phi_real_hands
        .map(|hand| hand.len());;

    remaining_card_slots[current_player] = 0;

    debug_println!("Verbleibende Kartenslots: {:?}", remaining_card_slots);

    // Wir berechnen von allen Spielern (außer dem aktuellen Spieler)
    // die Karten, die sie nach den bekannten Spielinfos auf der Hand haben könnten.
    let mut possible_cards = PlayerZeroOrientedArr::from_full([
        available_cards.clone(),
        available_cards.clone(),
        available_cards.clone(),
        available_cards.clone(),
    ]);

    // Wenn ein Spieler (sichtbar) eine Hochzeit angesagt hat,
    // dann können die anderen Spieler keine Kreuz-Dame mehr haben.
    for (player, visible_reservation) in obs
        .visible_reservations
        .iter_with_player() {
        match visible_reservation {
            FdoVisibleReservation::Wedding => {
                // Der Spieler hat eine Hochzeit angesagt.
                // Die anderen Spieler können keine Kreuz-Dame mehr haben.
                for other_player in FdoPlayerSet::all().iter() {
                    if other_player != player {
                        possible_cards[other_player].remove_ignore(FdoCard::ClubQueen);
                        possible_cards[other_player].remove_ignore(FdoCard::ClubQueen);

                        debug_println!("Spieler {:?} hat Hochzeit angesagt. Spieler {:?} kann keine Kreuz-Dame haben.", player, other_player);
                    } else {
                        debug_println!("Spieler {:?} hat Hochzeit angesagt.", player);
                    }
                }
            }
            _ => {
                // Keine weitere Aussage möglich.
            }
        }
    }

    possible_cards[current_player] = FdoHand::empty();

    let mut number_of_played_q_clubs = PlayerZeroOrientedArr::from_full([
        0,
        0,
        0,
        0
    ]);

    for trick in &obs.tricks {
        for (player, card) in trick.iter_with_player() {
            if *card == FdoCard::ClubQueen {
                number_of_played_q_clubs[player] += 1;
            }
        }
    }

    let mut must_have_q_club = PlayerZeroOrientedArr::from_full([false, false, false, false]);

    if let Some(game_type) = state.game_type {
        // Die ganzen Regeln gelten nur, sobald schon die Kartenphase erreicht wurde.

        // Wenn ein Spieler eine Farbe nicht bedient hat, dann kann er keine Karte
        // dieser Farbe haben.
        let mut impossible_colors_of_players =
            gather_impossible_colors::gather_impossible_colors(&obs.tricks, state.game_type);

        for player in FdoPlayerSet::all().iter() {
            let player = player;

            if player == current_player {
                continue;
            }

            for color in impossible_colors_of_players[player] {
                // Dann entfernen wir die Farbe aus den möglichen Karten des Spielers.
                possible_cards[player].remove_color(color, game_type);
            }
        }

        // Wenn ein Spieler im Normalspiel bekannterweise Kontra oder Re ist, können wir ihm
        // eine Kreuz-Dame unterstellen oder ausschließen.
        if game_type == FdoGameType::Normal {
            let re_players = obs
                .phi_re_players
                .expect("Im Normalspiel stehen die Re-Spieler fest.");

            for announcement in &obs.announcements {
                let announcement_player = announcement.player;

                if re_players.contains(announcement_player) {
                    // Der Spieler ist Re (und muss wenigstens eine Kreuz-Dame haben).
                    let already_played_q_clubs = number_of_played_q_clubs[announcement_player] > 0;

                    must_have_q_club[announcement_player] = if already_played_q_clubs {
                        false
                    } else {
                        true
                    };
                } else {
                    // Der Spieler ist Kontra (und hat keine Kreuz-Dame).
                    possible_cards[announcement_player].remove_ignore(FdoCard::ClubQueen);
                    possible_cards[announcement_player].remove_ignore(FdoCard::ClubQueen);
                }
            }
        }
    }

    debug_println!("Mögliche Karten: {:?}", possible_cards);

    let mut card_assignments = PlayerZeroOrientedArr::from_full([
        FdoHand::empty(),
        FdoHand::empty(),
        FdoHand::empty(),
        FdoHand::empty(),
    ]);

    for player in FdoPlayerSet::all().iter() {
        if remaining_card_slots[player] == 0 {
            possible_cards[player] = FdoHand::empty();
        }
    }

    let mut card_matching_state = CardMatchingState {
        observing_player: current_player,
        available_cards,
        must_have_q_club,
        remaining_card_slots,
        card_assignments,
        possible_cards,
    };

    card_matching_state.execute(rng);

    let mut card_assignments = card_matching_state
        .card_assignments;

    // Vom aktuellen Spieler nehmen wir das tatsächliche
    // Blatt.
    card_assignments[current_player] = state.hands[current_player];

    // Wir haben alle Karten zugeordnet.

    // Nun erstellen wir noch eine konsistente Menge von Vorbehalten
    // für die Spieler.
    let mut reservations = PlayerZeroOrientedArr::from_full([None; 4]);

    for player in FdoPlayerSet::all().iter() {
        let visible_reservation = obs
            .visible_reservations[player];

        let real_reservation = match visible_reservation {
            FdoVisibleReservation::NoneYet => None,
            visible_reservation => {
                let mut possible_reservations = EnumSet::empty();

                possible_reservations.insert(FdoReservation::DiamondsSolo);
                possible_reservations.insert(FdoReservation::HeartsSolo);
                possible_reservations.insert(FdoReservation::SpadesSolo);
                possible_reservations.insert(FdoReservation::ClubsSolo);
                possible_reservations.insert(FdoReservation::QueensSolo);
                possible_reservations.insert(FdoReservation::JacksSolo);
                possible_reservations.insert(FdoReservation::TrumplessSolo);

                if card_assignments[player].contains_both(FdoCard::ClubQueen)
                    || (card_assignments[player].contains(FdoCard::ClubQueen) && number_of_played_q_clubs[player] == 1)
                    || number_of_played_q_clubs[player] == 2 {
                    possible_reservations.insert(FdoReservation::Wedding);
                }

                match visible_reservation {
                    FdoVisibleReservation::Wedding => Some(FdoReservation::Wedding),
                    FdoVisibleReservation::Healthy => Some(FdoReservation::Healthy),
                    FdoVisibleReservation::NotRevealed => possible_reservations.into_iter().choose(rng),
                    FdoVisibleReservation::DiamondsSolo => Some(FdoReservation::DiamondsSolo),
                    FdoVisibleReservation::HeartsSolo => Some(FdoReservation::HeartsSolo),
                    FdoVisibleReservation::SpadesSolo => Some(FdoReservation::SpadesSolo),
                    FdoVisibleReservation::ClubsSolo => Some(FdoReservation::ClubsSolo),
                    FdoVisibleReservation::QueensSolo => Some(FdoReservation::QueensSolo),
                    FdoVisibleReservation::JacksSolo => Some(FdoReservation::JacksSolo),
                    FdoVisibleReservation::TrumplessSolo => Some(FdoReservation::TrumplessSolo),
                    _ => { panic!("Unbekannter Vorbehalt."); }
                }
            }
        };

        reservations[player] = real_reservation;
    }

    (card_assignments, reservations)
}

pub fn card_matching_full(
    state: &FdoState,
    obs: &FdoObservation,
    rng: &mut SmallRng,
) -> FdoState {
    let (card_assignments, reservations) = card_matching(
        state,
        obs,
        rng
    );

    // Wir erstellen einen neuen State und setzen die Karten und Vorbehalte.
    let mut new_state = state.clone();

    for player in FdoPlayerSet::all().iter() {
        new_state.hands[player] = card_assignments[player].clone();
    }

    new_state.reservations_round = FdoReservationRound::existing(
        new_state.reservations_round.reservations.starting_player,
        reservations
            .rotate_to(new_state.reservations_round.reservations.starting_player)
            .all_present()
            .map(|r| *r)
            .collect()
    );

    new_state
}

#[cfg(test)]
mod tests {
    use rand::prelude::IndexedRandom;
    use rand::{Rng, SeedableRng};
    use crate::action::action::FdoAction;
    use crate::display::display::display_game;
    use crate::matching::card_matching::{card_matching, card_matching_full};
    use crate::matching::is_consistent;
    use crate::matching::is_consistent::is_consistent;
    use crate::state::state::FdoState;

    pub fn output_change(state: &FdoState) {
        let mut rng = rand::prelude::SmallRng::from_os_rng();

        let new_state = card_matching_full(state, &state.observation_for_current_player(), &mut rng);

        println!("========== ORIGINAL STATE ==========");
        println!("{}", display_game(state.observation_for_current_player()));

        println!("========== MATCHED STATE ===========");
        println!("{}", display_game(new_state.observation_for_current_player()));
    }

    #[test]
    fn test_card_matching_with_output() {
        let mut rng = rand::prelude::SmallRng::from_os_rng();
        let mut state = FdoState::new_game(&mut rng);

        loop {
            let obs = state.observation_for_current_player();

            if obs.finished_stats.is_some() {
                break;
            }

            output_change(&state);

            // Dann einfach mal eine Aktion ausführen:
            let action = obs
                .allowed_actions_current_player
                .to_vec()
                .choose(&mut rng)
                .unwrap()
                .clone();

            state.play_action(action);
        }
    }
    #[test]
    fn test_card_matching() {
        println!("Testing card matching...");
        let mut rng = rand::prelude::SmallRng::from_os_rng();

        // Wir testen einfach zufällig, ob
        // das Card-Matching immer konsistente Ergebnisse liefert.
        for i in 0..3 {
            let mut state = FdoState::new_game(&mut rng);

            loop {
                let obs = state.observation_for_current_player();

                if obs.finished_stats.is_some() {
                    break;
                }

                for j in 0..10  {
                    let c_state = card_matching(
                        &state,
                        &obs,
                        &mut rng
                    );

                    //if rng.random_bool(0.001) {
                        println!("Iteration: {}", i);
                        println!("State: {:?}", &c_state.0);
                        println!("Reservations: {:?}", c_state.1);

                        println!("Real Hands: {:?}", &obs.phi_real_hands);
                        println!("Game Start Player: {:?}", obs.game_starting_player);
                        println!("Real reservations: {:?}", obs.phi_real_reservations);
                        println!("Visible reservations: {:?}", obs.visible_reservations);
                    //}

                    let _is_consistent = is_consistent(&state, &obs, c_state.0, c_state.1.to_oriented_arr());

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
    
}
