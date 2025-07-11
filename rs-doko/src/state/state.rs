use rand::prelude::SmallRng;
use rand::Rng;
use strum::EnumCount;
use crate::action::action::{action_to_type, DoAction, DoActionType};
use crate::action::allowed_actions::{calculate_allowed_actions_in_normal_game, DoAllowedActions, random_action};
use crate::basic::color::DoColor;
use crate::basic::phase::DoPhase;
use crate::card::cards::DoCard;
use crate::consistent::consistent_state::are_hands_consistent;
use crate::hand::hand::{DoHand, hand_remove};
use crate::hand::hand_random::distribute_cards;
use crate::observation::observation::DoObservation;
use crate::player::player::{DoPlayer, PLAYER_BOTTOM, player_increase, player_wraparound};
use crate::player::player_set::DoPlayerSet;
use crate::reservation::reservation::DoReservation;
use crate::reservation::reservation_play::play_reservation;
use crate::reservation::reservation_round::DoReservationRound;
use crate::reservation::reservation_winning_logic::{DoReservationResult, winning_player_in_reservation_round};
use crate::reservation::visible_reservations_logic::get_visible_reservations;
use crate::stats::stats::{calculate_end_of_game_stats, DoEndOfGameStats};
use crate::teams::team_logic::{DoTeamState, is_final_team_state, resolve_team_state};
use crate::trick::trick::DoTrick;
use crate::trick::trick_eyes::trick_eyes;
use crate::trick::trick_play::play_card;
use crate::trick::trick_winning_player_logic::winning_player_in_trick_in_normal_game;
use crate::util::bitflag::bitflag::bitflag_contains;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DoState {
    /// Vorbehaltsrunde
    reservations_round: DoReservationRound,

    /// Stichrunden, 12 Stiche
    ///
    /// Some(DoTrick) wenn der Stich gespielt wurde
    /// None wenn der Stich noch nicht gespielt wurde
    pub tricks: [Option<DoTrick>; 12],

    /// Die aktuellen Karten der Spieler
    /// ausgehend von PLAYER_BOTTOM
    pub hands: [DoHand; 4],

    // Folgende Eigenschaften sind theoretisch ableitbar,
    // wir speichern sie aber trotzdem, um die Berechnung
    // zu vereinfachen.

    /// Der Spieler, der aktuell an der Reihe ist. Oder None,
    /// wenn das Spiel vorbei ist.
    pub current_player: Option<DoPlayer>,

    /// Die aktuelle Phase des Spiels.
    pub current_phase: DoPhase,

    /// Der Index des aktuellen Stiches, bei dem wir uns befinden,
    /// bei 0 beginnend.
    current_trick_index: usize,

    /// Ergebnis der Vorbehaltsrunde, gespeichert,
    /// damit wir es nicht immer neu berechnen. Gesetzt,
    /// wenn die Vorbehaltsrunde beendet ist.
    reservation_result: Option<DoReservationResult>,

    /// Augenzahlen der Spieler, die sie bereits gefangen haben,
    /// dabei werden nur fertige Stiche gezäglt.
    player_eyes: [u32; 4],

    /// Anzahl der Stiche, die ein Spieler gemacht hat.
    player_num_tricks: [u32; 4],

    /// Aktueller Stand des Spiels bezüglich der Teamverteilung,
    /// bei einer Hochzeit gibt es zusätzliche Angaben.
    team_state: DoTeamState,

    /// Die Ergebnisse der Runde, wenn das Spiel vorbei ist.
    end_of_game_stats: Option<DoEndOfGameStats>
}

impl DoState {

    /// Überprüft, ob der übergebene Zustand konsistent
    /// mit dem aktuellen Spielgeschehen ist. Also ob aus einer
    /// Informationsmenge ein tatsächlich vorliegender Zustand gesampled wurde.
    pub fn check_consistent_state(
        &self,
        hands: [DoHand; 4]
    ) -> bool {

        let tricks: heapless::Vec<DoTrick, 12> = Vec::from(&self.tricks)
            .iter()
            .filter(|trick| trick.is_some())
            .map(|trick| trick.unwrap())
            .collect::<heapless::Vec<DoTrick, 12>>();

        are_hands_consistent(
            self.team_state,
            tricks,
            self.hands,
            hands
        )
    }

    pub fn clone_with_different_hands(
        &self,
        hands: [DoHand; 4]
    ) -> DoState {
        let mut new_state = self.clone();

        new_state.hands = hands;

        return new_state
    }

    /// Nur für Testzwecke direkt aufrufen!
    fn new_game_from_hand_and_start_player(
        hands: [DoHand; 4],
        start_player: DoPlayer
    ) -> DoState {
        DoState {
            // Eine Vorbehaltsrunde besteht am Anfang immer und
            // beginnt mit dem Startspieler des Spiels.
            reservations_round: DoReservationRound::empty(start_player),

            // Am Anfang gibt es noch keinen Stich.
            tricks: [None; 12],

            // Jeder Spieler erhält 12 zufällige Karten
            hands: hands,

            // Der Startspieler beginnt das Spiel
            current_player: Some(start_player),

            // Das Spiel beginnt in der Vorbehaltsphase
            current_phase: DoPhase::Reservation,

            // Das Spiel beginnt mit dem Stich-Index 0
            current_trick_index: 0,

            // Am Anfang gibt es noch kein Ergebnis der Vorbehaltsrunde.
            reservation_result: None,

            // Am Anfang hat noch kein Spieler Augen gesammelt.
            player_eyes: [
                0, 0, 0, 0
            ],

            // Am Anfang hat noch kein Spieler Stiche gemacht.
            player_num_tricks: [
                0, 0, 0, 0
            ],

            // Am Anfang sind die Teams noch nicht bekannt.
            team_state: DoTeamState::InReservations,

            // Am Anfang gibt es noch kein Ergebnis.
            end_of_game_stats: None,
        }
    }

    pub fn new_game(
        rng: &mut SmallRng
    ) -> DoState {
        // Zufälliger Startspieler
        let start_player: DoPlayer = rng.gen_range(0..4);
        // Zufällige Karten
        let hands = distribute_cards(rng);

        DoState::new_game_from_hand_and_start_player(hands, start_player)
    }

    fn get_re_players(&self) -> Option<DoPlayerSet> {
        match self.team_state {
            DoTeamState::InReservations => { None }
            DoTeamState::WeddingUnsolved {..} => { None }
            DoTeamState::WeddingSolved { re_players, .. } => { Some(re_players) }
            DoTeamState::NoWedding { re_players } => { Some(re_players) }
        }
    }

    fn get_re_players_or_throw(&self) -> DoPlayerSet {
        // ToDo: Besser auslagern in die team_logic.rs?
        match self.team_state {
            DoTeamState::InReservations => { panic!("Die Teams stehen in der Vorbehaltsphase noch nicht fest.") }
            DoTeamState::WeddingUnsolved {..} => { panic!("Die Teams stehen noch nicht fest. Die Hochzeit ist ungeklärt.") }
            DoTeamState::WeddingSolved { re_players, .. } => { re_players }
            DoTeamState::NoWedding { re_players } => { re_players }
        }
    }

    pub fn play_action(&mut self, action: DoAction) {
        debug_assert!(self.current_phase != DoPhase::Finished, "Es kann keine Aktion gespielt werden, wenn das Spiel vorbei ist.");

        let current_player = self.current_player.unwrap();

        match action_to_type(action) {
            DoActionType::Card(card) => {
                debug_assert!(self.current_phase == DoPhase::PlayCard, "Es kann nur eine Karte gespielt werden, wenn die Phase PlayCard ist.");

                let reservation_result = self.reservation_result.unwrap();

                // Karte aus der Hand des Spielers nehmen.
                self.hands[current_player] = hand_remove(self.hands[current_player], card);

                // Karte im Stich hinzufügen
                let current_trick = self.tricks[self.current_trick_index].as_mut().unwrap();

                play_card(current_trick, card);

                // Wenn der Stich vollständig ist, dann beginnen wir den nächsten Stich.
                if current_trick.is_completed() {
                    self.current_trick_index += 1;

                    // Wir berechnen den Gewinner des Stiches.
                    let winning_player = winning_player_in_trick_in_normal_game(&current_trick);

                    // Wir erhöhen die Augenzahl des Spielers.
                    self.player_eyes[winning_player] += trick_eyes(&current_trick);

                    // Wir erhöhen die Anzahl der Stiche des Spielers.
                    self.player_num_tricks[winning_player] += 1;

                    // Wir überprüfen, ob durch den Stich sich ein neuer Team-Status
                    // ergibt, z.B. dadurch, dass eine Hochzeit geklärt wurde.
                    if !is_final_team_state(self.team_state) {
                        self.team_state = resolve_team_state(
                            reservation_result,
                            &self.tricks,
                            self.hands

                        );
                    }

                    if self.current_trick_index == 12 {
                        // Wenn das ganze Spiel vorbei ist, dann ist die Spielphase vorbei.
                        self.current_phase = DoPhase::Finished;
                        // Dann gibt es keinen aktuellen Spieler mehr.
                        self.current_player = None;

                        // Dann berechnen wir auch direkt die Punkte.
                        self.end_of_game_stats = Some(
                            calculate_end_of_game_stats(
                                self.get_re_players_or_throw(),
                                self.player_eyes,
                                self.player_num_tricks
                            )
                        );

                        return;
                    }

                    // Ansonsten: Wir beginnen den nächsten Stich mit dem
                    // Gewinner des vorherigen Stiches.
                    self.tricks[self.current_trick_index] = Some(DoTrick::empty(winning_player));

                    // Er ist außerdem der nächste Spieler
                    self.current_player = Some(winning_player);

                    return;
                }

                // Grundsätzlich ist der nächste Spieler dran.
                let new_current_player = player_increase(current_player);
                self.current_player = Some(new_current_player);

            }
            DoActionType::Reservation(reservation) => {
                debug_assert!(self.current_phase == DoPhase::Reservation, "Es kann nur eine Vorbehaltsaktion gespielt werden, wenn die Phase Reservation ist.");

                play_reservation(&mut self.reservations_round, reservation);

                let new_current_player = player_increase(current_player);
                self.current_player = Some(new_current_player);

                // ToDo: Eine eigene Methode, welche die Sanity-Checks immer wieder ausführt?
                // debug_assert! {
                //     current_player_reservations = self.reservations_round.current_player();
                //
                //     match current_player_reservations {
                //         Some(current_player) => current_player == self.current_player
                //         None => true
                //     }
                // }

                // Die Vorbehaltsrunde ist beendet.
                if self.reservations_round.is_completed() {
                    // Dann sind wir nun in der Spielphase.
                    self.current_phase = DoPhase::PlayCard;

                    let reservation_result = winning_player_in_reservation_round(&self.reservations_round);
                    // Wir berechnen den Gewinner der Vorbehaltsrunde
                    self.reservation_result = Some(reservation_result);

                    // Den ersten Stich beginnen
                    self.current_trick_index = 0;
                    self.tricks[0] = Some(DoTrick::empty(new_current_player));

                    // Wir überprüfen, ob sich ein neuer Team-Status ergibt. (Re-Spieler
                    // könnten jetzt feststehen.)
                    if !is_final_team_state(self.team_state) {
                        self.team_state = resolve_team_state(
                            reservation_result,
                            &self.tricks,
                            self.hands
                        );
                    }
                }
            }
        }

    }

    /// Wählt für den aktuellen Spieler eine zufällige Aktion aus. Für Optimierungszwecke,
    /// wenn wir im Rahmen des MCTS ein zufälliges Rollout durchgehen und die Observationen
    /// dazwischen gar nicht benötigen. Gibt true zurück, wenn keine Aktion ausgeführt werden
    /// konnte, da das Spiel vorbei ist.
    pub fn random_action_for_current_player(
        &mut self,
        rng: &mut SmallRng
    ) -> bool {
        // Spiel beendet, daher kann das Rollout abgebrochen werden.
        if self.current_phase == DoPhase::Finished {
            return true;
        }

        let allowed_actions = calculate_allowed_actions_in_normal_game(
            self.current_phase,
            self.tricks[self.current_trick_index].and_then(|trick| trick.color()),
            self.hands[self.current_player.unwrap()]
        );

        self.play_action(random_action(allowed_actions, rng));

        // ToDo: Rückgabe fragwürdig :O
        false
    }

    pub fn observation_for_current_player(&self) -> DoObservation {
        let observing_player = self
            .current_player
            .unwrap_or_else(|| { PLAYER_BOTTOM });

        let current_trick_color: Option<DoColor> = if self.current_trick_index < 12 {
            self
                .tricks[self.current_trick_index]
                .and_then(|trick| trick.color())
        } else {
            None
        };

        let wedding_player_if_wedding_announced = match self.team_state {
            DoTeamState::WeddingUnsolved { wedding_player, .. } => Some(wedding_player),
            DoTeamState::WeddingSolved { wedding_player, .. } => Some(wedding_player),
            _ => None
        };


        // // ToDo: In-Place berechnen
        let mut re_eyes = 0;
        let mut kontra_eyes = 0;

        // Wir rechnen die Punkte der Teams zusammen
        // und die Anzahl der Stiche, die sie gemacht haben.
        // let team_eyes = self.get_re_players()
        //     .map(|re_players| {
        //         let mut re_eyes = 0;
        //         let mut kontra_eyes = 0;
        //
        //         for player in 0..4 {
        //             if bitflag_contains(re_players, player) {
        //                 re_eyes += self.player_eyes[player];
        //             } else {
        //                 kontra_eyes += self.player_eyes[player];
        //             }
        //         }
        //
        //         [re_eyes, kontra_eyes]
        //     })
        //     .unwrap_or_else(|| { [0, 0] });

        DoObservation {
            phase: self.current_phase,
            observing_player: observing_player,

            current_player: self.current_player,
            allowed_actions_current_player: calculate_allowed_actions_in_normal_game(
                self.current_phase,
                current_trick_color,
                self.hands[observing_player]
            ),

            game_starting_player: self.reservations_round.start_player,

            wedding_player_if_wedding_announced: wedding_player_if_wedding_announced,
            tricks: self.tricks,

            visible_reservations: get_visible_reservations(&self.reservations_round, observing_player),

            player_eyes: self.player_eyes,
            observing_player_hand: self.hands[observing_player],

            finished_observation: self.end_of_game_stats.clone(),

            phi_re_players: self.get_re_players(),

            phi_real_reservations: self.reservations_round,

            phi_real_hands: self.hands,

            phi_team_eyes: [re_eyes, kontra_eyes]
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use crate::action::allowed_actions::allowed_actions_from_vec;
    use crate::basic::team::DoTeam;
    use crate::card::cards::DoCard;
    use crate::hand::hand::{hand_from_vec, hand_len};
    use crate::player::player::{PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use super::*;
    use crate::player::player_set::player_set_create;
    use crate::reservation::reservation::DoVisibleReservation;

    #[test]
    fn test_new_game() {
        let mut rng = SmallRng::seed_from_u64(42);

        let state = DoState::new_game(&mut rng);

        assert_eq!(state.current_phase, DoPhase::Reservation);
        assert_eq!(state.current_trick_index, 0);
        assert_eq!(state.current_player, Some(0));
        assert_eq!(state.reservations_round.start_player, 0);
        assert_eq!(state.tricks, [None; 12]);
        assert_eq!(hand_len(state.hands[0]), 12);
        assert_eq!(hand_len(state.hands[1]), 12);
        assert_eq!(hand_len(state.hands[2]), 12);
        assert_eq!(hand_len(state.hands[3]), 12);
        assert_eq!(state.reservation_result, None);
        assert_eq!(state.player_eyes, [0, 0, 0, 0]);
        assert_eq!(state.player_num_tricks, [0, 0, 0, 0]);
        assert_eq!(state.team_state, DoTeamState::InReservations);
        assert!(
            matches!(state.end_of_game_stats, None)
        );
    }

    #[test]
    fn full_normal_game() {

    }

    #[test]
    fn full2() {



    }

}