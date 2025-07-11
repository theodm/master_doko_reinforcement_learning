use crate::action::allowed_actions::DoAllowedActions;
use crate::basic::phase::DoPhase;
use crate::hand::hand::DoHand;
use crate::player::player::DoPlayer;
use crate::player::player_set::DoPlayerSet;
use crate::reservation::reservation::DoVisibleReservation;
use crate::reservation::reservation_round::DoReservationRound;
use crate::stats::stats::DoEndOfGameStats;
use crate::trick::trick::DoTrick;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DoObservation {
    pub phase: DoPhase,
    pub observing_player: DoPlayer,

    pub current_player: Option<DoPlayer>,
    pub allowed_actions_current_player: DoAllowedActions,

    pub game_starting_player: DoPlayer,

    // Gibt den Spieler an, der eine Hochzeit angekündigt hat. Nur gefüllt,
    // wenn ein Vorbehalt angesagt und die Hochzeit bereits getauft wurde.
    pub wedding_player_if_wedding_announced: Option<DoPlayer>,

    pub tricks: [Option<DoTrick>; 12],
    pub visible_reservations: [Option<DoVisibleReservation>; 4],

    pub player_eyes: [u32; 4],
    pub observing_player_hand: DoHand,

    // Das Ergebnis
    pub finished_observation: Option<DoEndOfGameStats>,

    // Werte des ganzen Spiels, diese
    // stehen dem Spieler grundsätzlich
    // nicht zur Verfügung. (phi = possible hidden information)
    pub phi_re_players: Option<DoPlayerSet>,

    pub phi_real_reservations: DoReservationRound,

    // Die echten Karten der Spieler
    // ausgehend von PLAYER_BOTTOM
    pub phi_real_hands: [DoHand; 4],

    pub phi_team_eyes: [u32; 2],

}