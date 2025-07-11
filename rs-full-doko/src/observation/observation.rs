use crate::action::allowed_actions::FdoAllowedActions;
use crate::announcement::announcement::{FdoAnnouncement, FdoAnnouncementOccurrence};
use crate::basic::phase::FdoPhase;
use crate::game_type::game_type::FdoGameType;
use crate::hand::hand::FdoHand;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation::FdoVisibleReservation;
use crate::reservation::reservation_round::FdoReservationRound;
use crate::stats::stats::FdoEndOfGameStats;
use crate::trick::trick::FdoTrick;
use crate::util::po_arr::PlayerOrientedArr;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FdoObservation {
    pub phase: FdoPhase,
    pub observing_player: FdoPlayer,

    pub current_player: Option<FdoPlayer>,
    pub allowed_actions_current_player: FdoAllowedActions,

    pub game_starting_player: FdoPlayer,

    // Gibt den Spieler an, der eine Hochzeit angekündigt hat. Nur gefüllt,
    // wenn ein Vorbehalt angesagt und die Hochzeit bereits getauft wurde.
    pub wedding_player_if_wedding_announced: Option<FdoPlayer>,

    pub tricks: heapless::Vec<FdoTrick, 12>,
    pub visible_reservations: PlayerOrientedArr<FdoVisibleReservation>,

    // Die Ansagen, die in diesem Spiel gemacht wurden.
    pub announcements: heapless::Vec<FdoAnnouncementOccurrence, 12>,

    pub player_eyes: PlayerZeroOrientedArr<u32>,
    pub observing_player_hand: FdoHand,

    // Das Ergebnis
    pub finished_stats: Option<FdoEndOfGameStats>,

    // Werte des ganzen Spiels, diese
    // stehen dem Spieler grundsätzlich
    // nicht zur Verfügung. (phi = possible hidden information)
    pub phi_re_players: Option<FdoPlayerSet>,

    pub phi_real_reservations: FdoReservationRound,

    // Die echten Karten der Spieler
    // ausgehend von PLAYER_BOTTOM
    pub phi_real_hands: PlayerZeroOrientedArr<FdoHand>,

    pub phi_team_eyes: [u32; 2],

    pub game_type: Option<FdoGameType>,

    pub re_lowest_announcement: Option<FdoAnnouncement>,
    pub contra_lowest_announcement: Option<FdoAnnouncement>,

}