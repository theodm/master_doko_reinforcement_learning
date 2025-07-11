use serde::{Deserialize, Serialize};
use strum_macros::EnumCount;
use crate::announcement::announcement_set::FdoAnnouncementSet;
use crate::announcement::calc_announcement::{calc_allowed_announcements};
use crate::basic::team::FdoTeam;
use crate::player::player::FdoPlayer;
use crate::team::team_logic::FdoTeamState;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCount, Serialize, Deserialize)]
pub enum FdoAnnouncement {
    ReContra = 1 << 0,
    No90 = 1 << 1,
    No60 = 1 << 2,
    No30 = 1 << 3,
    Black = 1 << 4,

    CounterReContra = 1 << 5,

    NoAnnouncement = 1 << 6
}

impl ToString for FdoAnnouncement {
    fn to_string(&self) -> String {
        match self {
            FdoAnnouncement::ReContra => "Re/Contra".to_string(),
            FdoAnnouncement::No90 => "Keine_90".to_string(),
            FdoAnnouncement::No60 => "Keine_60".to_string(),
            FdoAnnouncement::No30 => "Keine_30".to_string(),
            FdoAnnouncement::Black => "Schwarz".to_string(),
            FdoAnnouncement::CounterReContra => "Counter_Re/Contra".to_string(),
            FdoAnnouncement::NoAnnouncement => "Keine_Ansage".to_string(),
        }
    }
}

/// Welcher Spieler hat in welchem Stich welche Ansage gemacht?
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct FdoAnnouncementOccurrence {
    pub card_index: usize,
    pub player: FdoPlayer,
    pub announcement: FdoAnnouncement,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct FdoAnnouncements {
    pub announcements: heapless::Vec<FdoAnnouncementOccurrence, 12>,

    pub re_lowest_announcement: Option<FdoAnnouncement>,
    pub contra_lowest_announcement: Option<FdoAnnouncement>,

    // Eigenschaften der aktuellen Runde (falls gestartet)
    pub number_of_turns_without_announcement: usize,

    pub starting_player: FdoPlayer,

    pub current_player_allowed_announcements: FdoAnnouncementSet,
}


#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum FdoAnnouncementProgressResult {
    NextPlayerIs(FdoPlayer),
    RoundIsOver(FdoPlayer),
}


impl FdoAnnouncements {
    pub fn new() -> FdoAnnouncements {
        FdoAnnouncements {
            announcements: heapless::Vec::new(),
            re_lowest_announcement: None,
            contra_lowest_announcement: None,

            starting_player: FdoPlayer::BOTTOM,

            number_of_turns_without_announcement: 0,
            current_player_allowed_announcements: FdoAnnouncementSet::new(),
        }
    }

    pub(crate) fn start_round(
        &mut self,

        card_index: usize,
        starting_player: FdoPlayer,

        current_player_number_of_cards_on_hand: PlayerZeroOrientedArr<usize>,
        team_state: FdoTeamState,
    ) -> FdoAnnouncementProgressResult {
        self.number_of_turns_without_announcement = 0;
        self.starting_player = starting_player;

        self.internal_progress(
            starting_player,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        )
    }

    pub(crate) fn play_announcement(
        &mut self,

        current_player: FdoPlayer,
        announcement: FdoAnnouncement,

        // Sollten sich innerhalb einer Runde nicht verändern.
        card_index: usize,
        player_number_of_cards_on_hand: PlayerZeroOrientedArr<usize>,
        team_state: FdoTeamState,
    ) -> FdoAnnouncementProgressResult {
        self.internal_play_announcement(
            current_player,
            announcement,

            card_index,
            team_state,
        );
        self.internal_progress(
            current_player + 1,
            card_index,
            player_number_of_cards_on_hand,
            team_state,
        )
    }

    fn internal_progress(
        &mut self,

        current_player: FdoPlayer,

        card_index: usize,
        player_number_of_cards_on_hand: PlayerZeroOrientedArr<usize>,
        team_state: FdoTeamState,
    ) -> FdoAnnouncementProgressResult {
        let mut current_player = current_player;

        loop {
            if self.number_of_turns_without_announcement == 4 {
                self.current_player_allowed_announcements = FdoAnnouncementSet::new();

                // Alle Spieler hatten die Möglichkeit anzusagen, aber keiner hat es getan.
                return FdoAnnouncementProgressResult::RoundIsOver(self.starting_player);
            }

            let allowed_announcements = calc_allowed_announcements(
                current_player,
                player_number_of_cards_on_hand[current_player],
                team_state,

                self.re_lowest_announcement,
                self.contra_lowest_announcement
            );

            if allowed_announcements.len() == 0 {
                // Spieler hat gar keine Möglichkeit etwas anzusagen. Dann überspringen wir
                // den Spieler einfach.
                self.internal_play_announcement(
                    current_player,
                    FdoAnnouncement::NoAnnouncement,
                    card_index,
                    team_state
                );
            } else {
                // Ein Spieler hat die Möglichkeit etwas zu machen :)
                self.current_player_allowed_announcements = allowed_announcements;

                return FdoAnnouncementProgressResult::NextPlayerIs(current_player);
            }

            current_player = current_player + 1;
        }
    }

    fn internal_play_announcement(
        &mut self,

        current_player: FdoPlayer,
        announcement: FdoAnnouncement,

        card_index: usize,
        team_state: FdoTeamState
    ) {
        match announcement {
            FdoAnnouncement::NoAnnouncement => {
                self.number_of_turns_without_announcement += 1;
                return;
            }
            announcement => {
                self.number_of_turns_without_announcement = 0;

                let announcement_occurrence = FdoAnnouncementOccurrence {
                    card_index,
                    player: current_player,
                    announcement,
                };

                self.announcements.push(announcement_occurrence);

                match current_player.team(team_state.get_re_players().unwrap()) {
                    FdoTeam::Re => {
                        self.re_lowest_announcement = Some(announcement);
                    }
                    FdoTeam::Kontra => {
                        self.contra_lowest_announcement = Some(announcement);
                    }
                }

            }
        }

    }
}

#[cfg(test)]
mod tests {
    use crate::announcement::announcement::{FdoAnnouncement, FdoAnnouncementProgressResult, FdoAnnouncements};
    use crate::announcement::announcement_set::FdoAnnouncementSet;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::team::team_logic::FdoTeamState;
    use crate::util::po_zero_arr::PlayerZeroOrientedArr;

    #[test]
    fn test_leere_round() {
        let mut announcements = FdoAnnouncements::new();

        let current_player_number_of_cards_on_hand = PlayerZeroOrientedArr::from_full([12, 12, 12, 12]);
        let team_state = FdoTeamState::NoWedding {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
        };
        let card_index = 0;

        let result = announcements.start_round(
            card_index,
            FdoPlayer::BOTTOM,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::BOTTOM));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 0);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::BOTTOM,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::LEFT));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 1);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::LEFT,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::TOP));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 2);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::TOP,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::RoundIsOver(FdoPlayer::BOTTOM));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 4);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::new());
    }

    #[test]
    fn test_round_with_autoskip() {
        let mut announcements = FdoAnnouncements::new();

        // Nurnoch RIGHT darf ansagen.
        let current_player_number_of_cards_on_hand = PlayerZeroOrientedArr::from_full([8, 8, 8, 11]);
        let team_state = FdoTeamState::NoWedding {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
        };
        let card_index = 0;

        let result = announcements.start_round(
            card_index,
            FdoPlayer::BOTTOM,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::ReContra,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 1);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::ReContra));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::No90,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::RoundIsOver(FdoPlayer::BOTTOM));
        assert_eq!(announcements.announcements.len(), 1);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 4);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::ReContra));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::new());
    }

    #[test]
    fn test_with_autoskip_counter() {
        let mut announcements = FdoAnnouncements::new();

        // Nurnoch RIGHT darf ansagen.
        let current_player_number_of_cards_on_hand = PlayerZeroOrientedArr::from_full([4, 4, 10, 11]);
        let team_state = FdoTeamState::NoWedding {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
        };
        let card_index = 0;

        let result = announcements.start_round(
            card_index,
            FdoPlayer::BOTTOM,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 0);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, None);
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::ReContra,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::ReContra,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::TOP));
        assert_eq!(announcements.announcements.len(), 1);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 2);
        assert_eq!(announcements.re_lowest_announcement, None);
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::ReContra));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::CounterReContra
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::TOP,
            FdoAnnouncement::CounterReContra,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 2);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.announcements[1].player, FdoPlayer::TOP);
        assert_eq!(announcements.announcements[1].card_index, 0);
        assert_eq!(announcements.announcements[1].announcement, FdoAnnouncement::CounterReContra);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 0);
        assert_eq!(announcements.re_lowest_announcement, Some(FdoAnnouncement::CounterReContra));
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::ReContra));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::No90,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::No90,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 3);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.announcements[1].player, FdoPlayer::TOP);
        assert_eq!(announcements.announcements[1].card_index, 0);
        assert_eq!(announcements.announcements[1].announcement, FdoAnnouncement::CounterReContra);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, Some(FdoAnnouncement::CounterReContra));
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::No90));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::No60,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::No60,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::NextPlayerIs(FdoPlayer::RIGHT));
        assert_eq!(announcements.announcements.len(), 4);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.announcements[1].player, FdoPlayer::TOP);
        assert_eq!(announcements.announcements[1].card_index, 0);
        assert_eq!(announcements.announcements[1].announcement, FdoAnnouncement::CounterReContra);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 3);
        assert_eq!(announcements.re_lowest_announcement, Some(FdoAnnouncement::CounterReContra));
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::No60));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::from_vec(
            vec![
                FdoAnnouncement::No30,
            ]
        ));

        let result = announcements.play_announcement(
            FdoPlayer::RIGHT,
            FdoAnnouncement::NoAnnouncement,

            card_index,
            current_player_number_of_cards_on_hand,
            team_state
        );

        assert_eq!(result, FdoAnnouncementProgressResult::RoundIsOver(FdoPlayer::BOTTOM));
        assert_eq!(announcements.announcements.len(), 4);
        assert_eq!(announcements.announcements[0].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[0].card_index, 0);
        assert_eq!(announcements.announcements[0].announcement, FdoAnnouncement::ReContra);
        assert_eq!(announcements.announcements[1].player, FdoPlayer::TOP);
        assert_eq!(announcements.announcements[1].card_index, 0);
        assert_eq!(announcements.announcements[1].announcement, FdoAnnouncement::CounterReContra);
        assert_eq!(announcements.announcements[2].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[2].card_index, 0);
        assert_eq!(announcements.announcements[2].announcement, FdoAnnouncement::No90);
        assert_eq!(announcements.announcements[3].player, FdoPlayer::RIGHT);
        assert_eq!(announcements.announcements[3].card_index, 0);
        assert_eq!(announcements.announcements[3].announcement, FdoAnnouncement::No60);
        assert_eq!(announcements.starting_player, FdoPlayer::BOTTOM);
        assert_eq!(announcements.number_of_turns_without_announcement, 4);
        assert_eq!(announcements.re_lowest_announcement, Some(FdoAnnouncement::CounterReContra));
        assert_eq!(announcements.contra_lowest_announcement, Some(FdoAnnouncement::No60));
        assert_eq!(announcements.current_player_allowed_announcements, FdoAnnouncementSet::new());
    }

}