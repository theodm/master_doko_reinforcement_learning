use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::team::team_logic::FdoTeamState;

/// Diese Methode berechnet die Anzahl der Karten, die ein Spieler mindestens gehabt haben muss,
/// um die letzte Ansage die getätigt wurde, zu machen. Dies ist erforderlich für die Berechnung
/// bis zu welcher Anzahl von Karten in der Hand eine Erwiderung möglich ist.
///
/// z.B.:
/// calc_number_of_cards_announcement_possible_for_last_announcement({RE}, None) -> 11
///
/// Hier war die gemachte Re-Ansage mit 11 Karten in der Hand möglich. (Daraus lässt sich später
/// ableiten, dass eine Erwiderung mit 10 Karten möglich ist.)
///
/// calc_number_of_cards_announcement_possible_for_last_announcement({RE_NO_90}, None) -> 10
///
/// Hier war die gemachte Re-Unter-90-Ansage mit 10 Karten in der Hand möglich. (Daraus lässt sich später
/// ableiten, dass eine Erwiderung mit 9 Karten möglich ist.)
pub fn calc_number_of_cards_announcement_possible_for_last_announcement(
    previous_announcements_of_team: FdoAnnouncementSet,

    wedding_index_solved_trick: Option<usize>
) -> Option<usize> {
    let wedding_index_solved_trick = wedding_index_solved_trick
        .unwrap_or_else(|| 0);

    let number_of_cards_announcement_possible = 11 - wedding_index_solved_trick;
    let number_of_cards_u90_announcement_possible = 10 - wedding_index_solved_trick;
    let number_of_cards_u60_announcement_possible = 9 - wedding_index_solved_trick;
    let number_of_cards_u30_announcement_possible = 8 - wedding_index_solved_trick;
    let number_of_cards_black_announcement_possible = 7 - wedding_index_solved_trick;

    return if previous_announcements_of_team.contains(FdoAnnouncement::Black) {
        Some(number_of_cards_black_announcement_possible)
    } else if previous_announcements_of_team.contains(FdoAnnouncement::No30) {
        Some(number_of_cards_u30_announcement_possible)
    } else if previous_announcements_of_team.contains(FdoAnnouncement::No60) {
        Some(number_of_cards_u60_announcement_possible)
    } else if previous_announcements_of_team.contains(FdoAnnouncement::No90) {
        Some(number_of_cards_u90_announcement_possible)
    } else if previous_announcements_of_team.contains(FdoAnnouncement::ReContra) {
        Some(number_of_cards_announcement_possible)
    } else {
        None
    }
}


pub fn internal_calc_allowed_annoucements(
    // Anzahl der Karten in der Hand des aktuellen Spielers
    current_player_number_of_cards_on_hand: usize,

    // Bereits gemachte Ansagen des Teams des aktuellen Spielers
    previous_announcements_of_team: FdoAnnouncementSet,

    // Wenn die Hochzeit geklärt ist, dann verschiebt sich der Zeitpunkt
    // in dem Ansagen gemacht werden können.
    wedding_index_solved_trick: Option<usize>,

    // Anzahl der Karten, die der Spieler des Gegner-Teams mindestens haben musste, um die letzte Ansage
    // zu machen. Diese Information wird benötigt, um zu berechnen, ob eine Erwiderung
    // auf eine Ansage noch möglich ist. Falls keine Ansage gemacht wurde, dann ist dieser Wert None.
    number_of_cards_possible_last_enemy_announcement: Option<usize>,
) -> FdoAnnouncementSet {
    let wedding_index_solved_trick = wedding_index_solved_trick
        .unwrap_or_else(|| 0);

    let number_of_cards_announcement_possible = 11 - wedding_index_solved_trick;
    let number_of_cards_u90_announcement_possible = 10 - wedding_index_solved_trick;
    let number_of_cards_u60_announcement_possible = 9 - wedding_index_solved_trick;
    let number_of_cards_u30_announcement_possible = 8 - wedding_index_solved_trick;
    let number_of_cards_black_announcement_possible = 7 - wedding_index_solved_trick;

    let mut remaining_announcements = FdoAnnouncementSet::new();

    let announcement_possible = current_player_number_of_cards_on_hand >= number_of_cards_announcement_possible;
    let announcement_under_90_possible = announcement_possible || (current_player_number_of_cards_on_hand >= number_of_cards_u90_announcement_possible && previous_announcements_of_team.contains(FdoAnnouncement::ReContra));
    let announcement_under_60_possible = announcement_under_90_possible || (current_player_number_of_cards_on_hand >= number_of_cards_u60_announcement_possible && previous_announcements_of_team.contains(FdoAnnouncement::No90));
    let announcement_under_30_possible = announcement_under_60_possible || (current_player_number_of_cards_on_hand >= number_of_cards_u30_announcement_possible && previous_announcements_of_team.contains(FdoAnnouncement::No60));
    let announcement_black_possible = announcement_under_30_possible || (current_player_number_of_cards_on_hand >= number_of_cards_black_announcement_possible && previous_announcements_of_team.contains(FdoAnnouncement::No30));

    // Erst die Ansagen hinzufügen, die nach der Anzahl der Karten in der Hand
    // des Spielers möglich sind. Update: Aber nur 1 unter der aktuellen
    if announcement_possible {
        remaining_announcements.add(FdoAnnouncement::ReContra);
    }
    if announcement_under_90_possible {
        remaining_announcements.add(FdoAnnouncement::No90);
    }
    if announcement_under_60_possible {
        remaining_announcements.add(FdoAnnouncement::No60);
    }
    if announcement_under_30_possible {
        remaining_announcements.add(FdoAnnouncement::No30);
    }
    if announcement_black_possible {
        remaining_announcements.add(FdoAnnouncement::Black);
    }

    // Dann bereits gemachte Ansagen entfernen, die bereits gemacht wurden.
    // ToDo: Bit-Fiddling möglich?
    if previous_announcements_of_team.contains(FdoAnnouncement::ReContra) {
        remaining_announcements.remove(FdoAnnouncement::ReContra);
    }
    if previous_announcements_of_team.contains(FdoAnnouncement::No90) {
        remaining_announcements.remove(FdoAnnouncement::No90);
    }
    if previous_announcements_of_team.contains(FdoAnnouncement::No60) {
        remaining_announcements.remove(FdoAnnouncement::No60);
    }
    if previous_announcements_of_team.contains(FdoAnnouncement::No30) {
        remaining_announcements.remove(FdoAnnouncement::No30);
    }
    if previous_announcements_of_team.contains(FdoAnnouncement::Black) {
        remaining_announcements.remove(FdoAnnouncement::Black);
    }

    // Jetzt noch Announcements auf eins unter dem aktuellen Level beschränken.
    if remaining_announcements.contains(FdoAnnouncement::ReContra) {
        remaining_announcements.remove(FdoAnnouncement::No90);
        remaining_announcements.remove(FdoAnnouncement::No60);
        remaining_announcements.remove(FdoAnnouncement::No30);
        remaining_announcements.remove(FdoAnnouncement::Black);
    }

    if remaining_announcements.contains(FdoAnnouncement::No90) {
        remaining_announcements.remove(FdoAnnouncement::No60);
        remaining_announcements.remove(FdoAnnouncement::No30);
        remaining_announcements.remove(FdoAnnouncement::Black);
    }

    if remaining_announcements.contains(FdoAnnouncement::No60) {
        remaining_announcements.remove(FdoAnnouncement::No30);
        remaining_announcements.remove(FdoAnnouncement::Black);
    }

    if remaining_announcements.contains(FdoAnnouncement::No30) {
        remaining_announcements.remove(FdoAnnouncement::Black);
    }

    // Eine Erwiderung ist nur möglich, wenn keine reguläre Ansage oder Absage möglich ist.
    let regular_announcement_allowed = remaining_announcements.len() > 0;

    if number_of_cards_possible_last_enemy_announcement.is_some() && !regular_announcement_allowed {
        // Der Spieler darf ein Erwiderungs-Re nur bis zu dem Zeitpunkt absagen, bis er
        // eine Karte weniger hat als die Anzahl von Karten bis zu der die vorangegange Absage des Kontra-Teams
        // möglich war.
        //
        // Bsp.:
        // Das Kontra-Team hat mit 12 Karten Kontra angesagt. Das Re-Team kann noch mit 10 Karten ein Erwiderungs-Re
        // ansagen, da das Kontra auch noch mit 11 Karten angesagt hätte können.
        let counter_re_per_number_of_cards_allowed = current_player_number_of_cards_on_hand >= number_of_cards_possible_last_enemy_announcement.unwrap() - 1;

        // Der Spieler darf kein Erwiderungs-Re mehr ansagen, wenn er bereits ein reguläres Re oder
        // ein Erwiderungs-Re angesagt hat.
        let counter_re_already_announced = previous_announcements_of_team.contains(FdoAnnouncement::CounterReContra);
        // Der Spieler darf auch kein Erwiderungs-Re mehr ansagen, wenn er bereits ein normales Re
        // angesagt hat.
        let regular_re_already_announced = previous_announcements_of_team.contains(FdoAnnouncement::ReContra);
        // Irgendein Re angesagt.
        let re_already_announced = counter_re_already_announced || regular_re_already_announced;

        if counter_re_per_number_of_cards_allowed &&
            !counter_re_already_announced &&
            !re_already_announced {
            remaining_announcements.add(FdoAnnouncement::CounterReContra);
        }
    }

    return remaining_announcements;
}


pub fn calc_allowed_announcements(
    current_player: FdoPlayer,
    current_player_number_of_cards_on_hand: usize,

    team_state: FdoTeamState,

    re_lowest_announcement: Option<FdoAnnouncement>,
    contra_lowest_announcement: Option<FdoAnnouncement>,
) -> FdoAnnouncementSet {
    // Wenn die Hochzeit geklärt ist, dann verschiebt sich der Zeitpunkt
    // in dem Ansagen gemacht werden können.
    let mut wedding_index_solved_trick: Option<usize> = None;

    let mut _re_players: FdoPlayerSet = FdoPlayerSet::empty();

    // Wenn die Hochzeit noch nicht geklärt ist, dann können
    // noch keine Ansagen gemacht werden.
    match team_state {
        FdoTeamState::InReservations => {
            panic!("should not happen")
        },
        FdoTeamState::WeddingUnsolved { .. } => {
            return FdoAnnouncementSet::new()
        },
        FdoTeamState::WeddingSolved { wedding_player: r#wedding_player, re_players, solved_trick_index } => {
            wedding_index_solved_trick = Some(solved_trick_index);
            _re_players = re_players;
        },
        FdoTeamState::NoWedding { re_players } => {
            _re_players = re_players;
        }
    };

    let re_players = _re_players;

    let current_player_is_re = re_players.contains(current_player);

    let (current_player_team_previous_announcements, current_player_enemy_team_previous_announcements) = match current_player_is_re {
        true => (re_lowest_announcement, contra_lowest_announcement),
        false => (contra_lowest_announcement, re_lowest_announcement),
    };

    let number_of_cards_possible_enemy_team = calc_number_of_cards_announcement_possible_for_last_announcement(
        FdoAnnouncementSet::all_higher_than(current_player_enemy_team_previous_announcements),
        wedding_index_solved_trick,
    );

    return internal_calc_allowed_annoucements(
        current_player_number_of_cards_on_hand,
        FdoAnnouncementSet::all_higher_than(current_player_team_previous_announcements),
        wedding_index_solved_trick,
        number_of_cards_possible_enemy_team,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_calc_allowed_announcements() {
        assert_eq!(
            calc_allowed_announcements(
                FdoPlayer::TOP,
                10,
                FdoTeamState::NoWedding {
                    re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::BOTTOM])
                },
                Some(FdoAnnouncement::No60),
                Some(FdoAnnouncement::No60)
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::No30
            ])
        );

        // Sonderfall: Eine Erwiderung ist möglich.
        assert_eq!(
            calc_allowed_announcements(
                FdoPlayer::TOP,
                10,
                FdoTeamState::NoWedding {
                    re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::BOTTOM])
                },
                None,
                Some(FdoAnnouncement::ReContra)
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::CounterReContra
            ])
        );

        // Sonderfall: Eine Erwiderung ist möglich. (Kontra-Fall)
        assert_eq!(
            calc_allowed_announcements(
                FdoPlayer::LEFT,
                6,
                FdoTeamState::NoWedding {
                    re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::BOTTOM])
                },
                Some(FdoAnnouncement::Black),
                None
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::CounterReContra
            ])
        );

        // Sonderfall: Wenn eine Erwiderung schon da ist, kann diese danach nicht mehr gemacht werden.
        assert_eq!(
            calc_allowed_announcements(
                FdoPlayer::LEFT,
                10,
                FdoTeamState::NoWedding {
                    re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::BOTTOM])
                },
                Some(FdoAnnouncement::Black),
                Some(FdoAnnouncement::CounterReContra)
            ),
            FdoAnnouncementSet::new()
        );


    }

    #[test]
    pub fn test_internal_calc_allowed_annoucements() {
        // Wenn eine Hochzeit vorliegt und im 2. Stich geklärt ist, wer mit wem spielt,
        // dann können im 3. Stich alle Ansagen gemacht werden, auch wenn er nur noch 10
        // Karten hat.
        assert_eq!(internal_calc_allowed_annoucements(
            // 1. Stich: Keine Klärung
            // 2. Stich: Klärung
            // Danach: 10 Karten
            10,
            FdoAnnouncementSet::all_higher_than(None),
            Some(1),
            None
        ), FdoAnnouncementSet::from_vec(vec![
            FdoAnnouncement::ReContra,
        ]));

        // Wenn eine Hochzeit vorliegt und im 2. Stich geklärt ist, wer mit wem spielt,
        // dann können im 4. Stich aber keine Ansagen mehr gemacht werden, wenn die Re-Ansage nicht
        // gemacht wurde.
        assert_eq!(internal_calc_allowed_annoucements(
            // 1. Stich: Keine Klärung
            // 2. Stich: Klärung
            // 3. Stich: Keine Ansage (aber letzter Zeitpunkt)
            // 9 Karten
            9,
            FdoAnnouncementSet::all_higher_than(None),
            Some(1),
            None
        ), FdoAnnouncementSet::new());

        // Ohne Hochzeit, wenn die Re-Ansage gemacht wurde, dann können mit 10 Karten alle darunterliegenden Ansagen
        // oder Absagen noch gemacht werden.
        assert_eq!(internal_calc_allowed_annoucements(
            // 1. Stich: Keine Ansage
            // 2. Stich: RE
            // 10 Karten
            10,
            FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::ReContra)),
            None,
            None
        ), FdoAnnouncementSet::from_vec(vec![
            FdoAnnouncement::No90,
        ]));

        // Ohne Hochzeit, wenn die Re-Ansage gemacht wurde, dann können mit 9 Karten
        // keine darunterliegenden Ansagen oder Absagen mehr gemacht werden.
        assert_eq!(
            internal_calc_allowed_annoucements(
                // 1. Stich: Keine Ansage
                // 2. Stich: RE
                // 3. Stich: Keine Ansage (aber letzter Zeitpunkt)
                // 9 Karten
                9,
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::ReContra)),
                None,
                None
            ),
            FdoAnnouncementSet::new()
        );

        // Ohne Hochzeit, wenn die Kontra-Ansage gemacht wurde, dann können mit 10 Karten alle darunterliegenden Ansagen
        // oder Absagen noch gemacht werden.
        assert_eq!(
            internal_calc_allowed_annoucements(
                // 1. Stich: Keine Ansage
                // 2. Stich: Kontra
                // 10 Karten
                10,
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::ReContra)),
                None,
                None
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::No90,
            ])
        );

        // Sonderfall: Eine Erwiderung ist nicht möglich, da eine reguläre Ansage noch möglich ist.
        assert_eq!(
            // 1. Stich: Keine Ansage
            // 2. Stich: Kontra
            // 11 Karten (Jetzt ist noch eine Re-Ansage ohne Erwiderung möglich)
            internal_calc_allowed_annoucements(
                11,
                FdoAnnouncementSet::all_higher_than(None),
                None,
                Some(11)
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::ReContra,
            ])
        );

        // Sonderfall: Eine Erwiderung ist noch einen Stich länger möglich.
        assert_eq!(
            // 1. Stich: Keine Ansage
            // 2. Stich: Kontra
            // 10 Karten (Jetzt ist nur noch eine Erwiderung möglich.)
            internal_calc_allowed_annoucements(
                10,
                FdoAnnouncementSet::all_higher_than(None),
                None,
                Some(11)
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::CounterReContra
            ])
        );

        // Sonderfall: Eine Erwiderung ist noch einen Stich länger möglich, wir testen
        // aber Kontra-Team und eine Absage des Re-Teams.
        assert_eq!(
            // 1. Stich: Keine Ansage
            // 2. Stich: Kontra
            // 10 Karten (Jetzt ist nur noch eine Erwiderung möglich.)
            internal_calc_allowed_annoucements(
                6,
                FdoAnnouncementSet::all_higher_than(None),
                None,
                Some(7)
            ),
            FdoAnnouncementSet::from_vec(vec![
                FdoAnnouncement::CounterReContra
            ])
        );

        // Sonderfall: Eine Erwiderung ist nicht möglich, wenn das Team
        // bereits ein reguläres Re gemacht hat.
        assert_eq!(
            // ... 10 Karten,
            internal_calc_allowed_annoucements(
                10,
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                None,
                Some(11)
            ),
            FdoAnnouncementSet::new()
        );
    }

    #[test]
    fn test_calc_number_of_cards_announcement_possible_for_last_announcement() {
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::ReContra)),
                None
            ),
            Some(11)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No90)),
                None
            ),
            Some(10)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No60)),
                None
            ),
            Some(9)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No30)),
                None
            ),
            Some(8)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                None
            ),
            Some(7)
        );

        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::ReContra)),
                Some(2)
            ),
            Some(9)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No90)),
                Some(2)
            ),
            Some(8)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No60)),
                Some(2)
            ),
            Some(7)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::No30)),
                Some(2)
            ),
            Some(6)
        );
        assert_eq!(
            calc_number_of_cards_announcement_possible_for_last_announcement(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                Some(2)
            ),
            Some(5)
        );
    }
}
