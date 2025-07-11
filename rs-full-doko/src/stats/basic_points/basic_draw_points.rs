use serde::{Deserialize, Serialize};
use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoBasicDrawPointsDetails {
    pub winning_under_90_re: i32,
    pub winning_under_60_re: i32,
    pub winning_under_30_re: i32,

    pub winning_under_90_kontra: i32,
    pub winning_under_60_kontra: i32,
    pub winning_under_30_kontra: i32,

    pub re_reached_120_against_no_90: i32,
    pub re_reached_90_against_no_60: i32,
    pub re_reached_60_against_no_30: i32,
    pub re_reached_30_against_black: i32,

    pub kontra_reached_120_against_no_90: i32,
    pub kontra_reached_90_against_no_60: i32,
    pub kontra_reached_60_against_no_30: i32,
    pub kontra_reached_30_against_black: i32,
}

impl FdoBasicDrawPointsDetails {
    pub fn calculate(
        re_previous_announcements: FdoAnnouncementSet,
        kontra_previous_announcements: FdoAnnouncementSet,

        re_eyes: u32,
        kontra_eyes: u32,
    ) -> (i32, i32, FdoBasicDrawPointsDetails) {
        // Nach 7.1.4 DKV-TR werden nur die unter 7.2.2 (a) und 7.2.2 (e und f) gennanten
        // Punkte (sowie Sonderpunkte vergeben). Die Berechnung der Punkta nach 7.2.2 (a), (e) und (f)
        // erfolgt in dieser Methode.
        let re_under_90_announcement = re_previous_announcements.contains(FdoAnnouncement::No90);
        let re_under_60_announcement = re_previous_announcements.contains(FdoAnnouncement::No60);
        let re_under_30_announcement = re_previous_announcements.contains(FdoAnnouncement::No30);
        let re_black_announcement = re_previous_announcements.contains(FdoAnnouncement::Black);

        let kontra_under_90_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No90);
        let kontra_under_60_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No60);
        let kontra_under_30_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No30);
        let kontra_black_announcement = kontra_previous_announcements.contains(FdoAnnouncement::Black);

        let mut re_game_points = 0;
        let mut kontra_game_points = 0;

        let mut details = FdoBasicDrawPointsDetails {
            winning_under_90_re: 0,
            winning_under_60_re: 0,
            winning_under_30_re: 0,

            winning_under_90_kontra: 0,
            winning_under_60_kontra: 0,
            winning_under_30_kontra: 0,

            re_reached_120_against_no_90: 0,
            re_reached_90_against_no_60: 0,
            re_reached_60_against_no_30: 0,
            re_reached_30_against_black: 0,

            kontra_reached_120_against_no_90: 0,
            kontra_reached_90_against_no_60: 0,
            kontra_reached_60_against_no_30: 0,
            kontra_reached_30_against_black: 0,
        };

        // 7.2.2 DKV-TR (a)
        // Nr. 1: Gewonnen kommt nicht vor.
        // Nr. 2: Unter 90 gespielt.
        if re_eyes < 90 {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.winning_under_90_kontra = 1;
        }

        if kontra_eyes < 90 {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.winning_under_90_re = 1;
        }

        // Nr. 3: Unter 60 gespielt.
        if re_eyes < 60 {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.winning_under_60_kontra = 1;
        }

        if kontra_eyes < 60 {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.winning_under_60_re = 1;
        }

        // Nr. 4: Unter 30 gespielt.
        if re_eyes < 30 {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.winning_under_30_kontra = 1;
        }

        if kontra_eyes < 30 {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.winning_under_30_re = 1;
        }

        // Nr. 5: Schwarz gespielt.
        // Kann nicht vorkommen.

        // 7.2.2 DKV-TR (e)
        // Re hat 120 Augen gegen "keine 90" gemacht.
        if re_eyes >= 120 && kontra_under_90_announcement {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.re_reached_120_against_no_90 = 1;
        }

        // Re hat 90 Augen gegen "keine 60" gemacht.
        if re_eyes >= 90 && kontra_under_60_announcement {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.re_reached_90_against_no_60 = 1;
        }

        // Re hat 60 Augen gegen "keine 30" gemacht.
        if re_eyes >= 60 && kontra_under_30_announcement {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.re_reached_60_against_no_30 = 1;
        }

        // Re hat 30 Augen gegen "Schwarz" gemacht.
        if re_eyes >= 30 && kontra_black_announcement {
            re_game_points = re_game_points + 1;
            kontra_game_points = kontra_game_points - 1;
            details.re_reached_30_against_black = 1;
        }

        // 7.2.2 DKV-TR (f)
        // Kontra hat 120 Augen gegen "keine 90" gemacht.
        if kontra_eyes >= 120 && re_under_90_announcement {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.kontra_reached_120_against_no_90 = 1;
        }

        // Kontra hat 90 Augen gegen "keine 60" gemacht.
        if kontra_eyes >= 90 && re_under_60_announcement {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.kontra_reached_90_against_no_60 = 1;
        }

        // Kontra hat 60 Augen gegen "keine 30" gemacht.
        if kontra_eyes >= 60 && re_under_30_announcement {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.kontra_reached_60_against_no_30 = 1;
        }

        // Kontra hat 30 Augen gegen "Schwarz" gemacht.
        if kontra_eyes >= 30 && re_black_announcement {
            kontra_game_points = kontra_game_points + 1;
            re_game_points = re_game_points - 1;
            details.kontra_reached_30_against_black = 1;
        }

        return (re_game_points, kontra_game_points, details);
    }

}

#[cfg(test)]
mod tests {
    use crate::announcement::announcement_set::FdoAnnouncementSet;
    use super::*;

    #[test]
    fn test_calc_basis_draw_points() {
        // Es werden keine Punkte vergeben.
        assert_eq!(
            FdoBasicDrawPointsDetails::calculate(
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                120,
                120,
            ),
            (0, 0, FdoBasicDrawPointsDetails {
                winning_under_90_re: 0,
                winning_under_60_re: 0,
                winning_under_30_re: 0,

                winning_under_90_kontra: 0,
                winning_under_60_kontra: 0,
                winning_under_30_kontra: 0,

                re_reached_120_against_no_90: 0,
                re_reached_90_against_no_60: 0,
                re_reached_60_against_no_30: 0,
                re_reached_30_against_black: 0,

                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            })
        );

        // Re-Partei hat mehr als 211 Punkte gegen "schwarz" der Kontra-Partei erreicht.
        assert_eq!(
            FdoBasicDrawPointsDetails::calculate(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                211,
                29,
            ),
            (
                7,
                -7,
                FdoBasicDrawPointsDetails {
                    winning_under_90_re: 1,
                    winning_under_60_re: 1,
                    winning_under_30_re: 1,

                    winning_under_90_kontra: 0,
                    winning_under_60_kontra: 0,
                    winning_under_30_kontra: 0,

                    re_reached_120_against_no_90: 1,
                    re_reached_90_against_no_60: 1,
                    re_reached_60_against_no_30: 1,
                    re_reached_30_against_black: 1,

                    kontra_reached_120_against_no_90: 0,
                    kontra_reached_90_against_no_60: 0,
                    kontra_reached_60_against_no_30: 0,
                    kontra_reached_30_against_black: 0,
                }
            )
        );

        // Kontra hat mehr als 211 Punkte gegen "schwarz" der Re-Partei erreicht.
        assert_eq!(
            FdoBasicDrawPointsDetails::calculate(
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                FdoAnnouncementSet::all_higher_than(Some(FdoAnnouncement::Black)),
                29,
                211,
            ),
            (
                -7,
                7,
                FdoBasicDrawPointsDetails {
                    winning_under_90_re: 0,
                    winning_under_60_re: 0,
                    winning_under_30_re: 0,

                    winning_under_90_kontra: 1,
                    winning_under_60_kontra: 1,
                    winning_under_30_kontra: 1,

                    re_reached_120_against_no_90: 0,
                    re_reached_90_against_no_60: 0,
                    re_reached_60_against_no_30: 0,
                    re_reached_30_against_black: 0,

                    kontra_reached_120_against_no_90: 1,
                    kontra_reached_90_against_no_60: 1,
                    kontra_reached_60_against_no_30: 1,
                    kontra_reached_30_against_black: 1,
                }
            )
        )

    }
}
