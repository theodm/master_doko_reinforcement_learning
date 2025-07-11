use serde::{Deserialize, Serialize};
use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoBasicWinningPointsDetails {
    // 7.1.4 DKV-TR
    // (a)
    pub winning_points: i32,
    pub winning_under_90: i32,
    pub winning_under_60: i32,
    pub winning_under_30: i32,
    pub winning_black: i32,

    // (b)
    pub re_announcement: i32,
    pub kontra_announcement: i32,

    // (c)
    pub re_under_90_announcement: i32,
    pub re_under_60_announcement: i32,
    pub re_under_30_announcement: i32,
    pub re_black_announcement: i32,

    // (d)
    pub kontra_under_90_announcement: i32,
    pub kontra_under_60_announcement: i32,
    pub kontra_under_30_announcement: i32,
    pub kontra_black_announcement: i32,

    // (e)
    pub re_reached_120_against_no_90: i32,
    pub re_reached_90_against_no_60: i32,
    pub re_reached_60_against_no_30: i32,
    pub re_reached_30_against_black: i32,

    // (f)
    pub kontra_reached_120_against_no_90: i32,
    pub kontra_reached_90_against_no_60: i32,
    pub kontra_reached_60_against_no_30: i32,
    pub kontra_reached_30_against_black: i32,
}

impl FdoBasicWinningPointsDetails {

    /// Grund-Punkteberechnung im Falle, dass es ein Gewinner-Partei gibt. Hier gilt 7.2.2 DKV-TR. Für
    /// die vollständige Punkteberechnung müssen noch die Sonderpunkte berechnet werden.
    pub fn calculate(
        winner_points: u32,
        looser_points: u32,
        winner_party_won_all_tricks: bool,
        re_previous_announcements: FdoAnnouncementSet,
        kontra_previous_announcements: FdoAnnouncementSet,
        re_points: u32,
        kontra_points: u32,
    ) -> (i32, i32, FdoBasicWinningPointsDetails) {
        debug_assert!(
            winner_points == re_points || winner_points == kontra_points,
            "Die Gewinner-Partei hat nicht die richtige Punktzahl."
        );
        debug_assert!(
            looser_points == re_points || looser_points == kontra_points,
            "Die Verlierer-Partei hat nicht die richtige Punktzahl."
        );
        debug_assert!(
            winner_points + looser_points == 240,
            "Die Punktzahl der Gewinner- und Verlierer-Partei ist nicht korrekt."
        );
        debug_assert!(
            re_points + kontra_points == 240,
            "Die Punktzahl der Re- und Kontra-Partei ist nicht korrekt."
        );

        let mut details = FdoBasicWinningPointsDetails {
            winning_points: 0,
            winning_under_90: 0,
            winning_under_60: 0,
            winning_under_30: 0,
            winning_black: 0,
            re_announcement: 0,
            kontra_announcement: 0,
            re_under_90_announcement: 0,
            re_under_60_announcement: 0,
            re_under_30_announcement: 0,
            re_black_announcement: 0,
            kontra_under_90_announcement: 0,
            kontra_under_60_announcement: 0,
            kontra_under_30_announcement: 0,
            kontra_black_announcement: 0,
            re_reached_120_against_no_90: 0,
            re_reached_90_against_no_60: 0,
            re_reached_60_against_no_30: 0,
            re_reached_30_against_black: 0,
            kontra_reached_120_against_no_90: 0,
            kontra_reached_90_against_no_60: 0,
            kontra_reached_60_against_no_30: 0,
            kontra_reached_30_against_black: 0,
        };

        let re_announcement = re_previous_announcements.contains(FdoAnnouncement::ReContra)
            || re_previous_announcements.contains(FdoAnnouncement::CounterReContra);
        let re_under_90_announcement = re_previous_announcements.contains(FdoAnnouncement::No90);
        let re_under_60_announcement = re_previous_announcements.contains(FdoAnnouncement::No60);
        let re_under_30_announcement = re_previous_announcements.contains(FdoAnnouncement::No30);
        let re_black_announcement = re_previous_announcements.contains(FdoAnnouncement::Black);

        let kontra_announcement = kontra_previous_announcements.contains(FdoAnnouncement::ReContra)
            || kontra_previous_announcements.contains(FdoAnnouncement::CounterReContra);
        let kontra_under_90_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No90);
        let kontra_under_60_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No60);
        let kontra_under_30_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No30);
        let kontra_black_announcement = kontra_previous_announcements.contains(FdoAnnouncement::Black);

        let mut winner_party_points = 0;
        let mut looser_party_points = 0;

        // (a)
        // Gewonnen: 1 Punkt als Grundwert
        winner_party_points += 1;
        looser_party_points -= 1;
        details.winning_points = 1;

        // unter 90 gespielt: 1 Punkt zusätzlich
        if looser_points < 90 {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.winning_under_90 = 1;
        }

        // unter 60 gespielt: 1 Punkt zusätzlich
        if looser_points < 60 {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.winning_under_60 = 1;
        }

        // unter 30 gespielt: 1 Punkt zusätzlich
        if looser_points < 30 {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.winning_under_30 = 1;
        }

        // schwarz gespielt: 1 Punkt zusätzlich
        if winner_party_won_all_tricks {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.winning_black = 1;
        }

        // (b)
        // "Re" angesagt: 2 Punkte zusätzlich
        if re_announcement {
            winner_party_points += 2;
            looser_party_points -= 2;
            details.re_announcement = 2;
        }

        // "Kontra" angesagt: 2 Punkte zusätzlich
        if kontra_announcement {
            winner_party_points += 2;
            looser_party_points -= 2;
            details.kontra_announcement = 2;
        }

        // (c) Es wurde von der Re-Partei:
        // "keine 90" abgesagt
        if re_under_90_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_under_90_announcement = 1;
        }

        // "keine 60" abgesagt
        if re_under_60_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_under_60_announcement = 1;
        }

        // "keine 30" abgesagt
        if re_under_30_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_under_30_announcement = 1;
        }

        // "schwarz" abgesagt
        if re_black_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_black_announcement = 1;
        }

        // (d) Es wurde von der Kontra-Partei:
        // "keine 90" abgesagt
        if kontra_under_90_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_under_90_announcement = 1;
        }

        // "keine 60" abgesagt
        if kontra_under_60_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_under_60_announcement = 1;
        }

        // "keine 30" abgesagt
        if kontra_under_30_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_under_30_announcement = 1;
        }

        // "schwarz" abgesagt
        if kontra_black_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_black_announcement = 1;
        }

        // (e) Es wurden von der Re-Partei:
        // 120 Augen gegen "keine 90" erreicht
        if re_points >= 120 && kontra_under_90_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_reached_120_against_no_90 = 1;
        }

        // 90 Augen gegen "keine 60" erreicht
        if re_points >= 90 && kontra_under_60_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_reached_90_against_no_60 = 1;
        }

        // 60 Augen gegen "keine 30" erreicht
        if re_points >= 60 && kontra_under_30_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_reached_60_against_no_30 = 1;
        }

        // 30 Augen gegen "schwarz" erreicht
        if re_points >= 30 && kontra_black_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.re_reached_30_against_black = 1;
        }

        // (f) Es wurden von der Kontra-Partei:
        // 120 Augen gegen "keine 90" erreicht
        if kontra_points >= 120 && re_under_90_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_reached_120_against_no_90 = 1;
        }

        // 90 Augen gegen "keine 60" erreicht
        if kontra_points >= 90 && re_under_60_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_reached_90_against_no_60 = 1;
        }

        // 60 Augen gegen "keine 30" erreicht
        if kontra_points >= 60 && re_under_30_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_reached_60_against_no_30 = 1;
        }

        // 30 Augen gegen "schwarz" erreicht
        if kontra_points >= 30 && re_black_announcement {
            winner_party_points += 1;
            looser_party_points -= 1;
            details.kontra_reached_30_against_black = 1;
        }

        (winner_party_points, looser_party_points, details)

    }
}

#[cfg(test)]
mod tests {
    use crate::announcement::announcement::FdoAnnouncement;
    use crate::announcement::announcement_set::FdoAnnouncementSet;
    use crate::stats::basic_points::basic_winning_points::FdoBasicWinningPointsDetails;

    #[test]
    fn test_calc_basic_winning_points() {
        // Kontra hat mit 120 Punkten gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                120,
                120,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                120,
                120,
            ),
            (1, -1, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit 121 Punkten gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                121,
                119,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                121,
                119,
            ),
            (1, -1, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit 151 Punkten gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                151,
                89,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                151,
                89,
            ),
            (2, -2, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit 181 Punkten gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                181,
                59,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                181,
                59,
            ),
            (3, -3, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit 211 Punkten gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                211,
                29,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                211,
                29,
            ),
            (4, -4, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit 240 Punkten gewonnen (und auch nur bis unter 30, da nicht alle Stiche gewonnen)
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                240,
                0,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                240,
                0,
            ),
            (4, -4, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat mit allen Stichen gewonnen
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                240,
                0,
                true,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::new(),
                240,
                0,
            ),
            (5, -5, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 1,

                re_announcement: 0,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Re hat 121 Punkte und "Re" angesagt
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                121,
                119,
                false,
                FdoAnnouncementSet::from_vec(
                    vec![FdoAnnouncement::ReContra]
                ),
                FdoAnnouncementSet::new(),
                121,
                119,
            ),
            (3, -3, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 2,
                kontra_announcement: 0,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Kontra hat 121 Punkte und "Kontra" angesagt
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                121,
                119,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                121,
                119,
            ),
            (3, -3, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 2,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Alle Absagen der Re-Partei wurden erreicht.
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                240,
                0,
                true,
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                FdoAnnouncementSet::new(),
                240,
                0,
            ),
            (11, -11, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 1,

                re_announcement: 2,
                kontra_announcement: 0,

                re_under_90_announcement: 1,
                re_under_60_announcement: 1,
                re_under_30_announcement: 1,
                re_black_announcement: 1,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

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

        // Alle Absagen der Kontra-Partei wurden erreicht.
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                240,
                0,
                true,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                0,
                240,
            ),
            (11, -11, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 1,

                re_announcement: 0,
                kontra_announcement: 2,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 1,
                kontra_under_60_announcement: 1,
                kontra_under_30_announcement: 1,
                kontra_black_announcement: 1,

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

        // Es wurden von der Re-Partei 120 Augen gegen "keine 90" erreicht
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                120,
                120,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                ]),
                120,
                120,
            ),
            (5, -5, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 2,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 1,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

                re_reached_120_against_no_90: 1,
                re_reached_90_against_no_60: 0,
                re_reached_60_against_no_30: 0,
                re_reached_30_against_black: 0,

                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            })
        );

        // Es wurden von der Re-Partei 90 Augen gegen "keine 60" erreicht
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                90,
                150,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60
                ]),
                90,
                150,
            ),
            (6, -6, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 2,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 1,
                kontra_under_60_announcement: 1,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

                re_reached_120_against_no_90: 0,
                re_reached_90_against_no_60: 1,
                re_reached_60_against_no_30: 0,
                re_reached_30_against_black: 0,

                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            })
        );

        // Es wurden von der Re-Partei 120 Augen gegen "schwarz" erreicht
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                120,
                120,
                false,
                FdoAnnouncementSet::new(),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                120,
                120,
            ),
            (11, -11, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 0,
                kontra_announcement: 2,

                re_under_90_announcement: 0,
                re_under_60_announcement: 0,
                re_under_30_announcement: 0,
                re_black_announcement: 0,

                kontra_under_90_announcement: 1,
                kontra_under_60_announcement: 1,
                kontra_under_30_announcement: 1,
                kontra_black_announcement: 1,

                re_reached_120_against_no_90: 1,
                re_reached_90_against_no_60: 1,
                re_reached_60_against_no_30: 1,
                re_reached_30_against_black: 1,

                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            })
        );

        // Es wurden von der Kontra-Partei 120 Augen gegen "schwarz" erreicht
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                120,
                120,
                false,
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                FdoAnnouncementSet::new(),
                120,
                120,
            ),
            (11, -11, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 0,
                winning_under_60: 0,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 2,
                kontra_announcement: 0,

                re_under_90_announcement: 1,
                re_under_60_announcement: 1,
                re_under_30_announcement: 1,
                re_black_announcement: 1,

                kontra_under_90_announcement: 0,
                kontra_under_60_announcement: 0,
                kontra_under_30_announcement: 0,
                kontra_black_announcement: 0,

                re_reached_120_against_no_90: 0,
                re_reached_90_against_no_60: 0,
                re_reached_60_against_no_30: 0,
                re_reached_30_against_black: 0,

                kontra_reached_120_against_no_90: 1,
                kontra_reached_90_against_no_60: 1,
                kontra_reached_60_against_no_30: 1,
                kontra_reached_30_against_black: 1,
            })
        );

        // Es wurden die maximale Punktzahl durch Re erreicht
        assert_eq!(
            FdoBasicWinningPointsDetails::calculate(
                240,
                0,
                true,
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black
                ]
                ),
                240,
                0,
            ),
            (21, -21, FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 1,
                winning_black: 1,

                re_announcement: 2,
                kontra_announcement: 2,

                re_under_90_announcement: 1,
                re_under_60_announcement: 1,
                re_under_30_announcement: 1,
                re_black_announcement: 1,

                kontra_under_90_announcement: 1,
                kontra_under_60_announcement: 1,
                kontra_under_30_announcement: 1,
                kontra_black_announcement: 1,

                re_reached_120_against_no_90: 1,
                re_reached_90_against_no_60: 1,
                re_reached_60_against_no_30: 1,
                re_reached_30_against_black: 1,

                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            })
        );


    }

}
