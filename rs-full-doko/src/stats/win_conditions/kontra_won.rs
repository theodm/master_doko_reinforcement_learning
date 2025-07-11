use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;

pub fn kontra_won(
    kontra_eyes: u32,

    re_previous_announcements: FdoAnnouncementSet,
    kontra_previous_announcements: FdoAnnouncementSet,

    re_won_all_tricks: bool,
    kontra_won_all_tricks: bool
) -> bool {
    let re_announcement = re_previous_announcements.contains(FdoAnnouncement::ReContra)
        || re_previous_announcements.contains(FdoAnnouncement::CounterReContra);
    let re_under_90_announcement = re_previous_announcements.contains(FdoAnnouncement::No90);
    let re_under_60_announcement = re_previous_announcements.contains(FdoAnnouncement::No60);
    let re_under_30_announcement = re_previous_announcements.contains(FdoAnnouncement::No30);
    let re_black_announcement = re_previous_announcements.contains(FdoAnnouncement::Black);

    let only_re_no_announcement = re_previous_announcements.len() == 0;
    let only_re_announcement = re_announcement && !re_under_90_announcement;
    let only_re_under_90_announcement = re_under_90_announcement && !re_under_60_announcement;
    let only_re_under_60_announcement = re_under_60_announcement && !re_under_30_announcement;
    let only_re_under_30_announcement = re_under_30_announcement && !re_black_announcement;
    let only_re_black_announcement = re_black_announcement;

    let kontra_announcement = kontra_previous_announcements.contains(FdoAnnouncement::ReContra)
        || kontra_previous_announcements.contains(FdoAnnouncement::CounterReContra);
    let kontra_under_90_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No90);
    let kontra_under_60_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No60);
    let kontra_under_30_announcement = kontra_previous_announcements.contains(FdoAnnouncement::No30);
    let kontra_black_announcement = kontra_previous_announcements.contains(FdoAnnouncement::Black);

    let only_kontra_no_announcement = kontra_previous_announcements.len() == 0;
    let only_kontra_announcement = kontra_announcement && !kontra_under_90_announcement;
    let only_kontra_under_90_announcement = kontra_under_90_announcement && !kontra_under_60_announcement;
    let only_kontra_under_60_announcement = kontra_under_60_announcement && !kontra_under_30_announcement;
    let only_kontra_under_30_announcement = kontra_under_30_announcement && !kontra_black_announcement;
    let only_kontra_black_announcement = kontra_black_announcement;

    // Aus den Regeln (7.1.3)
    // Man könnte es sicherlich vereinfachen, aber wenn wir uns
    // sturr an die Regeln halten, werden wohl alle Fälle abgedeckt.

    // Kontra gewinnt mit dem 120. Auge, wenn keine Ansage bzw. Absage getroffen wurde.
    if kontra_eyes >= 120 && only_re_no_announcement && only_kontra_no_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 120. Auge, wenn nur "Re" angesagt wurde.
    if kontra_eyes >= 120 && only_re_announcement && !kontra_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 120. Auge, wenn "Re" und "Kontra" angesagt wurde (unabhängig von der Reihenfolge).
    if kontra_eyes >= 120 && only_re_announcement && only_kontra_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 121. Auge, wenn nur "Kontra" angesagt wurde.
    if kontra_eyes >= 121 && !re_announcement && only_kontra_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 151. Auge, wenn sie der Re-Partei "keine 90" abgesagt hat.
    if kontra_eyes >= 151 && only_kontra_under_90_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 181. Auge, wenn sie der Re-Partei "keine 60" abgesagt hat.
    if kontra_eyes >= 181 && only_kontra_under_60_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 211. Auge, wenn sie der Re-Partei "keine 30" abgesagt hat.
    if kontra_eyes >= 211 && only_kontra_under_30_announcement {
        return true;
    }

    // Kontra gewinnt falls sie alle Stiche erhalten hat wenn sie der Re-Partei "schwarz" abgesagt hat.
    if kontra_won_all_tricks && only_kontra_black_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 90. Auge, wenn von der Re-Partei "keine 90" abgesagt
    // wurde und die Kontra-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if kontra_eyes >= 90 && only_re_under_90_announcement && !kontra_under_90_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 60. Auge, wenn von der Re-Partei "keine 60" abgesagt
    // wurde und die Kontra-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if kontra_eyes >= 60 && only_re_under_60_announcement && !kontra_under_90_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 30. Auge, wenn von der Re-Partei "keine 30" abgesagt
    // wurde und die Kontra-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if kontra_eyes >= 30 && only_re_under_30_announcement && !kontra_under_90_announcement {
        return true;
    }

    // Kontra gewinnt mit dem 1. Stich den sie erhält, wenn von der Re-Partei "schwarz" abgesagt
    // wurde und die Kontra-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if !re_won_all_tricks && only_re_black_announcement && !kontra_under_90_announcement {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::announcement::announcement::FdoAnnouncement;
    use crate::announcement::announcement_set::FdoAnnouncementSet;

    #[test]
    fn test_kontra_won() {
        // 7.1.3 DKV-TR Nr. 1
        // Kontra hat 119 Augen und es wurde keine Ansage oder Absage getroffen.
        assert_eq!(
            kontra_won(
                119,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![]),
                false,
                false
            ),
            false
        );

        // Kontra hat 120 Augen und es wurde keine Ansage oder Absage getroffen.
        assert_eq!(
            kontra_won(
                120,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![]),
                false,
                false
            ),
            true
        );

        // 7.1.3 DKV-TR Nr. 2
        // Kontra hat 119 Augen und nur Re wurde angesagt.
        assert_eq!(
            kontra_won(
                119,
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                FdoAnnouncementSet::from_vec(vec![]),
                false,
                false
            ),
            false
        );

        // Kontra hat 120 Augen und nur Re wurde angesagt.
        assert_eq!(
            kontra_won(
                120,
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                FdoAnnouncementSet::from_vec(vec![]),
                false,
                false
            ),
            true
        );

        // 7.1.3 DKV-TR Nr. 3
        // Kontra hat 119 Augen und Re und Kontra wurden angesagt.
        assert_eq!(
            kontra_won(
                119,
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
                false,
                false
            ),
            false
        );

        // Kontra hat 120 Augen und Re und Kontra wurden angesagt.
        assert_eq!(
            kontra_won(
                120,
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
                false,
                false
            ),
            true
        );

        // 7.1.3 DKV-TR Nr. 4
        // Kontra hat 120 Augen und nur Kontra wurde angesagt.
        assert_eq!(
            kontra_won(
                120,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                false,
                false
            ),
            false
        );

        // Kontra hat 121 Augen und nur Kontra wurde angesagt.
        assert_eq!(
            kontra_won(
                121,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                false,
                false
            ),
            true
        );

        // 7.1.3 DKV-TR Nr. 5
        // Kontra hat 150 Augen wenn sie der Re-Partei "keine 90" abgesagt hat.
        assert_eq!(
            kontra_won(
                150,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                false,
                false
            ),
            false
        );

        // Kontra hat 151 Augen wenn sie der Re-Partei "keine 90" abgesagt hat.
        assert_eq!(
            kontra_won(
                151,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                false,
                false
            ),
            true
        );

        // 7.1.3 DKV-TR Nr. 6
        // Kontra hat nicht alle Stiche erhalten und hat der Re-Partei "schwarz" abgesagt.
        assert_eq!(
            kontra_won(
                240,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                false,
                false
            ),
            false
        );

        // Kontra hat alle Stiche erhalten und hat der Re-Partei "schwarz" abgesagt.
        assert_eq!(
            kontra_won(
                240,
                FdoAnnouncementSet::from_vec(vec![]),
                FdoAnnouncementSet::from_vec(vec![
                    FdoAnnouncement::ReContra,
                    FdoAnnouncement::No90,
                    FdoAnnouncement::No60,
                    FdoAnnouncement::No30,
                    FdoAnnouncement::Black,
                ]),
                false,
                true
            ),
            true
        );

        fn test_kontra_won() {
            // 7.1.3 DKV-TR Nr. 1
            // Kontra hat 119 Augen und es wurde keine Ansage oder Absage getroffen.
            assert_eq!(
                kontra_won(
                    119,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 120 Augen und es wurde keine Ansage oder Absage getroffen.
            assert_eq!(
                kontra_won(
                    120,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 2
            // Kontra hat 119 Augen und nur Re wurde angesagt.
            assert_eq!(
                kontra_won(
                    119,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 120 Augen und nur Re wurde angesagt.
            assert_eq!(
                kontra_won(
                    120,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 3
            // Kontra hat 119 Augen und Re und Kontra wurden angesagt.
            assert_eq!(
                kontra_won(
                    119,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 120 Augen und Re und Kontra wurden angesagt.
            assert_eq!(
                kontra_won(
                    120,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 4
            // Kontra hat 120 Augen und nur Kontra wurde angesagt.
            assert_eq!(
                kontra_won(
                    120,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 121 Augen und nur Kontra wurde angesagt.
            assert_eq!(
                kontra_won(
                    121,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 5
            // Kontra hat 150 Augen wenn sie der Re-Partei "keine 90" abgesagt hat.
            assert_eq!(
                kontra_won(
                    150,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 151 Augen wenn sie der Re-Partei "keine 90" abgesagt hat.
            assert_eq!(
                kontra_won(
                    151,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 6
            // Kontra hat nicht alle Stiche erhalten und hat der Re-Partei "schwarz" abgesagt.
            assert_eq!(
                kontra_won(
                    240,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                        FdoAnnouncement::Black,
                    ]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat alle Stiche erhalten und hat der Re-Partei "schwarz" abgesagt.
            assert_eq!(
                kontra_won(
                    240,
                    FdoAnnouncementSet::from_vec(vec![]),
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                        FdoAnnouncement::Black,
                    ]),
                    false,
                    true
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 7
            // Kontra hat 89 Augen und es wurde von der Re-Partei "keine 90" abgesagt.
            assert_eq!(
                kontra_won(
                    89,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 90 Augen und es wurde von der Re-Partei "keine 90" abgesagt.
            assert_eq!(
                kontra_won(
                    90,
                    FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra, FdoAnnouncement::No90]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // Kontra hat 59 Augen und es wurde von der Re-Partei "keine 60" abgesagt.
            assert_eq!(
                kontra_won(
                    59,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 60 Augen und es wurde von der Re-Partei "keine 60" abgesagt.
            assert_eq!(
                kontra_won(
                    60,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // Kontra hat 29 Augen und es wurde von der Re-Partei "keine 30" abgesagt.
            assert_eq!(
                kontra_won(
                    29,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                false
            );

            // Kontra hat 30 Augen und es wurde von der Re-Partei "keine 30" abgesagt.
            assert_eq!(
                kontra_won(
                    30,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // 7.1.3 DKV-TR Nr. 8
            // Kontra hat keinen Stich erhalten und Re hat "schwarz" abgesagt. (keine Absage von Kontra)
            assert_eq!(
                kontra_won(
                    0,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                        FdoAnnouncement::Black,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    true,
                    false
                ),
                false
            );

            // Kontra hat einen Stich erhalten und Re hat "schwarz" abgesagt. (keine Absage von Kontra)
            assert_eq!(
                kontra_won(
                    0,
                    FdoAnnouncementSet::from_vec(vec![
                        FdoAnnouncement::ReContra,
                        FdoAnnouncement::No90,
                        FdoAnnouncement::No60,
                        FdoAnnouncement::No30,
                        FdoAnnouncement::Black,
                    ]),
                    FdoAnnouncementSet::from_vec(vec![]),
                    false,
                    false
                ),
                true
            );

            // Kontra hat keinen Stich erhalten und Re hat "schwarz" abgesagt. (Absage von Kontra)
            assert_eq!(
                kontra_won(
                    0,
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
                    ]),
                    true,
                    false
                ),
                false
            );

            // Kontra hat einen Stich erhalten und Re hat "schwarz" abgesagt. (Absage von Kontra)
            assert_eq!(
                kontra_won(
                    0,
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
                    ]),
                    false,
                    false
                ),
                false
            );
        }
    }
}
