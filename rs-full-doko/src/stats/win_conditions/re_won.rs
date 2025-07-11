use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;

/// Gibt an ob Re nach den Regeln in 7.1.2 DKV-TR gewonnen hat.
pub fn re_won(
    re_eyes: u32,

    re_previous_announcements: FdoAnnouncementSet,
    kontra_previous_announcements: FdoAnnouncementSet,

    re_won_all_tricks: bool,
    kontra_won_all_tricks: bool,
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

    // Aus den Regeln (7.1.2)
    // Man könnte es sicherlich vereinfachen, aber wenn wir uns
    // sturr an die Regeln halten, werden wohl alle Fälle abgedeckt.

    // Re gewinnt mit dem 121. Auge, wenn keine Ansage bzw. Absage getroffen wurde.
    if re_eyes >= 121 && only_re_no_announcement && only_kontra_no_announcement {
        return true;
    }

    // Re gewinnt mit dem 121. Auge, wenn nur "Re" angesagt wurde. (unabhängig von der Reihenfolge)
    if re_eyes >= 121 && only_re_announcement && !kontra_announcement {
        return true;
    }

    // Re gewinnt mit dem 121. Auge, wenn "Re" und "Kontra" angesagt wurde.
    if re_eyes >= 121 && only_re_announcement && only_kontra_announcement {
        return true;
    }

    // Re gewinnt mit dem 120. Auge, wenn nur "Kontra" angesagt wurde.
    if re_eyes >= 120 && !re_announcement && only_kontra_announcement {
        return true;
    }

    // Re gewinnt mit dem 151. Auge, wenn sie der Kontra-Partei "keine 90" abgesagt hat.
    if re_eyes >= 151 && only_re_under_90_announcement {
        return true;
    }

    // Re gewinnt mit dem 181. Auge, wenn sie der Kontra-Partei "keine 60" abgesagt hat.
    if re_eyes >= 181 && only_re_under_60_announcement {
        return true;
    }

    // Re gewinnt mit dem 211. Auge, wenn sie der Kontra-Partei "keine 30" abgesagt hat.
    if re_eyes >= 211 && only_re_under_30_announcement {
        return true;
    }

    // Re gewinnt falls sie alle Stiche erhalten hat wenn sie der Kontra-Partei "schwarz" abgesagt hat.
    if re_won_all_tricks && only_re_black_announcement {
        return true;
    }

    // Re gewinnt mit dem 90. Auge, wenn von der Kontra-Partei "keine 90" abgesagt
    // wurde und die Re-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if re_eyes >= 90 && only_kontra_under_90_announcement && !re_under_90_announcement {
        return true;
    }

    // Re gewinnt mit dem 60. Auge, wenn von der Kontra-Partei "keine 60" abgesagt
    // wurde und die Re-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if re_eyes >= 60 && only_kontra_under_60_announcement && !re_under_90_announcement {
        return true;
    }

    // Re gewinnt mit dem 30. Auge, wenn von der Kontra-Partei "keine 30" abgesagt
    // wurde und die Re-Partei sich nicht durch eine Absage zu einer
    // höheren Augenzahl verpflichtet hat.
    if re_eyes >= 30 && only_kontra_under_30_announcement && !re_under_90_announcement {
        return true;
    }

    // Re gewinnt mit dem 1. Stich den sie erhält, wenn von der Kontra-Partei "schwarz"
    // abgesagt wurde und die Re-Partei sich nicht durch eine Absage zu einer höheren
    // Augenzahl verpflichtet hat.
    if !kontra_won_all_tricks && only_kontra_black_announcement && !re_under_90_announcement {
        return true;
    }

    // Ansonsten hat Re nicht gewonnen.
    false
}


#[test]
fn test_calc_re_won() {
    // 7.1.2 DKV-TR Nr. 1
    // Re hat 120 Augen und es wurde keine Ansage oder Absage getroffen.
    assert_eq!(
        re_won(
            120,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 121 Augen und es wurde keine Ansage oder Absage getroffen.
    assert_eq!(
        re_won(
            121,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 2
    // Re hat 120 Augen und es wurde nur "Re" angesagt.
    assert_eq!(
        re_won(
            120,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 121 Augen und es wurde nur "Re" angesagt.
    assert_eq!(
        re_won(
            121,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 3
    // Re hat 120 Augen und es wurde "Re" und "Kontra" angesagt.
    assert_eq!(
        re_won(
            120,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
            false,
            false
        ),
        false
    );

    // Re hat 121 Augen und es wurde "Re" und "Kontra" angesagt.
    assert_eq!(
        re_won(
            121,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::CounterReContra]),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 4
    // Re hat 119 Augen und es wurde nur "Kontra" angesagt.
    assert_eq!(
        re_won(
            119,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            false,
            false
        ),
        false
    );

    // Re hat 120 Augen und es wurde nur "Kontra" angesagt.
    assert_eq!(
        re_won(
            120,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::ReContra]),
            false,
            false
        ),
        true
    );
    // 7.1.2 DKV-TR Nr. 5
    // Re hat 150 Augen und Re hat "keine 90" abgesagt.
    assert_eq!(
        re_won(
            150,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 151 Augen und Re hat "keine 90" abgesagt.
    assert_eq!(
        re_won(
            151,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 6
    // Re hat 180 Augen und Re hat "keine 60" abgesagt.
    assert_eq!(
        re_won(
            180,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 181 Augen und Re hat "keine 60" abgesagt.
    assert_eq!(
        re_won(
            181,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 7
    // Re hat 210 Augen und Re hat "keine 30" abgesagt.
    assert_eq!(
        re_won(
            210,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60, FdoAnnouncement::No30]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 211 Augen und Re hat "keine 30" abgesagt.
    assert_eq!(
        re_won(
            211,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60, FdoAnnouncement::No30]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        true
    );

    // 7.1.2 DKV-TR Nr. 8
    // Re hat alle Stiche gemacht und Re hat "schwarz" abgesagt.
    assert_eq!(
        re_won(
            240,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60, FdoAnnouncement::No30, FdoAnnouncement::Black]),
            FdoAnnouncementSet::new(),
            true,
            false
        ),
        true
    );

    // Re hat nicht alle Stiche gemacht und Re hat "schwarz" abgesagt.
    assert_eq!(
        re_won(
            240,
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::No90, FdoAnnouncement::No60, FdoAnnouncement::No30, FdoAnnouncement::Black]),
            FdoAnnouncementSet::new(),
            false,
            false
        ),
        false
    );

    // Re hat 0 Augen und Kontra hat "schwarz" abgesagt.
    assert_eq!(
        re_won(
            0,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::Black]),
            false,
            true
        ),
        false
    );

    // Re hat einen Stich gemacht und Kontra hat "schwarz" abgesagt.
    assert_eq!(
        re_won(
            0,
            FdoAnnouncementSet::new(),
            FdoAnnouncementSet::from_vec(vec![FdoAnnouncement::Black]),
            false,
            false
        ),
        true
    );

}


