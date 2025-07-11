use crate::full_doko::var1::player::encode_player_or_none;
use array_concat::concat_arrays;
use rs_full_doko::announcement::announcement::{FdoAnnouncement, FdoAnnouncementOccurrence};
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::player::player_set::FdoPlayerSet;

/// Anzahl der Ansagen im Spiel (für die Embedding-Größe).
pub const ANNOUNCEMENT_OR_NONE_COUNT: i64 = 6;

pub const NUM_CARD_POSITIONS: i64 = 48;
pub const NUM_POSTIONS_IN_CARD_POSITION: i64 = 10;

pub fn encode_announcement(announcement: Option<FdoAnnouncement>) -> [i64; 1] {
    fn map_announcement(announcement: Option<FdoAnnouncement>) -> i64 {
        match announcement {
            None => 0,
            Some(announcement) => match announcement {
                FdoAnnouncement::ReContra => 1,
                FdoAnnouncement::CounterReContra => 1,
                FdoAnnouncement::No90 => 2,
                FdoAnnouncement::No60 => 3,
                FdoAnnouncement::No30 => 4,
                FdoAnnouncement::Black => 5,
                FdoAnnouncement::NoAnnouncement => {
                    panic!("NoAnnouncement ist keine gültige Ansage.")
                }
            },
        }
    }

    let announcement_num = map_announcement(announcement);

    debug_assert!(announcement_num < 7);
    debug_assert!(announcement_num >= 0);

    [announcement_num]
}

pub fn encode_lowest_announcements(
    lowest_announcement_re: Option<FdoAnnouncement>,
    lowest_announcement_kontra: Option<FdoAnnouncement>,
) -> [i64; 2] {
    return concat_arrays!(
        encode_announcement(lowest_announcement_re),
        encode_announcement(lowest_announcement_kontra)
    );
}

pub fn encode_full_announcements(
    announcement_occurences: heapless::Vec<FdoAnnouncementOccurrence, 12>,

    current_player: FdoPlayer,
    re_players: FdoPlayerSet,
) -> ([i64; 10], [i64; 10], [i64; 10]) {
    // Re
    let mut re_announcement_card_position: [i64; 1] = [0];
    let mut re_announcement_position_in_card_position: [i64; 1] = [0];
    let mut re_announcement_player: [i64; 1] = [0];

    // Kontra
    let mut kontra_announcement_card_position: [i64; 1] = [0];
    let mut kontra_announcement_position_in_card_position: [i64; 1] = [0];
    let mut kontra_announcement_player: [i64; 1] = [0];

    // Re unter 90
    let mut re_announcement_card_position_under_90: [i64; 1] = [0];
    let mut re_announcement_position_in_card_position_under_90: [i64; 1] = [0];
    let mut re_announcement_player_under_90: [i64; 1] = [0];

    // Kontra unter 90
    let mut kontra_announcement_card_position_under_90: [i64; 1] = [0];
    let mut kontra_announcement_position_in_card_position_under_90: [i64; 1] = [0];
    let mut kontra_announcement_player_under_90: [i64; 1] = [0];

    // Re unter 60
    let mut re_announcement_card_position_under_60: [i64; 1] = [0];
    let mut re_announcement_position_in_card_position_under_60: [i64; 1] = [0];
    let mut re_announcement_player_under_60: [i64; 1] = [0];

    // Kontra unter 60
    let mut kontra_announcement_card_position_under_60: [i64; 1] = [0];
    let mut kontra_announcement_position_in_card_position_under_60: [i64; 1] = [0];
    let mut kontra_announcement_player_under_60: [i64; 1] = [0];

    // Re unter 30
    let mut re_announcement_card_position_under_30: [i64; 1] = [0];
    let mut re_announcement_position_in_card_position_under_30: [i64; 1] = [0];
    let mut re_announcement_player_under_30: [i64; 1] = [0];

    // Kontra unter 30
    let mut kontra_announcement_card_position_under_30: [i64; 1] = [0];
    let mut kontra_announcement_position_in_card_position_under_30: [i64; 1] = [0];
    let mut kontra_announcement_player_under_30: [i64; 1] = [0];

    // Re schwarz
    let mut re_announcement_card_position_under_0: [i64; 1] = [0];
    let mut re_announcement_position_in_card_position_under_0: [i64; 1] = [0];
    let mut re_announcement_player_under_0: [i64; 1] = [0];

    // Kontra schwarz
    let mut kontra_announcement_card_position_under_0: [i64; 1] = [0];
    let mut kontra_announcement_position_in_card_position_under_0: [i64; 1] = [0];
    let mut kontra_announcement_player_under_0: [i64; 1] = [0];

    let mut current_card_position: i64 = -1;
    let mut current_position_in_card_position: i64 = 0;

    for occurence in announcement_occurences {
        if current_card_position == occurence.card_index as i64 {
            current_position_in_card_position += 1;
        } else {
            current_card_position = occurence.card_index as i64;
            current_position_in_card_position = 0;
        }

        if occurence.announcement == FdoAnnouncement::ReContra
            || occurence.announcement == FdoAnnouncement::CounterReContra
        {
            if re_players.contains(occurence.player) {
                re_announcement_card_position = [occurence.card_index as i64];
                re_announcement_position_in_card_position = [current_position_in_card_position];
                re_announcement_player =
                    encode_player_or_none(occurence.player.into(), current_player);
            } else {
                kontra_announcement_card_position = [occurence.card_index as i64];
                kontra_announcement_position_in_card_position = [current_position_in_card_position];
                kontra_announcement_player =
                    encode_player_or_none(occurence.player.into(), current_player);
            }
        }

        if occurence.announcement == FdoAnnouncement::No90 {
            if re_players.contains(occurence.player) {
                re_announcement_card_position_under_90 = [occurence.card_index as i64];
                re_announcement_position_in_card_position_under_90 =
                    [current_position_in_card_position];
                re_announcement_player_under_90 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if re_announcement_player == [0] {
                    re_announcement_card_position = [occurence.card_index as i64];
                    re_announcement_position_in_card_position = [current_position_in_card_position];
                    re_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }
            } else {
                kontra_announcement_card_position_under_90 = [occurence.card_index as i64];
                kontra_announcement_position_in_card_position_under_90 =
                    [current_position_in_card_position];
                kontra_announcement_player_under_90 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if kontra_announcement_player == [0] {
                    kontra_announcement_card_position = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position = [current_position_in_card_position];
                    kontra_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }
            }
        }

        if occurence.announcement == FdoAnnouncement::No60 {
            if re_players.contains(occurence.player) {
                re_announcement_card_position_under_60 = [occurence.card_index as i64];
                re_announcement_position_in_card_position_under_60 =
                    [current_position_in_card_position];
                re_announcement_player_under_60 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if re_announcement_player == [0] {
                    re_announcement_card_position = [occurence.card_index as i64];
                    re_announcement_position_in_card_position = [current_position_in_card_position];
                    re_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_90 == [0] {
                    re_announcement_card_position_under_90 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    re_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }
            } else {
                kontra_announcement_card_position_under_60 = [occurence.card_index as i64];
                kontra_announcement_position_in_card_position_under_60 =
                    [current_position_in_card_position];
                kontra_announcement_player_under_60 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if kontra_announcement_player == [0] {
                    kontra_announcement_card_position = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position = [current_position_in_card_position];
                    kontra_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_90 == [0] {
                    kontra_announcement_card_position_under_90 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    kontra_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }
            }
        }

        if occurence.announcement == FdoAnnouncement::No30 {
            if re_players.contains(occurence.player) {
                re_announcement_card_position_under_30 = [occurence.card_index as i64];
                re_announcement_position_in_card_position_under_30 =
                    [current_position_in_card_position];
                re_announcement_player_under_30 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if re_announcement_player == [0] {
                    re_announcement_card_position = [occurence.card_index as i64];
                    re_announcement_position_in_card_position = [current_position_in_card_position];
                    re_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_90 == [0] {
                    re_announcement_card_position_under_90 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    re_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_60 == [0] {
                    re_announcement_card_position_under_60 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_60 = [current_position_in_card_position];
                    re_announcement_player_under_60 = encode_player_or_none(occurence.player.into(), current_player);
                }

            } else {
                kontra_announcement_card_position_under_30 = [occurence.card_index as i64];
                kontra_announcement_position_in_card_position_under_30 =
                    [current_position_in_card_position];
                kontra_announcement_player_under_30 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if kontra_announcement_player == [0] {
                    kontra_announcement_card_position = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position = [current_position_in_card_position];
                    kontra_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_90 == [0] {
                    kontra_announcement_card_position_under_90 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    kontra_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_60 == [0] {
                    kontra_announcement_card_position_under_60 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_60 = [current_position_in_card_position];
                    kontra_announcement_player_under_60 = encode_player_or_none(occurence.player.into(), current_player);
                }
            }
        }

        if occurence.announcement == FdoAnnouncement::Black {
            if re_players.contains(occurence.player) {
                re_announcement_card_position_under_0 = [occurence.card_index as i64];
                re_announcement_position_in_card_position_under_0 =
                    [current_position_in_card_position];
                re_announcement_player_under_0 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if re_announcement_player == [0] {
                    re_announcement_card_position = [occurence.card_index as i64];
                    re_announcement_position_in_card_position = [current_position_in_card_position];
                    re_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_90 == [0] {
                    re_announcement_card_position_under_90 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    re_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_60 == [0] {
                    re_announcement_card_position_under_60 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_60 = [current_position_in_card_position];
                    re_announcement_player_under_60 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if re_announcement_player_under_30 == [0] {
                    re_announcement_card_position_under_30 = [occurence.card_index as i64];
                    re_announcement_position_in_card_position_under_30 = [current_position_in_card_position];
                    re_announcement_player_under_30 = encode_player_or_none(occurence.player.into(), current_player);
                }
            } else {
                kontra_announcement_card_position_under_0 = [occurence.card_index as i64];
                kontra_announcement_position_in_card_position_under_0 =
                    [current_position_in_card_position];
                kontra_announcement_player_under_0 =
                    encode_player_or_none(occurence.player.into(), current_player);

                if kontra_announcement_player == [0] {
                    kontra_announcement_card_position = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position = [current_position_in_card_position];
                    kontra_announcement_player = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_90 == [0] {
                    kontra_announcement_card_position_under_90 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_90 = [current_position_in_card_position];
                    kontra_announcement_player_under_90 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_60 == [0] {
                    kontra_announcement_card_position_under_60 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_60 = [current_position_in_card_position];
                    kontra_announcement_player_under_60 = encode_player_or_none(occurence.player.into(), current_player);
                }

                if kontra_announcement_player_under_30 == [0] {
                    kontra_announcement_card_position_under_30 = [occurence.card_index as i64];
                    kontra_announcement_position_in_card_position_under_30 = [current_position_in_card_position];
                    kontra_announcement_player_under_30 = encode_player_or_none(occurence.player.into(), current_player);
                }
            }
        }
    }

    let player_encodings = concat_arrays!(
        re_announcement_player,
        kontra_announcement_player,
        re_announcement_player_under_90,
        kontra_announcement_player_under_90,
        re_announcement_player_under_60,
        kontra_announcement_player_under_60,
        re_announcement_player_under_30,
        kontra_announcement_player_under_30,
        re_announcement_player_under_0,
        kontra_announcement_player_under_0
    );

    let position_encodings = concat_arrays!(
        re_announcement_card_position,
        kontra_announcement_card_position,
        re_announcement_card_position_under_90,
        kontra_announcement_card_position_under_90,
        re_announcement_card_position_under_60,
        kontra_announcement_card_position_under_60,
        re_announcement_card_position_under_30,
        kontra_announcement_card_position_under_30,
        re_announcement_card_position_under_0,
        kontra_announcement_card_position_under_0
    );

    let position_in_position_encodings: [i64; 10] = concat_arrays!(
        re_announcement_position_in_card_position,
        kontra_announcement_position_in_card_position,

        re_announcement_position_in_card_position_under_90,
        kontra_announcement_position_in_card_position_under_90,

        re_announcement_position_in_card_position_under_60,
        kontra_announcement_position_in_card_position_under_60,

        re_announcement_position_in_card_position_under_30,
        kontra_announcement_position_in_card_position_under_30,

        re_announcement_position_in_card_position_under_0,
        kontra_announcement_position_in_card_position_under_0
    );

    (player_encodings, position_encodings, position_in_position_encodings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_full_doko::announcement::announcement::FdoAnnouncementOccurrence;

    #[test]
    fn test_encode_announcement() {
        assert_eq!(encode_announcement(None), [0]);
        assert_eq!(encode_announcement(Some(FdoAnnouncement::ReContra)), [1]);
        assert_eq!(
            encode_announcement(Some(FdoAnnouncement::CounterReContra)),
            [1]
        );
        assert_eq!(encode_announcement(Some(FdoAnnouncement::No90)), [2]);
        assert_eq!(encode_announcement(Some(FdoAnnouncement::No60)), [3]);
        assert_eq!(encode_announcement(Some(FdoAnnouncement::No30)), [4]);
        assert_eq!(encode_announcement(Some(FdoAnnouncement::Black)), [5]);
    }

    #[test]
    fn test_encode_lowest_announcements() {
        assert_eq!(encode_lowest_announcements(None, None), [0, 0]);
        assert_eq!(
            encode_lowest_announcements(Some(FdoAnnouncement::ReContra), None),
            [1, 0]
        );
        assert_eq!(
            encode_lowest_announcements(Some(FdoAnnouncement::No90), Some(FdoAnnouncement::No60)),
            [2, 3]
        );
        assert_eq!(
            encode_lowest_announcements(Some(FdoAnnouncement::Black), Some(FdoAnnouncement::No30)),
            [5, 4]
        );
    }

    #[test]
    fn test_encode_full_announcements_empty() {
        let occurences = heapless::Vec::new();
        let current_player = FdoPlayer::BOTTOM;
        let re_players = FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]);

        let (player_encodings, position_encodings, position_in_position_encodings) =
            encode_full_announcements(occurences, current_player, re_players);

        assert_eq!(player_encodings, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            position_encodings,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            position_in_position_encodings,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_encode_full_announcements_all() {
        let occurences = heapless::Vec::from_slice(&[
            FdoAnnouncementOccurrence {
                card_index: 0,
                player: FdoPlayer::BOTTOM,
                announcement: FdoAnnouncement::ReContra,
            },
            FdoAnnouncementOccurrence {
                card_index: 0,
                player: FdoPlayer::TOP,
                announcement: FdoAnnouncement::No90,
            },
            FdoAnnouncementOccurrence {
                card_index: 0,
                player: FdoPlayer::BOTTOM,
                announcement: FdoAnnouncement::No60,
            },

            FdoAnnouncementOccurrence {
                card_index: 5,
                player: FdoPlayer::RIGHT,
                announcement: FdoAnnouncement::CounterReContra,
            },

            FdoAnnouncementOccurrence {
                card_index: 8,
                player: FdoPlayer::LEFT,
                announcement: FdoAnnouncement::No30,
            },

            FdoAnnouncementOccurrence {
                card_index: 11,
                player: FdoPlayer::BOTTOM,
                announcement: FdoAnnouncement::Black,
            },
        ]).unwrap();
        let current_player = FdoPlayer::TOP;
        let re_players = FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]);

        let (player_encodings, position_encodings, position_in_position_encodings) =
            encode_full_announcements(occurences, current_player, re_players);

        assert_eq!(player_encodings, [
            // Re
            3,
            // Kontra
            2,
            // Re unter 90
            1,
            // Kontra unter 90
            4,
            // Re unter 60
            3,
            // Kontra unter 60
            4,
            // Re unter 30
            3,
            // Kontra unter 30
            4,
            // Re schwarz
            3,
            // Kontra schwarz
            0,
        ]);
        assert_eq!(
            position_encodings,
            [
                // Re
                0,
                // Kontra
                5,
                // Re unter 90
                0,
                // Kontra unter 90
                8,
                // Re unter 60
                0,
                // Kontra unter 60
                8,
                // Re unter 30
                11,
                // Kontra unter 30
                8,
                // Re schwarz
                11,
                // Kontra schwarz
                0,
            ]
        );
        assert_eq!(
            position_in_position_encodings,
            [
                // Re
                0,
                // Kontra
                0,
                // Re unter 90
                1,
                // Kontra unter 90
                0,
                // Re unter 60
                2,
                // Kontra unter 60
                0,
                // Re unter 30
                0,
                // Kontra unter 30
                0,
                // Re schwarz
                0,
                // Kontra schwarz
                0
            ]

        )
    }
}
