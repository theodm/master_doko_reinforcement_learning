use serde::{Deserialize, Serialize};
use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;
use crate::basic::team::FdoTeam;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::stats::additional_points::additional_points::FdoAdditionalPointsDetails;
use crate::stats::basic_points::basic_draw_points::FdoBasicDrawPointsDetails;
use crate::stats::basic_points::basic_winning_points::FdoBasicWinningPointsDetails;
use crate::stats::win_conditions::kontra_won::kontra_won;
use crate::stats::win_conditions::re_won::re_won;
use crate::trick::trick::FdoTrick;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoEndOfGameStats {
    pub re_players: FdoPlayerSet,

    pub is_solo: bool,

    pub player_eyes: PlayerZeroOrientedArr<u32>,

    pub re_eyes: u32,
    pub kontra_eyes: u32,

    pub re_points: i32,
    pub kontra_points: i32,

    pub player_points: PlayerZeroOrientedArr<i32>,

    // Nur bei einem Gewinn einer Partei ist diese Struktur
    // gefüllt. Nicht bei einem Unentschieden.
    pub basic_winning_point_details: Option<FdoBasicWinningPointsDetails>,

    // Nur bei einem Unentschieden ist diese Struktur
    // gefüllt.
    pub basic_draw_points_details: Option<FdoBasicDrawPointsDetails>,

    // Nur in einem Spiel, in dem die Extra-Punkte gezählt
    // werden, ist diese Struktur gefüllt. In Solis werden
    // keine Extra-Punkte gezählt.
    pub additional_points_details: Option<FdoAdditionalPointsDetails>,
}

impl FdoEndOfGameStats {
    pub fn calculate(
        player_eyes: PlayerZeroOrientedArr<u32>,
        player_num_tricks: PlayerZeroOrientedArr<u32>,
        re_players: FdoPlayerSet,
        re_lowest_announcements: Option<FdoAnnouncement>,
        contra_lowest_announcements: Option<FdoAnnouncement>,
        tricks: &heapless::Vec<FdoTrick, 12>,
    ) -> FdoEndOfGameStats {
        let mut re_tricks = 0;
        let mut kontra_tricks = 0;

        let mut re_eyes = 0;
        let mut kontra_eyes = 0;

        // Wir rechnen die Punkte der Teams zusammen
        // und die Anzahl der Stiche, die sie gemacht haben.
        for player in FdoPlayerSet::all().iter() {
            if re_players.contains(player) {
                re_eyes += player_eyes[player];
                re_tricks += player_num_tricks[player];
            } else {
                kontra_eyes += player_eyes[player];
                kontra_tricks += player_num_tricks[player];
            }
        }

        let re_won_all_tricks = re_tricks == 12;
        let kontra_won_all_tricks = kontra_tricks == 12;

        let re_previous_announcements = FdoAnnouncementSet::all_higher_than(re_lowest_announcements);
        let kontra_previous_announcements = FdoAnnouncementSet::all_higher_than(contra_lowest_announcements);

        // Gibt an ob Re nach den Regeln in 7.1.2 DKV-TR gewonnen hat.
        let re_won = re_won(
            re_eyes,
            re_previous_announcements,
            kontra_previous_announcements,
            re_won_all_tricks,
            kontra_won_all_tricks,
        );

        // Gibt an ob Kontra nach den Regeln in 7.1.3 DKV-TR gewonnen hat.
        let kontra_won = kontra_won(
            kontra_eyes,
            re_previous_announcements,
            kontra_previous_announcements,
            re_won_all_tricks,
            kontra_won_all_tricks,
        );

        let is_solo = re_players.len() == 1;

        let winning_team = if re_won {
            Some(FdoTeam::Re)
        } else if kontra_won {
            Some(FdoTeam::Kontra)
        } else {
            None
        };

        // Es kann auch keine Partei gewonnen haben, dann wenn beide Parteien
        // ihre Absagen nicht erreicht haben.
        let none_won = !re_won && !kontra_won;

        // Beide Parteien können aber nicht gleichzeitig gewonnen haben.
        debug_assert!(!(re_won && kontra_won));

        let mut basic_winning_points_details = None;
        let mut basic_draw_points_details = None;
        let additional_points_details;

        let mut re_game_points;
        let kontra_game_points;

        if none_won {
            // Die Basispunkte nach 7.2.2 (a), (e) und (f) werden berechnet.
            let (re_game_basic_points, kontra_basic_points, _basic_draw_details) = FdoBasicDrawPointsDetails::calculate(
                re_previous_announcements,
                kontra_previous_announcements,
                re_eyes,
                kontra_eyes,
            );

            let (re_game_extra_points, kontra_game_extra_points, _extra_details) = if !is_solo {
                // Die Sonderpunkte werden berechnet.
                let (re_game_extra_points, kontra_game_extra_points, extra_details) = FdoAdditionalPointsDetails::calculate(
                    re_players,
                    winning_team,
                    tricks,
                );

                (re_game_extra_points, kontra_game_extra_points, Some(extra_details))
            } else {
                (0, 0, None)
            };

            re_game_points = re_game_basic_points + re_game_extra_points;
            kontra_game_points = kontra_basic_points + kontra_game_extra_points;

            basic_draw_points_details = Some(_basic_draw_details);
            additional_points_details = _extra_details;
        } else {
            let winner_eyes = if re_won {
                re_eyes
            } else {
                kontra_eyes
            };
            let looser_eyes = if re_won {
                kontra_eyes
            } else {
                re_eyes
            };

            let winner_party_won_all_tricks = if re_won {
                re_won_all_tricks
            } else if kontra_won {
                kontra_won_all_tricks
            } else {
                false
            };

            // Die Basispunkte nach 7.2.2 werden berechnet.
            let (winner_game_points, looser_game_points, _basic_winning_point_details) = FdoBasicWinningPointsDetails::calculate(
                winner_eyes,
                looser_eyes,
                winner_party_won_all_tricks,
                re_previous_announcements,
                kontra_previous_announcements,
                re_eyes,
                kontra_eyes,
            );

            let re_game_extra_points: i32;
            let kontra_game_extra_points: i32;
            let extra_details: Option<FdoAdditionalPointsDetails>;

            if !is_solo {
                // Die Sonderpunkte werden berechnet.
                let (_re_game_extra_points, _kontra_game_extra_points, _extra_details) = FdoAdditionalPointsDetails::calculate(
                    re_players,
                    winning_team,
                    tricks,
                );

                re_game_extra_points = _re_game_extra_points;
                kontra_game_extra_points = _kontra_game_extra_points;
                extra_details = Some(_extra_details);
            } else {
                re_game_extra_points = 0;
                kontra_game_extra_points = 0;
                extra_details = None;
            }

            re_game_points = if re_won {
                winner_game_points + re_game_extra_points
            } else {
                looser_game_points + re_game_extra_points
            };

            kontra_game_points = if kontra_won {
                winner_game_points + kontra_game_extra_points
            } else {
                looser_game_points + kontra_game_extra_points
            };

            basic_winning_points_details = Some(_basic_winning_point_details);
            additional_points_details = extra_details;
        }

        if is_solo {
            // Nach 7.2.4 DKV-TR wird die Punktzahl des Solo-Spielers verdreifacht.
            re_game_points = re_game_points * 3
        }

        let player_points = PlayerZeroOrientedArr::from_full([
            if re_players.contains(FdoPlayer::BOTTOM) { re_game_points } else { kontra_game_points },
            if re_players.contains(FdoPlayer::LEFT) { re_game_points } else { kontra_game_points },
            if re_players.contains(FdoPlayer::TOP) { re_game_points } else { kontra_game_points },
            if re_players.contains(FdoPlayer::RIGHT) { re_game_points } else { kontra_game_points },
        ]);

        FdoEndOfGameStats {
            re_players,
            is_solo,
            player_eyes,
            re_eyes,
            kontra_eyes,
            re_points: re_game_points,
            kontra_points: kontra_game_points,
            player_points: player_points,

            basic_winning_point_details: basic_winning_points_details,
            basic_draw_points_details,
            additional_points_details,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::announcement::announcement::FdoAnnouncement;
    use crate::card::cards::FdoCard;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::stats::additional_points::additional_points::FdoAdditionalPointsDetails;
    use crate::stats::basic_points::basic_draw_points::FdoBasicDrawPointsDetails;
    use crate::stats::basic_points::basic_winning_points::FdoBasicWinningPointsDetails;
    use crate::stats::stats::FdoEndOfGameStats;
    use crate::trick::trick::FdoTrick;
    use crate::util::po_zero_arr::PlayerZeroOrientedArr;

    #[test]
    fn test_end_of_game_stats_stilles_solo_re_gewinnt() {
        // 1. Testfall: Ein stilles Solo wird von Re gewonnen.
        // https://www.online-doppelkopf.com/spiele/95.949.649

        let actual = FdoEndOfGameStats::calculate(
            PlayerZeroOrientedArr::from_full([216, 0, 24, 0]),
            PlayerZeroOrientedArr::from_full([11, 0, 1, 0]),
            FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM]),
            Some(FdoAnnouncement::ReContra),
            None,
            &heapless::Vec::from_slice(&[
                // 1. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubAce, FdoCard::ClubTen, FdoCard::ClubNine, FdoCard::ClubNine],
                ),
                // 2. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartNine, FdoCard::HeartAce, FdoCard::DiamondJack, FdoCard::HeartAce],
                ),
                // 3. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![FdoCard::SpadeAce, FdoCard::SpadeKing, FdoCard::SpadeJack, FdoCard::SpadeNine],
                ),
                // 4. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::SpadeQueen, FdoCard::DiamondTen, FdoCard::DiamondNine, FdoCard::DiamondKing],
                ),
                // 5. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::SpadeQueen, FdoCard::DiamondTen, FdoCard::DiamondNine, FdoCard::DiamondKing],
                ),
                // 6. Stich (theoretisch Fuchs gefangen, aber wird im Solo nicht gezählt)
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubQueen, FdoCard::DiamondAce, FdoCard::DiamondNine, FdoCard::DiamondKing],
                ),
                // 7. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartTen, FdoCard::HeartJack, FdoCard::ClubJack, FdoCard::DiamondQueen],
                ),
                // 8. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartTen, FdoCard::SpadeJack, FdoCard::DiamondQueen, FdoCard::DiamondAce],
                ),
                // 9. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubQueen, FdoCard::HeartNine, FdoCard::HeartQueen, FdoCard::HeartQueen],
                ),
                // 10. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartKing, FdoCard::SpadeKing, FdoCard::SpadeNine, FdoCard::ClubKing],
                ),
                // 11. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartKing, FdoCard::SpadeTen, FdoCard::ClubTen, FdoCard::ClubKing],
                ),
                // 12. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubJack, FdoCard::SpadeTen, FdoCard::SpadeAce, FdoCard::ClubAce],
                ),
            ]).unwrap(),
        );

        let expected = FdoEndOfGameStats {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM]),
            is_solo: true,
            player_eyes: PlayerZeroOrientedArr::from_full([216, 0, 24, 0]),
            re_eyes: 216,
            kontra_eyes: 24,
            re_points: 18,
            kontra_points: -6,
            player_points: PlayerZeroOrientedArr::from_full([18, -6, -6, -6]),
            basic_winning_point_details: Some(FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_30: 1,
                winning_under_60: 1,
                winning_under_90: 1,
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
            }),

            // Kein Unentschieden
            basic_draw_points_details: None,
            // Keine Extra-Punkte im Solo.
            additional_points_details: None,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_end_of_game_stats_angesagtes_solo_unentschieden() {
        // 2. Testfall: Ein angesagtes Solo endet in einem Unentschieden.
        // https://www.online-doppelkopf.com/spiele/82.288.742 (aber modifiziert, ein Unentschieden kommt selten vor)

        let actual = FdoEndOfGameStats::calculate(
            PlayerZeroOrientedArr::from_full([136, 35, 0, 69]),
            PlayerZeroOrientedArr::from_full([9, 1, 0, 2]),
            FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM]),
            Some(FdoAnnouncement::No90),
            Some(FdoAnnouncement::No90),
            &heapless::Vec::from_slice(&[
                // 1. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubQueen, FdoCard::ClubJack, FdoCard::DiamondJack, FdoCard::DiamondQueen],
                ),
                // 2. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::DiamondQueen, FdoCard::DiamondTen, FdoCard::DiamondAce, FdoCard::DiamondKing],
                ),
                // 3. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartQueen, FdoCard::HeartJack, FdoCard::DiamondJack, FdoCard::SpadeJack],
                ),
                // 4. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartQueen, FdoCard::SpadeKing, FdoCard::SpadeJack, FdoCard::SpadeKing],
                ),
                // 5. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::SpadeQueen, FdoCard::HeartKing, FdoCard::ClubJack, FdoCard::ClubKing],
                ),
                // 6. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartJack, FdoCard::HeartTen, FdoCard::HeartTen, FdoCard::HeartAce],
                ),
                // 7. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![FdoCard::DiamondAce, FdoCard::DiamondTen, FdoCard::SpadeAce, FdoCard::DiamondKing],
                ),
                // 8. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![FdoCard::ClubAce, FdoCard::SpadeQueen, FdoCard::SpadeTen, FdoCard::ClubKing],
                ),
                // 9. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartKing, FdoCard::HeartAce, FdoCard::ClubTen, FdoCard::ClubTen],
                ),
                // 10. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::SpadeTen, FdoCard::SpadeAce, FdoCard::ClubAce, FdoCard::ClubQueen],
                ),
                // 11. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::SpadeNine, FdoCard::SpadeNine, FdoCard::ClubNine, FdoCard::ClubNine],
                ),
                // 12. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::DiamondNine, FdoCard::DiamondNine, FdoCard::HeartNine, FdoCard::HeartNine],
                ),
            ]).unwrap(),
        );

        let expected = FdoEndOfGameStats {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM]),
            is_solo: true,
            player_eyes: PlayerZeroOrientedArr::from_full([136, 35, 0, 69]),
            re_eyes: 136,
            kontra_eyes: 104,
            re_points: 3,
            kontra_points: -1,
            player_points: PlayerZeroOrientedArr::from_full([3, -1, -1, -1]),
            basic_winning_point_details: None,
            basic_draw_points_details: Some(FdoBasicDrawPointsDetails {
                winning_under_90_re: 0,
                winning_under_60_re: 0,
                winning_under_30_re: 0,
                winning_under_90_kontra: 0,
                winning_under_60_kontra: 0,
                winning_under_30_kontra: 0,
                re_reached_120_against_no_90: 1,
                re_reached_90_against_no_60: 0,
                re_reached_60_against_no_30: 0,
                re_reached_30_against_black: 0,
                kontra_reached_120_against_no_90: 0,
                kontra_reached_90_against_no_60: 0,
                kontra_reached_60_against_no_30: 0,
                kontra_reached_30_against_black: 0,
            }),
            additional_points_details: None,
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_end_of_game_stats_normalspiel_kontra_verloren() {
        // 3. Testfall: Ein normales Spiel, bei dem Kontra verloren hat.

        let actual = FdoEndOfGameStats::calculate(
            PlayerZeroOrientedArr::from_full([0, 33, 115, 92]),
            PlayerZeroOrientedArr::from_full([0, 0, 0, 0]),
            FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::RIGHT]),
            Some(FdoAnnouncement::ReContra),
            Some(FdoAnnouncement::Black),
            &heapless::Vec::from_slice(&[
                // 1. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartAce, FdoCard::HeartNine, FdoCard::DiamondAce, FdoCard::HeartNine],
                ),
                // 2. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![FdoCard::ClubAce, FdoCard::ClubKing, FdoCard::ClubAce, FdoCard::ClubNine],
                ),
                // 3. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![FdoCard::DiamondJack, FdoCard::ClubQueen, FdoCard::HeartQueen, FdoCard::HeartTen],
                ),
                // 4. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::SpadeAce, FdoCard::SpadeNine, FdoCard::DiamondAce, FdoCard::SpadeAce],
                ),
                // 5. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![FdoCard::HeartAce, FdoCard::HeartKing, FdoCard::HeartKing, FdoCard::SpadeTen],
                ),
                // 6. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![
                        FdoCard::DiamondNine,
                        FdoCard::DiamondQueen,
                        FdoCard::HeartTen,
                        FdoCard::HeartJack,
                    ],
                ),
                // 7. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::SpadeKing, FdoCard::SpadeNine, FdoCard::ClubQueen, FdoCard::SpadeTen],
                ),
                // 8. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::ClubNine, FdoCard::ClubTen, FdoCard::ClubTen, FdoCard::DiamondTen],
                ),
                // 9. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::SpadeJack,
                        FdoCard::SpadeQueen,
                        FdoCard::DiamondKing,
                        FdoCard::DiamondKing,
                    ],
                ),
                // 10. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![
                        FdoCard::DiamondNine,
                        FdoCard::DiamondTen,
                        FdoCard::HeartQueen,
                        FdoCard::SpadeQueen,
                    ],
                ),
                // 11. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::DiamondQueen,
                        FdoCard::DiamondJack,
                        FdoCard::ClubKing,
                        FdoCard::SpadeJack,
                    ],
                ),
                // 12. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![FdoCard::ClubJack, FdoCard::HeartJack, FdoCard::SpadeKing, FdoCard::ClubJack],
                ),
            ]).unwrap(),
        );

        let expected = FdoEndOfGameStats {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::TOP, FdoPlayer::RIGHT]),
            is_solo: false,
            player_eyes: PlayerZeroOrientedArr::from_full([0, 33, 115, 92]),
            re_eyes: 207,
            kontra_eyes: 33,
            re_points: 16,
            kontra_points: -16,
            player_points: PlayerZeroOrientedArr::from_full([-16, -16, 16, 16]),
            basic_winning_point_details: Some(FdoBasicWinningPointsDetails {
                winning_points: 1,
                winning_under_90: 1,
                winning_under_60: 1,
                winning_under_30: 0,
                winning_black: 0,

                re_announcement: 2,
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
            }),

            basic_draw_points_details: None,
            additional_points_details: Some(FdoAdditionalPointsDetails {
                against_club_queens: false,
                number_of_doppelkopf_re: 0,
                number_of_doppelkopf_kontra: 0,
                fuchs_gefangen_re: 0,
                fuchs_gefangen_kontra: 0,
                karlchen_last_trick_re: true,
                karlchen_last_trick_kontra: false,
            }),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_end_of_game_stats_normalspiel_unentschieden() {
        // 4. Testfall: Ein normales Spiel endet in einem Unentschieden.
        let actual = FdoEndOfGameStats::calculate(
            PlayerZeroOrientedArr::from_full([48, 81, 79, 32]),
            PlayerZeroOrientedArr::from_full([0, 0, 0, 0]),
            FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::LEFT]),
            Some(FdoAnnouncement::No90),
            Some(FdoAnnouncement::Black),
            &heapless::Vec::from_slice(&[
                // 1. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubAce, FdoCard::ClubTen, FdoCard::ClubNine, FdoCard::ClubNine],
                ),
                // 2. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartKing, FdoCard::HeartAce, FdoCard::HeartAce, FdoCard::HeartNine],
                ),
                // 3. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::SpadeAce, FdoCard::SpadeKing, FdoCard::SpadeAce, FdoCard::SpadeTen],
                ),
                // 4. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![FdoCard::SpadeTen, FdoCard::HeartTen, FdoCard::ClubKing, FdoCard::SpadeNine],
                ),
                // 5. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![FdoCard::DiamondNine, FdoCard::HeartTen, FdoCard::HeartJack, FdoCard::DiamondJack],
                ),
                // 6. Stich
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![FdoCard::HeartKing, FdoCard::ClubJack, FdoCard::HeartNine, FdoCard::DiamondQueen],
                ),
                // 7. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::DiamondNine,
                        FdoCard::ClubJack,
                        FdoCard::DiamondQueen,
                        FdoCard::DiamondTen,
                    ],
                ),
                // 8. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::SpadeNine, FdoCard::SpadeKing, FdoCard::DiamondKing, FdoCard::ClubAce],
                ),
                // 9. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::DiamondAce,
                        FdoCard::SpadeQueen,
                        FdoCard::DiamondJack,
                        FdoCard::SpadeJack,
                    ],
                ),
                // 10. Stich
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::SpadeJack,
                        FdoCard::SpadeQueen,
                        FdoCard::DiamondKing,
                        FdoCard::ClubQueen,
                    ],
                ),
                // 11. Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::ClubTen,
                        FdoCard::ClubKing,
                        FdoCard::HeartJack,
                        FdoCard::DiamondAce,
                    ],
                ),
                // 12. Stich
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::HeartQueen,
                        FdoCard::DiamondTen,
                        FdoCard::HeartQueen,
                        FdoCard::ClubQueen,
                    ],
                ),
            ]).unwrap(),
        );

        let expected = FdoEndOfGameStats {
            re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::LEFT]),
            is_solo: false,
            player_eyes: PlayerZeroOrientedArr::from_full([48, 81, 79, 32]),
            re_eyes: 129,
            kontra_eyes: 111,
            re_points: 4,
            kontra_points: -4,
            player_points: PlayerZeroOrientedArr::from_full([4, 4, -4, -4]),
            basic_winning_point_details: None,
            basic_draw_points_details: Some(FdoBasicDrawPointsDetails {
                winning_under_90_re: 0,
                winning_under_60_re: 0,
                winning_under_30_re: 0,
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
            }),
            additional_points_details: Some(FdoAdditionalPointsDetails {
                against_club_queens: false,
                number_of_doppelkopf_re: 0,
                number_of_doppelkopf_kontra: 0,
                fuchs_gefangen_re: 0,
                fuchs_gefangen_kontra: 0,
                karlchen_last_trick_re: false,
                karlchen_last_trick_kontra: false,
            }),
        };

        assert_eq!(actual, expected);
    }



}

