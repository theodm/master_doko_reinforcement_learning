use serde::{Deserialize, Serialize};
use crate::basic::team::FdoTeam;
use crate::player::player_set::FdoPlayerSet;
use crate::stats::additional_points::doppelkopf::calc_number_of_doppelkopf;
use crate::stats::additional_points::fuchs_gefangen::calc_fuchs_gefangen;
use crate::stats::additional_points::last_trick_karlchen::calc_trick_karlchen;
use crate::trick::trick::FdoTrick;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoAdditionalPointsDetails {
    pub against_club_queens: bool,

    pub number_of_doppelkopf_re: i32,
    pub number_of_doppelkopf_kontra: i32,

    pub fuchs_gefangen_re: i32,
    pub fuchs_gefangen_kontra: i32,

    pub karlchen_last_trick_re: bool,
    pub karlchen_last_trick_kontra: bool,
}

impl FdoAdditionalPointsDetails {

    pub fn calculate(
        re_players: FdoPlayerSet,
        winning_team: Option<FdoTeam>,
        tricks: &heapless::Vec<FdoTrick, 12>,
    ) -> (i32, i32, FdoAdditionalPointsDetails) {
        let (doko_re, doko_kontra) = calc_number_of_doppelkopf(
            re_players,
            tricks,
        );

        let (fuchs_re, fuchs_kontra) = calc_fuchs_gefangen(
            re_players,
            tricks
        );

        let (karlchen_re, karlchen_kontra) = calc_trick_karlchen(
            re_players,
            tricks
        );

        FdoAdditionalPointsDetails::internal_calculate(
            re_players,
            winning_team,
            doko_re,
            doko_kontra,
            fuchs_re,
            fuchs_kontra,
            karlchen_re > 0,
            karlchen_kontra > 0
        )
    }

    fn internal_calculate(
        re_players: FdoPlayerSet,

        winning_team: Option<FdoTeam>,

        number_of_doppelkopf_re: i32,
        number_of_doppelkopf_kontra: i32,

        fuchs_gefangen_re: i32,
        fuchs_gefangen_kontra: i32,

        karlchen_last_trick_re: bool,
        karlchen_last_trick_kontra: bool
    ) -> (i32, i32, FdoAdditionalPointsDetails) {
        debug_assert!(re_players.len() > 1, "Ein Solo hat keine Sonderpunkte.");

        let mut against_club_queens = false;

        let mut kontra_additional_points = 0;
        let mut re_additional_points = 0;

        // Nach 7.2.3 DKV-TR werden die folgenden Sonderpunkte vergeben:
        // Gegen die Kreuz-Damen gewonnen
        if winning_team == Some(FdoTeam::Kontra) {
            kontra_additional_points += 1;
            re_additional_points -= 1;

            against_club_queens = true;
        }

        // Doppelkopf (ein Stich mit 40 oder mehr Augen)
        re_additional_points += number_of_doppelkopf_re;
        kontra_additional_points -= number_of_doppelkopf_re;

        re_additional_points -= number_of_doppelkopf_kontra;
        kontra_additional_points += number_of_doppelkopf_kontra;

        // Fuchs gefangen
        re_additional_points += fuchs_gefangen_re;
        kontra_additional_points -= fuchs_gefangen_re;

        re_additional_points -= fuchs_gefangen_kontra;
        kontra_additional_points += fuchs_gefangen_kontra;

        // Karlchen im letzten Stich
        if karlchen_last_trick_re {
            re_additional_points += 1;
            kontra_additional_points -= 1;
        }

        if karlchen_last_trick_kontra {
            re_additional_points -= 1;
            kontra_additional_points += 1;
        }

        (
            re_additional_points,
            kontra_additional_points,
            FdoAdditionalPointsDetails {
                against_club_queens,
                number_of_doppelkopf_re,
                number_of_doppelkopf_kontra,
                fuchs_gefangen_re,
                fuchs_gefangen_kontra,
                karlchen_last_trick_re,
                karlchen_last_trick_kontra,
            }
        )


    }
}

#[cfg(test)]
mod tests {
    use crate::basic::team::FdoTeam;
    use crate::card::cards::FdoCard;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::stats::additional_points::additional_points::FdoAdditionalPointsDetails;
    use crate::trick::trick::FdoTrick;

    #[test]
    fn test_calculate() {
        let dummy_trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartAce,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartKing,
            ],
        );

        // Kontra gegen die Damen
        // Kontra hat einen Fuchs gefangen
        // Kontra hat 2 Doppelköpfe, Re hat 1 Doppelkopf
        // Kontra hat Karlchen im letzten Stich

        let (re_points, kontra_points, details) = FdoAdditionalPointsDetails::calculate(
            FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
            Some(FdoTeam::Kontra),
            &heapless::Vec::from_slice(&[
                dummy_trick.clone(),
                dummy_trick.clone(),
                dummy_trick.clone(),
                dummy_trick.clone(),
                dummy_trick.clone(),
                dummy_trick.clone(),
                dummy_trick.clone(),
                // Kontra hat einen Fuchs gefangen
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::DiamondAce, FdoCard::HeartTen, FdoCard::ClubNine, FdoCard::ClubNine]
                ),
                // Kontra hat einen Doppelkopf
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubAce, FdoCard::HeartTen, FdoCard::ClubAce, FdoCard::ClubTen]
                ),
                // Kontra hat noch einen Doppelkopf
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubAce, FdoCard::HeartTen, FdoCard::ClubAce, FdoCard::ClubTen]
                ),
                // Re hat einen Doppelkopf
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::HeartTen, FdoCard::ClubAce, FdoCard::ClubAce, FdoCard::ClubTen]
                ),
                // Kontra hat Karlchen im letzten Stich
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![FdoCard::ClubAce, FdoCard::ClubJack, FdoCard::ClubNine, FdoCard::ClubNine]
                ),
            ]).unwrap()
        );

        assert_eq!(re_points, -4);
        assert_eq!(kontra_points, 4);
        assert_eq!(details, FdoAdditionalPointsDetails {
            against_club_queens: true,
            number_of_doppelkopf_re: 1,
            number_of_doppelkopf_kontra: 2,
            fuchs_gefangen_re: 0,
            fuchs_gefangen_kontra: 1,
            karlchen_last_trick_re: false,
            karlchen_last_trick_kontra: true,
        });
    }

    #[test]
    fn test_internal_calculate() {
        // Gegen die Damen
        assert_eq!(
            FdoAdditionalPointsDetails::internal_calculate(
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
                Some(FdoTeam::Kontra),
                0,
                0,
                0,
                0,
                false,
                false,
            ),
            (
                -1,
                1,
                FdoAdditionalPointsDetails {
                    against_club_queens: true,
                    number_of_doppelkopf_re: 0,
                    number_of_doppelkopf_kontra: 0,
                    fuchs_gefangen_re: 0,
                    fuchs_gefangen_kontra: 0,
                    karlchen_last_trick_re: false,
                    karlchen_last_trick_kontra: false,
                },
            )
        );

        // Mit den Damen
        assert_eq!(
            FdoAdditionalPointsDetails::internal_calculate(
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
                Some(FdoTeam::Re),
                0,
                0,
                0,
                0,
                false,
                false,
            ),
            (
                0,
                0,
                FdoAdditionalPointsDetails {
                    against_club_queens: false,
                    number_of_doppelkopf_re: 0,
                    number_of_doppelkopf_kontra: 0,
                    fuchs_gefangen_re: 0,
                    fuchs_gefangen_kontra: 0,
                    karlchen_last_trick_re: false,
                    karlchen_last_trick_kontra: false,
                },
            )
        );

        // Doppelköpfe und Füchse
        assert_eq!(
            FdoAdditionalPointsDetails::internal_calculate(
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
                Some(FdoTeam::Re),
                2,
                1,
                1,
                1,
                false,
                false,
            ),
            (
                1,
                -1,
                FdoAdditionalPointsDetails {
                    against_club_queens: false,
                    number_of_doppelkopf_re: 2,
                    number_of_doppelkopf_kontra: 1,
                    fuchs_gefangen_re: 1,
                    fuchs_gefangen_kontra: 1,
                    karlchen_last_trick_re: false,
                    karlchen_last_trick_kontra: false,
                },
            )
        );

        // Karlchen durch Re
        assert_eq!(
            FdoAdditionalPointsDetails::internal_calculate(
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
                Some(FdoTeam::Re),
                0,
                0,
                0,
                0,
                true,
                false,
            ),
            (
                1,
                -1,
                FdoAdditionalPointsDetails {
                    against_club_queens: false,
                    number_of_doppelkopf_re: 0,
                    number_of_doppelkopf_kontra: 0,
                    fuchs_gefangen_re: 0,
                    fuchs_gefangen_kontra: 0,
                    karlchen_last_trick_re: true,
                    karlchen_last_trick_kontra: false,
                },
            )
        );

        // Karlchen durch Kontra
        assert_eq!(
            FdoAdditionalPointsDetails::internal_calculate(
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
                Some(FdoTeam::Kontra),
                0,
                0,
                0,
                0,
                false,
                true,
            ),
            (
                -2,
                2,
                FdoAdditionalPointsDetails {
                    against_club_queens: true,
                    number_of_doppelkopf_re: 0,
                    number_of_doppelkopf_kontra: 0,
                    fuchs_gefangen_re: 0,
                    fuchs_gefangen_kontra: 0,
                    karlchen_last_trick_re: false,
                    karlchen_last_trick_kontra: true,
                },
            )
        );
    }
}