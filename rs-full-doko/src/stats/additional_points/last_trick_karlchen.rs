use crate::basic::team::FdoTeam;
use crate::card::cards::FdoCard;
use crate::player::player_set::FdoPlayerSet;
use crate::trick::trick::FdoTrick;

pub fn calc_trick_karlchen(
    re_players: FdoPlayerSet,
    tricks: &heapless::Vec<FdoTrick, 12>,
) -> (i32, i32) {
    let mut karlchen_re = 0;
    let mut karlchen_kontra = 0;

    let last_trick = &tricks[11];

    if last_trick.winning_card == Some(FdoCard::ClubJack) {
        if last_trick
            .winning_player
            .unwrap()
            .team(re_players) == FdoTeam::Re {
            karlchen_re = 1;
        } else {
            karlchen_kontra = 1;
        }
    }

    (karlchen_re, karlchen_kontra)
}

#[cfg(test)]
mod tests {
    use crate::card::cards::FdoCard;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::stats::additional_points::last_trick_karlchen::calc_trick_karlchen;
    use crate::trick::trick::FdoTrick;

    #[test]
    fn test_calc_trick_karlchen() {
        let dummy_trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartAce,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartKing,
            ],
        );

        // Re-Partei hat ein Karlchen gemacht.
        assert_eq!(
            calc_trick_karlchen(
                FdoPlayerSet::from_vec(
                    vec![
                        FdoPlayer::TOP,
                        FdoPlayer::BOTTOM,
                    ],
                ),
                &heapless::Vec::from_slice(&[
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    FdoTrick::existing(
                        FdoPlayer::BOTTOM,
                        vec![
                            FdoCard::ClubJack,
                            FdoCard::DiamondAce,
                            FdoCard::ClubJack,
                            FdoCard::HeartNine,
                        ],
                    )
                ]).unwrap(),
            ),
            (1, 0),
        );

        // Kontra-Partei hat ein Karlchen gemacht.
        assert_eq!(
            calc_trick_karlchen(
                FdoPlayerSet::from_vec(
                    vec![
                        FdoPlayer::TOP,
                        FdoPlayer::BOTTOM,
                    ],
                ),
                &heapless::Vec::from_slice(&[
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    FdoTrick::existing(
                        FdoPlayer::BOTTOM,
                        vec![
                            FdoCard::HeartNine,
                            FdoCard::ClubJack,
                            FdoCard::HeartJack,
                            FdoCard::HeartNine,
                        ],
                    )
                ]).unwrap(),
            ),
            (0, 1)
        );

        // Ein Kreuz-Bube wurde übertrumpft, daher kein Karlchen.
        assert_eq!(
            calc_trick_karlchen(
                FdoPlayerSet::from_vec(
                    vec![
                        FdoPlayer::TOP,
                        FdoPlayer::BOTTOM,
                    ],
                ),
                &heapless::Vec::from_slice(&[
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    FdoTrick::existing(
                        FdoPlayer::BOTTOM,
                        vec![
                            FdoCard::ClubJack,
                            FdoCard::DiamondQueen,
                            FdoCard::ClubJack,
                            FdoCard::HeartNine,
                        ],
                    )
                ]).unwrap(),
            ),
            (0, 0)
        );

        // Kein Kreuz-Bube in Sicht
        assert_eq!(
            calc_trick_karlchen(
                FdoPlayerSet::from_vec(
                    vec![
                        FdoPlayer::TOP,
                        FdoPlayer::BOTTOM,
                    ],
                ),
                &heapless::Vec::from_slice(&[
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    dummy_trick.clone(),
                    FdoTrick::existing(
                        FdoPlayer::BOTTOM,
                        vec![
                            FdoCard::DiamondJack,
                            FdoCard::DiamondQueen,
                            FdoCard::HeartJack,
                            FdoCard::HeartJack,
                        ],
                    )
                ]).unwrap(),
            ),
            (0, 0)
        );
    }
}