use crate::basic::team::FdoTeam;
use crate::player::player_set::FdoPlayerSet;
use crate::trick::trick::FdoTrick;

/// Gibt die Anzahl der Doppelköpfe zurück, welche die Re-Partei und
/// die Kontra-Partei gemacht haben. Ein Doppelkopf ist ein Stich mit 40 oder
/// mehr Augen.
pub fn calc_number_of_doppelkopf(
    re_players: FdoPlayerSet,
    tricks: &heapless::Vec<FdoTrick, 12>,
) -> (i32, i32) {
    let mut number_of_doppelkopf_re = 0;
    let mut number_of_doppelkopf_kontra = 0;

    for trick in tricks {
        if trick.eyes() >= 40 {
            if trick.winning_player.unwrap().team(re_players) == FdoTeam::Re {
                number_of_doppelkopf_re = number_of_doppelkopf_re + 1;
            } else {
                number_of_doppelkopf_kontra = number_of_doppelkopf_kontra + 1;
            }
        }
    }

    return (number_of_doppelkopf_re, number_of_doppelkopf_kontra);
}

#[cfg(test)]
mod tests {
    use crate::card::cards::FdoCard;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::stats::additional_points::doppelkopf::calc_number_of_doppelkopf;
    use crate::trick::trick::FdoTrick;

    #[test]
    fn test_calc_number_of_doppelkopf() {
        let dummy_trick = FdoTrick::existing(
            FdoPlayer::TOP,
            vec![
                FdoCard::HeartAce,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartKing,
            ],
        );

        let (number_of_doppelkopf_re, number_of_doppelkopf_kontra) = calc_number_of_doppelkopf(
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
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::HeartAce,
                        FdoCard::HeartAce,
                        FdoCard::SpadeTen,
                        FdoCard::SpadeTen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::ClubAce,
                        FdoCard::ClubAce,
                        FdoCard::SpadeTen,
                        FdoCard::SpadeTen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::SpadeTen,
                        FdoCard::SpadeAce,
                        FdoCard::SpadeTen,
                        FdoCard::ClubTen,
                    ],
                )
            ]).unwrap()
        );

        assert_eq!(number_of_doppelkopf_re, 2);
        assert_eq!(number_of_doppelkopf_kontra, 1);
    }
}

