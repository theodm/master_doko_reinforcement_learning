use crate::basic::team::FdoTeam;
use crate::card::cards::FdoCard::DiamondAce;
use crate::player::player_set::FdoPlayerSet;
use crate::trick::trick::FdoTrick;

/// Gibt die Anzahl der gefangenen Füchse der Teams im übergebenen Stich zurück.
/// Ein Fuchs ist ein Karo-Ass einer Partei, welches in einem Stich von der anderen Partei
/// gemacht wurde.
fn calc_fuchs_gefangen_in_trick(
    trick: &FdoTrick,

    re_players: FdoPlayerSet
) -> (i32, i32) {
    let mut fuchs_gefangen_re = 0;
    let mut fuchs_gefangen_kontra = 0;

    for (player, card) in trick.cards.iter_with_player() {
        let is_fuchs = *card == DiamondAce;

        let player_who_played_card = player;
        let player_who_played_card_team = player_who_played_card.team(re_players);

        let player_who_won_trick = trick.winning_player.unwrap();
        let player_who_won_trick_team = player_who_won_trick.team(re_players);

        if is_fuchs && player_who_played_card_team != player_who_won_trick_team {
            if player_who_won_trick_team == FdoTeam::Re {
                fuchs_gefangen_re = fuchs_gefangen_re + 1;
            } else {
                fuchs_gefangen_kontra = fuchs_gefangen_kontra + 1;
            }
        }
    }

    (fuchs_gefangen_re, fuchs_gefangen_kontra)
}

/// Gibt die Anzahl der gefangenen Füchse der Teams zurück. Ein Fuchs ist ein Karo-Ass eines
/// Partei, welches in einem Stich von der anderen Partei gemacht wurde.
pub fn calc_fuchs_gefangen(
    re_players: FdoPlayerSet,
    tricks: &heapless::Vec<FdoTrick, 12>
) -> (i32, i32) {
    debug_assert!(re_players.len() == 2, "Im Solospiel werden keine Sonderpunkte für gefangene Füchse vergeben.");

    let mut fuchs_gefangen_re = 0;
    let mut fuchs_gefangen_kontra = 0;

    for trick in tricks {
        let (fuchs_gefangen_re_trick, fuchs_gefangen_kontra_trick) = calc_fuchs_gefangen_in_trick(
            trick,
            re_players
        );

        fuchs_gefangen_re = fuchs_gefangen_re + fuchs_gefangen_re_trick;
        fuchs_gefangen_kontra = fuchs_gefangen_kontra + fuchs_gefangen_kontra_trick;
    }

    (fuchs_gefangen_re, fuchs_gefangen_kontra)
}

mod tests {
    use crate::card::cards::FdoCard::{DiamondAce, DiamondJack, DiamondKing, DiamondQueen, HeartAce, HeartKing, HeartNine};
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::stats::additional_points::fuchs_gefangen::{calc_fuchs_gefangen, calc_fuchs_gefangen_in_trick};
    use crate::trick::trick::FdoTrick;

    #[test]
    fn test_calc_fuchs_gefangen() {

        let dummy_trick = FdoTrick::existing(
            FdoPlayer::BOTTOM,
            vec![HeartAce, HeartAce, HeartKing, HeartKing]
        );

        let (fuchs_gefangen_re, fuchs_gefangen_kontra) = calc_fuchs_gefangen(
            FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP]),
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
                // Hier hat Re 1 Fuchs gefangen gefangen
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![DiamondKing, DiamondAce, DiamondQueen, DiamondKing]
                ),
                // Hier hat Kontra 1 Fuchs gefangen
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![DiamondAce, DiamondJack, DiamondKing, DiamondKing]
                ),
            ]).unwrap()
        );

        assert_eq!(fuchs_gefangen_re, 1);
        assert_eq!(fuchs_gefangen_kontra, 1);
    }

    #[test]
    fn test_calc_fuchs_gefangen_in_trick() {
        // Re-Team hat einen Fuchs gefangen
        assert_eq!(
            calc_fuchs_gefangen_in_trick(
                &FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![HeartAce, DiamondAce, DiamondJack, HeartKing]
                ),
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])
            ),
            (1, 0)
        );

        // Kontra-Partei hat einen Fuchs gefangen
        assert_eq!(
            calc_fuchs_gefangen_in_trick(
                &FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![HeartAce, DiamondAce, DiamondJack, HeartKing]
                ),
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])
            ),
            (0, 1)
        );

        // Re-Partei hat keinen Fuchs gefangen, da es das eigene Karo-Ass war
        assert_eq!(
            calc_fuchs_gefangen_in_trick(
                &FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![DiamondAce, DiamondJack, DiamondQueen, HeartKing]
                ),
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])
            ),
            (0, 0)
        );

        // Re-Partei hat beide Füchse auf einmal gefangen
        assert_eq!(
            calc_fuchs_gefangen_in_trick(
                &FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![DiamondKing, DiamondAce, DiamondQueen, DiamondAce]
                ),
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])
            ),
            (2, 0)
        );

        // Kein Fuchs in Sichtweite
        assert_eq!(
            calc_fuchs_gefangen_in_trick(
                &FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![HeartAce, HeartKing, HeartKing, HeartNine]
                ),
                FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::TOP])
            ),
            (0, 0)
        );

    }
}