use rand::prelude::{IndexedRandom, SmallRng};
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::observation::observation::FdoObservation;

pub fn get_weighted_action(obs: &FdoObservation, rng: &mut SmallRng) -> FdoAction {
    *obs
        .allowed_actions_current_player
        .to_vec()
        .choose_weighted(rng, |action| {
            match action {
                FdoAction::CardDiamondNine => 20,
                FdoAction::CardDiamondTen => 20,
                FdoAction::CardDiamondJack => 20,
                FdoAction::CardDiamondQueen => 20,
                FdoAction::CardDiamondKing => 20,
                FdoAction::CardDiamondAce => 20,
                FdoAction::CardHeartNine => 20,
                FdoAction::CardHeartTen => 20,
                FdoAction::CardHeartJack => 20,
                FdoAction::CardHeartQueen => 20,
                FdoAction::CardHeartKing => 20,
                FdoAction::CardHeartAce => 20,
                FdoAction::CardClubNine => 20,
                FdoAction::CardClubTen => 20,
                FdoAction::CardClubJack => 20,
                FdoAction::CardClubQueen => 20,
                FdoAction::CardClubKing => 20,
                FdoAction::CardClubAce => 20,
                FdoAction::CardSpadeNine => 20,
                FdoAction::CardSpadeTen => 20,
                FdoAction::CardSpadeJack => 20,
                FdoAction::CardSpadeQueen => 20,
                FdoAction::CardSpadeKing => 20,
                FdoAction::CardSpadeAce => 20,

                FdoAction::ReservationHealthy => 20,
                FdoAction::ReservationWedding => 100,
                FdoAction::ReservationDiamondsSolo => 2,
                FdoAction::ReservationHeartsSolo => 2,
                FdoAction::ReservationSpadesSolo => 2,
                FdoAction::ReservationClubsSolo => 2,
                FdoAction::ReservationTrumplessSolo => 2,
                FdoAction::ReservationQueensSolo => 2,
                FdoAction::ReservationJacksSolo => 2,

                FdoAction::AnnouncementReContra => 1,
                FdoAction::AnnouncementNo90 => 1,
                FdoAction::AnnouncementNo60 => 1,
                FdoAction::AnnouncementNo30 => 1,
                FdoAction::AnnouncementBlack => 1,
                FdoAction::NoAnnouncement => 20,
            }
        })
        .unwrap()
}