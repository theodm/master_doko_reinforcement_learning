use std::fmt::{Debug};
use std::mem::transmute;

use rand::prelude::{SmallRng};
use strum::EnumCount;

use rs_game_utils::bit_flag::Bitflag;

use crate::action::action::FdoAction;
use crate::announcement::announcement::FdoAnnouncement;
use crate::announcement::announcement_set::FdoAnnouncementSet;
use crate::basic::color::FdoColor;
use crate::basic::phase::FdoPhase;
use crate::card::card_color_masks::get_color_masks_for_game_type;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use crate::hand::hand::FdoHand;

pub type DoAllowedActions = usize;

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct FdoAllowedActions(pub Bitflag<{ FdoAction::COUNT }>);

impl FdoAllowedActions {
    pub fn new() -> FdoAllowedActions {
        FdoAllowedActions(Bitflag::new())
    }

    pub fn add(&mut self, action: FdoAction) {
        self.0.add(action as u64);
    }

    pub fn contains(&self, action: FdoAction) -> bool {
        self.0.contains(action as u64)
    }

    pub fn remove(&mut self, action: FdoAction) {
        self.0.remove(action as u64);
    }

    pub fn len(&self) -> usize {
        self.0.number_of_ones() as usize
    }

    pub fn to_vec(&self) -> heapless::Vec<FdoAction, {FdoAction::COUNT}> {
        unsafe {
            transmute(self.0.to_vec())
        }
    }

    pub fn random(&self, rng: &mut SmallRng) -> FdoAction {
        unsafe {
            transmute(self.0.random_single(rng) as usize)
        }
    }

    /// Nur für Testzwecke!
    pub fn from_vec(actions: Vec<FdoAction>) -> FdoAllowedActions {
        let mut allowed_actions = FdoAllowedActions::new();

        for action in actions {
            allowed_actions.add(action);
        }

        allowed_actions
    }

    pub(crate) fn calculate_allowed_actions(
        phase: FdoPhase,
        trick_color: Option<FdoColor>,
        player_hand: FdoHand,
        game_type: Option<FdoGameType>,
        allowed_announcements: FdoAnnouncementSet,
    ) -> FdoAllowedActions {
        match phase {
            FdoPhase::Reservation => {
                let mut allowed_actions = FdoAllowedActions::new();

                allowed_actions.add(FdoAction::ReservationHealthy);

                // Wenn er beide Kreuz-Damen hat, darf er eine Hochzeit ansagen.
                if player_hand.contains_both(FdoCard::ClubQueen) {
                    allowed_actions.add(FdoAction::ReservationWedding);
                }

                // Solis kann er immer ansagen
                allowed_actions.add(FdoAction::ReservationJacksSolo);
                allowed_actions.add(FdoAction::ReservationQueensSolo);
                allowed_actions.add(FdoAction::ReservationDiamondsSolo);
                allowed_actions.add(FdoAction::ReservationHeartsSolo);
                allowed_actions.add(FdoAction::ReservationSpadesSolo);
                allowed_actions.add(FdoAction::ReservationClubsSolo);
                allowed_actions.add(FdoAction::ReservationTrumplessSolo);

                return allowed_actions;
            }
            FdoPhase::PlayCard => {
                let player_hand_bits = player_hand.0.0;

                // Er darf grundsätzlich alle Karten spielen, die er in der Hand hat.
                // ToDo: Bit_Magic erklären
                let player_hand_single = (player_hand_bits | (player_hand_bits >> 24)) & 0b111111111111111111111111;

                match trick_color {
                    // Er selbst spielt aus, darf also jede Karte legen.
                    None => return FdoAllowedActions(Bitflag(player_hand_single)),

                    // Es gibt eine Farbe des Stiches, ggf. muss er bekennen.
                    Some(color) => {
                        let masks_for_game_type = get_color_masks_for_game_type(game_type.unwrap());

                        // Wenn er eine Karte der Stichfarbe hat, muss er diese spielen.
                        let mask = match color {
                            FdoColor::Trump => {
                                masks_for_game_type.0
                            }
                            FdoColor::Diamond => {
                                masks_for_game_type.1
                            }
                            FdoColor::Heart => {
                                masks_for_game_type.2
                            }
                            FdoColor::Spade => {
                                masks_for_game_type.3
                            }
                            FdoColor::Club => {
                                masks_for_game_type.4
                            }
                        };

                        if (player_hand_single & mask) != 0 {
                            // Er hat noch Karten der Farbe auf der Hand
                            return FdoAllowedActions(Bitflag(player_hand_single & mask));
                        }

                        // Er hat keine Karten der Farbe auf der Hand und
                        // darf damit alle Karten seiner Hand spielen.
                        return FdoAllowedActions(Bitflag(player_hand_single));
                    }
                }
            }
            FdoPhase::Finished => {
                panic!("Es sind keine Aktionen mehr möglich, da das Spiel beendet ist.");
            }
            FdoPhase::Announcement => {
                let mut allowed_actions = FdoAllowedActions::new();

                if allowed_announcements.contains(FdoAnnouncement::CounterReContra) || allowed_announcements.contains(FdoAnnouncement::ReContra) {
                    allowed_actions.add(FdoAction::AnnouncementReContra);
                }
                if allowed_announcements.contains(FdoAnnouncement::No90) {
                    allowed_actions.add(FdoAction::AnnouncementNo90);
                }
                if allowed_announcements.contains(FdoAnnouncement::No60) {
                    allowed_actions.add(FdoAction::AnnouncementNo60);
                }
                if allowed_announcements.contains(FdoAnnouncement::No30) {
                    allowed_actions.add(FdoAction::AnnouncementNo30);
                }
                if allowed_announcements.contains(FdoAnnouncement::Black) {
                    allowed_actions.add(FdoAction::AnnouncementBlack);
                }

                allowed_actions.add(FdoAction::NoAnnouncement);

                return allowed_actions;
            }
        }
    }
}

impl Debug for FdoAllowedActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let actions = self.to_vec();

        write!(f, "FdoAllowedActions::from_vec(vec!{:?})", actions)
    }
}

//
// #[cfg(test)]
// mod tests {
//     use crate::action::action::FdoAction;
//     use crate::action::allowed_actions::FdoAllowedActions;
//     use crate::basic::phase::FdoPhase;
//     use crate::card::cards::FdoCard;
//     use crate::hand::hand::FdoHand;
//
//     #[test]
//     pub fn test_allowed_actions() {
//         // In der Vorbehaltsphase darf der Spieler keine Hochzeit ansagen, wenn er nicht beide Kreuz-Damen
//         // hat.
//         assert_eq!(FdoAllowedActions::calculate_allowed_actions(
//             FdoPhase::Reservation,
//             None,
//             FdoHand::from_vec(vec![FdoCard::ClubQueen]),
//             None,
//             vec![]
//         ), FdoAllowedActions::from_vec(vec![
//             FdoAction::ReservationHealthy
//         ]));
//
//         // In der Vorbehaltsphase darf der Spieler auch eine Hochzeit ansagen, wenn er beide Kreuz-Damen
//         // hat.
//         assert_eq!(calculate_allowed_actions_in_normal_game(
//             DoPhase::Reservation,
//             None,
//             hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen]),
//         ), allowed_actions_from_vec(vec![DoAction::ReservationHealthy, DoAction::ReservationWedding]));
//
//         // Wenn das Spiel abgeschlossen ist, sind keine Aktionen mehr möglich.
//         assert_eq!(calculate_allowed_actions_in_normal_game(
//             DoPhase::Finished,
//             None,
//             hand_from_vec(vec![]),
//         ), 0);
//
//         // Ist noch keine Stichfarbe des aktuellen Stiches bekannt, dann darf er jede Karte spielen.
//         assert_eq!(calculate_allowed_actions_in_normal_game(
//             DoPhase::PlayCard,
//             None,
//             hand_from_vec(vec![
//                 // Trumpf
//                 DoCard::ClubQueen,
//                 // Herz
//                 DoCard::HeartNine,
//                 // Pik
//                 DoCard::SpadeNine,
//                 // Kreuz
//                 DoCard::ClubNine,
//             ]),
//         ), allowed_actions_from_vec(vec![
//             DoAction::CardClubQueen,
//             DoAction::CardHeartNine,
//             DoAction::CardSpadeNine,
//             DoAction::CardClubNine,
//         ]));
//
//         // Ist eine Stichfarbe bekannt, dann muss er eine Karte der Stichfarbe spielen, da er eine hat.
//         // hier: Kreuz
//         assert_eq!(calculate_allowed_actions_in_normal_game(
//             DoPhase::PlayCard,
//             Some(DoColor::Club),
//             hand_from_vec(vec![
//                 // Trumpf
//                 DoCard::ClubQueen,
//                 // Herz
//                 DoCard::HeartNine,
//                 // Pik
//                 DoCard::SpadeNine,
//                 // Kreuz
//                 DoCard::ClubNine,
//             ]),
//         ), allowed_actions_from_vec(vec![
//             DoAction::CardClubNine
//         ]));
//
//         // Ist eine Stichfarbe bekannt, dann muss er keine Karte der Stichfarbe spielen, da er keine hat.
//         // hier: Kreuz
//         assert_eq!(calculate_allowed_actions_in_normal_game(
//             DoPhase::PlayCard,
//             Some(DoColor::Club),
//             hand_from_vec(vec![
//                 // Trumpf
//                 DoCard::ClubQueen,
//                 // Herz
//                 DoCard::HeartNine,
//                 // Pik
//                 DoCard::SpadeNine,
//             ]),
//         ), allowed_actions_from_vec(vec![
//             DoAction::CardClubQueen,
//             DoAction::CardHeartNine,
//             DoAction::CardSpadeNine,
//         ]));
//
//         // Ein in der Praxis aufgetretener Fehlerfall.
//         assert_eq!(
//             allowed_actions_to_vec(calculate_allowed_actions_in_normal_game(
//                 DoPhase::PlayCard,
//                 Some(DoColor::Trump),
//                 hand_from_vec(vec![
//                     DoCard::ClubAce,
//                     DoCard::ClubNine,
//                     DoCard::ClubQueen,
//                     DoCard::DiamondJack,
//                     DoCard::ClubKing,
//                 ]),
//             )).to_vec(),
//             vec![
//                 DoAction::CardDiamondJack,
//                 DoAction::CardClubQueen,
//             ]
//         );
//     }
// }