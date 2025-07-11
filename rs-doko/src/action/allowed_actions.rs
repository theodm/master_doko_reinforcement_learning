use std::mem;
use std::mem::transmute;
use rand::prelude::{IteratorRandom, SmallRng};
use rand::Rng;
use strum::EnumCount;
use crate::action::action::DoAction;
use crate::basic::color::DoColor;
use crate::basic::phase::DoPhase;
use crate::card::card_color_masks::{CLUB_COLOR_MASK_NORMAL_GAME, HEART_COLOR_MASK_NORMAL_GAME, SPADE_COLOR_MASK_NORMAL_GAME, TRUMP_COLOR_MASK_NORMAL_GAME};
use crate::card::cards::DoCard;
use crate::hand::hand::{DoHand, hand_contains_both};
use crate::util::bitflag::bitflag::{bitflag_add, bitflag_contains, bitflag_number_of_ones, bitflag_random_single, bitflag_to_vec};

pub type DoAllowedActions = usize;

pub fn allowed_actions_len(
    allowed_actions: DoAllowedActions
) -> usize {
    // ToDo: Tests!
    bitflag_number_of_ones::<DoAllowedActions, { DoAction::COUNT }>(allowed_actions) as usize
}

fn allowed_actions_add(
    allowed_actions: DoAllowedActions,
    action: DoAction
) -> DoAllowedActions {
    bitflag_add::<DoAllowedActions, { DoAction::COUNT }>(
        allowed_actions,
        action as usize
    )
}

pub fn allowed_actions_to_vec(
    allowed_actions: DoAllowedActions
) -> heapless::Vec<DoAction, 26> {
    allowed_actions_to_vec_fast(allowed_actions)
}

pub fn allowed_actions_to_vec_fast(
    allowed_actions: DoAllowedActions
) -> heapless::Vec<DoAction, 26> {
    unsafe {
        transmute(bitflag_to_vec::<DoAllowedActions, { DoAction::COUNT }>(allowed_actions as usize))
    }
}

pub fn allowed_actions_to_vec_slow(
    allowed_actions: DoAllowedActions
) -> Vec<DoAction> {
    let mut actions: Vec<DoAction> = Vec::with_capacity(
        bitflag_number_of_ones::<DoAllowedActions, { DoAction::COUNT }>(allowed_actions) as usize
    );

    macro_rules! add_action {
        ($action:expr) => {
            if (allowed_actions & ($action as usize)) != 0 {
                actions.push($action);
            }
        };
        () => {};
    }

    add_action!(DoAction::CardDiamondNine);
    add_action!(DoAction::CardDiamondTen);
    add_action!(DoAction::CardDiamondJack);
    add_action!(DoAction::CardDiamondQueen);
    add_action!(DoAction::CardDiamondKing);
    add_action!(DoAction::CardDiamondAce);

    add_action!(DoAction::CardHeartNine);
    add_action!(DoAction::CardHeartTen);
    add_action!(DoAction::CardHeartJack);
    add_action!(DoAction::CardHeartQueen);
    add_action!(DoAction::CardHeartKing);
    add_action!(DoAction::CardHeartAce);

    add_action!(DoAction::CardClubNine);
    add_action!(DoAction::CardClubTen);
    add_action!(DoAction::CardClubJack);
    add_action!(DoAction::CardClubQueen);
    add_action!(DoAction::CardClubKing);
    add_action!(DoAction::CardClubAce);

    add_action!(DoAction::CardSpadeNine);
    add_action!(DoAction::CardSpadeTen);
    add_action!(DoAction::CardSpadeJack);
    add_action!(DoAction::CardSpadeQueen);
    add_action!(DoAction::CardSpadeKing);
    add_action!(DoAction::CardSpadeAce);

    add_action!(DoAction::ReservationHealthy);
    add_action!(DoAction::ReservationWedding);

    actions
}

pub fn random_action(
    allowed_actions: DoAllowedActions,
    rng: &mut SmallRng
) -> DoAction {
    unsafe {
        transmute(bitflag_random_single(allowed_actions as u64, rng) as usize)
    }
}

pub fn random_action_slow(
    allowed_actions: DoAllowedActions,
    rng: &mut SmallRng
) -> DoAction {
    let actions = allowed_actions_to_vec(allowed_actions);

    let index = rng.gen_range(0..actions.len());

    actions[index]
}

/// Nur für Testzwecke!
pub fn allowed_actions_from_vec(
    actions: Vec<DoAction>
) -> DoAllowedActions {
    let mut allowed_actions = 0;

    for action in actions {
        allowed_actions = allowed_actions_add(allowed_actions, action);
    }

    allowed_actions
}


pub fn calculate_allowed_actions_in_normal_game(
    phase: DoPhase,

    trick_color: Option<DoColor>,
    player_hand: DoHand
) -> DoAllowedActions {

    match phase {
        DoPhase::Reservation => {
            let mut allowed_actions = 0;

            allowed_actions = allowed_actions_add(allowed_actions, DoAction::ReservationHealthy);

            // Wenn er beide Kreuz-Damen hat, darf er eine Hochzeit ansagen.
            if hand_contains_both(player_hand, DoCard::ClubQueen) {
                allowed_actions = allowed_actions_add(allowed_actions, DoAction::ReservationWedding);
            }

            bitflag_contains::<DoAllowedActions, { DoAction::COUNT }>(allowed_actions, 1);

            return allowed_actions;
        }
        DoPhase::PlayCard => {
            // Er darf grundsätzlich alle Karten spielen, die er in der Hand hat.
            // ToDo: Bit_Magic erklären
            let player_hand_single = (player_hand | (player_hand >> 24)) as usize & 0b111111111111111111111111;

            match trick_color {
                // Er selbst spielt aus, darf also jede Karte legen.
                None => return player_hand_single,

                // Es gibt eine Farbe des Stiches, ggf. muss er bekennen.
                Some(color) => {
                    // Wenn er eine Karte der Stichfarbe hat, muss er diese spielen.
                    let mask = match color {
                        DoColor::Trump => {
                            TRUMP_COLOR_MASK_NORMAL_GAME
                        }
                        DoColor::Heart => {
                            HEART_COLOR_MASK_NORMAL_GAME
                        }
                        DoColor::Spade => {
                            SPADE_COLOR_MASK_NORMAL_GAME
                        }
                        DoColor::Club => {
                            CLUB_COLOR_MASK_NORMAL_GAME
                        }
                    };

                    if (player_hand_single & mask) != 0 {
                        // Er hat noch Karten der Farbe auf der Hand
                        return player_hand_single & mask;
                    }

                    // Er hat keine Karten der Farbe auf der Hand und
                    // darf damit alle Karten seiner Hand spielen.
                    return player_hand_single;
                }
            }


        }
        DoPhase::Finished => {
            // Am Ende darf man nichts mehr machen :-)
            return 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::action::allowed_actions::allowed_actions_to_vec;
    use crate::action::action::DoAction;
    use crate::action::allowed_actions::{allowed_actions_from_vec, calculate_allowed_actions_in_normal_game};
    use crate::basic::color::DoColor;
    use crate::basic::phase::DoPhase;
    use crate::card::cards::DoCard;
    use crate::hand::hand::hand_from_vec;

    #[test]
    pub fn test_allowed_actions() {
        // In der Vorbehaltsphase darf der Spieler nur "gesund" ansagen, wenn er nicht beide Kreuz-Damen
        // hat.
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::Reservation,
            None,
            hand_from_vec(vec![DoCard::ClubQueen])
        ), allowed_actions_from_vec(vec![DoAction::ReservationHealthy]));

        // In der Vorbehaltsphase darf der Spieler auch eine Hochzeit ansagen, wenn er beide Kreuz-Damen
        // hat.
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::Reservation,
            None,
            hand_from_vec(vec![DoCard::ClubQueen, DoCard::ClubQueen])
        ), allowed_actions_from_vec(vec![DoAction::ReservationHealthy, DoAction::ReservationWedding]));

        // Wenn das Spiel abgeschlossen ist, sind keine Aktionen mehr möglich.
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::Finished,
            None,
            hand_from_vec(vec![])
        ), 0);

        // Ist noch keine Stichfarbe des aktuellen Stiches bekannt, dann darf er jede Karte spielen.
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::PlayCard,
            None,
            hand_from_vec(vec![
                // Trumpf
                DoCard::ClubQueen,
                // Herz
                DoCard::HeartNine,
                // Pik
                DoCard::SpadeNine,
                // Kreuz
                DoCard::ClubNine
            ])
        ), allowed_actions_from_vec(vec![
            DoAction::CardClubQueen,
            DoAction::CardHeartNine,
            DoAction::CardSpadeNine,
            DoAction::CardClubNine
        ]));

        // Ist eine Stichfarbe bekannt, dann muss er eine Karte der Stichfarbe spielen, da er eine hat.
        // hier: Kreuz
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::PlayCard,
            Some(DoColor::Club),
            hand_from_vec(vec![
                // Trumpf
                DoCard::ClubQueen,
                // Herz
                DoCard::HeartNine,
                // Pik
                DoCard::SpadeNine,
                // Kreuz
                DoCard::ClubNine
            ])
        ), allowed_actions_from_vec(vec![
            DoAction::CardClubNine
        ]));

        // Ist eine Stichfarbe bekannt, dann muss er keine Karte der Stichfarbe spielen, da er keine hat.
        // hier: Kreuz
        assert_eq!(calculate_allowed_actions_in_normal_game(
            DoPhase::PlayCard,
            Some(DoColor::Club),
            hand_from_vec(vec![
                // Trumpf
                DoCard::ClubQueen,
                // Herz
                DoCard::HeartNine,
                // Pik
                DoCard::SpadeNine,
            ])
        ), allowed_actions_from_vec(vec![
            DoAction::CardClubQueen,
            DoAction::CardHeartNine,
            DoAction::CardSpadeNine
        ]));

        // Ein in der Praxis aufgetretener Fehlerfall.
        assert_eq!(
            allowed_actions_to_vec(calculate_allowed_actions_in_normal_game(
                DoPhase::PlayCard,
                Some(DoColor::Trump),
                hand_from_vec(vec![
                    DoCard::ClubAce,
                    DoCard::ClubNine,
                    DoCard::ClubQueen,
                    DoCard::DiamondJack,
                    DoCard::ClubKing
                ])
            )).to_vec(),
            vec![
                DoAction::CardDiamondJack,
                DoAction::CardClubQueen,
            ]
        );
    }


}