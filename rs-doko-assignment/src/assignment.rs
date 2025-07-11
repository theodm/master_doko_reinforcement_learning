use rand::prelude::{IteratorRandom, SmallRng};
use rs_doko::basic::color::DoColor;
use rs_doko::card::card_to_color::card_to_color_in_normal_game;
use rs_doko::card::cards::DoCard;
use rs_doko::hand::hand::{DoHand, hand_from_vec, hand_len, hand_remove, hand_to_vec};
use rs_doko::player::player::{DoPlayer, player_increase};
use rs_doko::trick::trick::DoTrick;
use rs_doko::trick::trick_color_logic::trick_color_in_normal_game;
use itertools::Itertools;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;

fn calc_remaining_cards(
    tricks: &[Option<DoTrick>; 12],
    observing_player_hand: DoHand,
) -> DoHand {
    let mut available_cards = 0b111111111111111111111111_111111111111111111111111;

    for i in 0..12 {
        let trick = match tricks[i] {
            None => break,
            Some(trick) => trick
        };

        for j in 0..4 {
            match trick.cards[j] {
                None => break,
                Some(card) => {
                    available_cards = hand_remove(available_cards, card)
                }
            }
        }
    }

    for card in hand_to_vec(observing_player_hand) {
        available_cards = hand_remove(available_cards, card);
    }

    return available_cards;
}

fn available_cards_remove_all_of(
    available_cards: &mut Vec<DoCard>,
    cards_to_remove: &Vec<DoCard>,
) {
    for card in cards_to_remove {
        for i in 0..2 {
            let card_position = available_cards
                .iter()
                .position(|&x| x == *card);

            match card_position {
                None => {}
                Some(index) => {
                    available_cards.remove(index);
                }
            }
        }
    }
}

fn available_cards_remove_one_of(
    available_cards: &mut Vec<DoCard>,
    card_to_remove: DoCard,
) {
    let card_position = available_cards
        .iter()
        .position(|&x| x == card_to_remove);

    match card_position {
        None => {}
        Some(index) => {
            available_cards.remove(index);
        }
    }
}

fn player_allowed_to_have(
    player_marriage: Option<DoPlayer>,
    tricks: &[Option<DoTrick>; 12],

    observing_player_hand: DoHand,
    observing_player: DoPlayer,
) -> [Vec<DoCard>; 4] {
    // Folgende Dinge können wir immer ableiten:
    //
    // Der Spieler hat Trumpf nicht bekannt. -> Er hat (von den verbleibenden Karten) keinen Trumpf.
    // Der Spieler hat Herz nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Herz.
    // Der Spieler hat Pik nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Pik.
    // Der Spieler hat Kreuz nicht bekannt. -> Er hat (von den verbleibenden Karten) keine Kreuz.
    // Der Spieler hat eine Hochzeit angesagt. -> Er hat beide Kreuz-Damen, alle anderen haben keine Kreuz-Dame.

    let trump_cards = vec![
        DoCard::DiamondNine,
        DoCard::DiamondTen,
        DoCard::DiamondJack,
        DoCard::DiamondQueen,
        DoCard::DiamondKing,
        DoCard::DiamondAce,
        DoCard::DiamondJack,
        DoCard::HeartJack,
        DoCard::SpadeJack,
        DoCard::ClubJack,
        DoCard::DiamondQueen,
        DoCard::HeartQueen,
        DoCard::SpadeQueen,
        DoCard::ClubQueen,
        DoCard::HeartTen,
    ];

    let heart_cards = vec![
        DoCard::HeartNine,
        DoCard::HeartKing,
        DoCard::HeartAce,
    ];

    let spade_cards = vec![
        DoCard::SpadeNine,
        DoCard::SpadeTen,
        DoCard::SpadeKing,
        DoCard::SpadeAce,
    ];

    let club_cards = vec![
        DoCard::ClubNine,
        DoCard::ClubTen,
        DoCard::ClubKing,
        DoCard::ClubAce,
    ];

    let all_cards_two_times = vec![
        DoCard::DiamondNine,
        DoCard::DiamondNine,
        DoCard::DiamondTen,
        DoCard::DiamondTen,
        DoCard::DiamondJack,
        DoCard::DiamondJack,
        DoCard::DiamondQueen,
        DoCard::DiamondQueen,
        DoCard::DiamondKing,
        DoCard::DiamondKing,
        DoCard::DiamondAce,
        DoCard::DiamondAce,

        DoCard::HeartNine,
        DoCard::HeartNine,
        DoCard::HeartTen,
        DoCard::HeartTen,
        DoCard::HeartJack,
        DoCard::HeartJack,
        DoCard::HeartQueen,
        DoCard::HeartQueen,
        DoCard::HeartKing,
        DoCard::HeartKing,
        DoCard::HeartAce,
        DoCard::HeartAce,

        DoCard::SpadeNine,
        DoCard::SpadeNine,
        DoCard::SpadeTen,
        DoCard::SpadeTen,
        DoCard::SpadeJack,
        DoCard::SpadeJack,
        DoCard::SpadeQueen,
        DoCard::SpadeQueen,
        DoCard::SpadeKing,
        DoCard::SpadeKing,
        DoCard::SpadeAce,
        DoCard::SpadeAce,

        DoCard::ClubNine,
        DoCard::ClubNine,
        DoCard::ClubTen,
        DoCard::ClubTen,
        DoCard::ClubJack,
        DoCard::ClubJack,
        DoCard::ClubQueen,
        DoCard::ClubQueen,
        DoCard::ClubKing,
        DoCard::ClubKing,
        DoCard::ClubAce,
        DoCard::ClubAce,
    ];


    let mut player_possible_cards = [
        all_cards_two_times.clone(),
        all_cards_two_times.clone(),
        all_cards_two_times.clone(),
        all_cards_two_times.clone()
    ];

    player_possible_cards[observing_player] = hand_to_vec(observing_player_hand);

    // Alle vorherigen Stiche nach bereits gespielten
    // Karten und nicht-Bekennen durchsuchen
    for j in 0..12 {
        let current_trick = match tricks[j] {
            None => break,
            Some(trick) => trick
        };

        let current_trick_color = match trick_color_in_normal_game(&current_trick) {
            None => break,
            Some(color) => color
        };

        let starting_player = current_trick.start_player;

        let mut current_player = starting_player;
        for k in 0..4 {
            let current_card = match current_trick.cards[k] {
                None => break,
                Some(card) => card
            };

            // Bereits gespielte Karte können bei allen nicht mehr in der Hand sein.
            for i in 0..4 {
                available_cards_remove_one_of(
                    &mut player_possible_cards[i],
                    current_card
                );
            }

            let current_player_did_not_play_color
                = card_to_color_in_normal_game(current_card) != current_trick_color;

            if current_player_did_not_play_color {
                let cards_to_remove = match current_trick_color {
                    DoColor::Trump => &trump_cards,
                    DoColor::Heart => &heart_cards,
                    DoColor::Spade => &spade_cards,
                    DoColor::Club => &club_cards
                };

                available_cards_remove_all_of(
                    &mut player_possible_cards[current_player],
                    cards_to_remove,
                );
            }

            current_player = player_increase(current_player);
        }
    }

    // Hochzeitsregeln
    match player_marriage {
        None => {}
        Some(wedding_player) => {
            let wedding_player_index = wedding_player as usize;

            for i in 0..4 {
                if i == wedding_player_index {
                    continue;
                }

                available_cards_remove_all_of(
                    &mut player_possible_cards[i],
                    &vec![DoCard::ClubQueen]
                );
            }
        }
    }

    // Alle Karten, die wir selbst in der Hand haben können nicht vorkommen.
    for i in 0..4 {
        for card in hand_to_vec(observing_player_hand) {
            available_cards_remove_one_of(
                &mut player_possible_cards[i],
                card
            );
        }
    }

    return player_possible_cards;
}

fn distribute_card(
    player: DoPlayer,
    current_card: DoCard,

    player_hands: &mut [Vec<DoCard>; 4],
    player_hands_length: &mut [usize; 4],
    player_allowed_to_have: &mut [Vec<DoCard>; 4],
    remaining_cards_as_vec: &mut Vec<DoCard>,
) {
    player_hands[player].push(current_card);
    player_hands_length[player] -= 1;

    for i in 0..4 {
        available_cards_remove_one_of(
            &mut player_allowed_to_have[i],
            current_card
        );
    }

    available_cards_remove_one_of(
        remaining_cards_as_vec,
        current_card
    );

    for i in 0..4 {
        if player_hands_length[i] == 0 {
            available_cards_remove_all_of(
                &mut player_allowed_to_have[i],
                &remaining_cards_as_vec
            );
        }
    }
}

fn players_who_can_this_card_be_assigned_to(
    current_card: DoCard,
    player_allowed_to_have: &[Vec<DoCard>; 4],
    player_hands_length: &[usize; 4]
) -> Vec<DoPlayer> {
    let mut players_that_can_have_this_card = vec![];

    for i in 0..4 {
        if player_allowed_to_have[i].contains(&current_card) &&
            player_hands_length[i] > 0 {
            players_that_can_have_this_card.push(i);
        }
    }

    return players_that_can_have_this_card;
}

fn distribute_single_cards(
    remaining_cards_as_vec: &mut Vec<DoCard>,
    player_hands: &mut [Vec<DoCard>; 4],
    player_hands_length: &mut [usize; 4],
    player_allowed_to_have: &mut [Vec<DoCard>; 4],
) -> bool {
    // Wir weisen eine Karte, die nur einem Spieler zugewiesen werden kann,
    // diesem Spieler zu.
    let current_card_to_assign = remaining_cards_as_vec
        .iter()
        .map(|&card| {
            let players_that_can_have_this_card = players_who_can_this_card_be_assigned_to(
                card,
                &player_allowed_to_have,
                &player_hands_length
            );

            (card, players_that_can_have_this_card)
        })
        .find(|(card, players_that_can_have_this_card)| {
            players_that_can_have_this_card.len() == 1
        });


    let (current_card_to_assign, player_that_has_this_card) = match current_card_to_assign {
        None => return false,
        Some((card, players_that_can_have_this_card)) => (card, players_that_can_have_this_card[0])
    };

    // println!("assigning {:?} to {:?}, because it can only be assigned to this player", current_card_to_assign, player_that_has_this_card);

    distribute_card(
        player_that_has_this_card,
        current_card_to_assign,
        player_hands,
        player_hands_length,
        player_allowed_to_have,
        remaining_cards_as_vec,
    );

    return true;
}

fn distribute_exactly_as_per_hand(
    player_hands: &mut [Vec<DoCard>; 4],
    player_hands_length: &mut [usize; 4],
    player_allowed_to_have: &mut [Vec<DoCard>; 4],
    remaining_cards_as_vec: &mut Vec<DoCard>,
    rng: &mut SmallRng
) -> bool {
    let mut can_this_happe_multiple_times = 0;

    // println!("player_hands_length: {:?}", player_hands_length);
    // println!("player_allowed_to_have: {:?}", player_allowed_to_have);

    for i in 0..4 {
        if player_hands_length[i] == 0 {
            continue;
        }

        if player_hands_length[i] == player_allowed_to_have[i].len() {
            //println!("i: {}", i);

            // Hier der Fall, alle zuweisen
            for card in player_allowed_to_have[i].clone() {
                distribute_card(
                    i,
                    card,
                    player_hands,
                    player_hands_length,
                    player_allowed_to_have,
                    remaining_cards_as_vec,
                );
            }

            return true;
        }
    }


    return can_this_happe_multiple_times > 0;
}

fn distribute_single_card_randomly(
    remaining_cards_as_vec: &mut Vec<DoCard>,
    player_hands: &mut [Vec<DoCard>; 4],
    player_hands_length: &mut [usize; 4],
    player_allowed_to_have: &mut [Vec<DoCard>; 4],
    rng: &mut SmallRng
) -> bool {
    let current_card_to_assign = remaining_cards_as_vec
        .iter()
        .choose(rng);

    // println!("current_card_to_assign: {:?}", current_card_to_assign);

    let current_card_to_assign = match current_card_to_assign {
        None => return false,
        Some(card) => *card
    };

    let players_that_can_have_this_card = players_who_can_this_card_be_assigned_to(
        current_card_to_assign,
        player_allowed_to_have,
        player_hands_length
    );

    // if (players_that_can_have_this_card.len() == 0) {
    //     println!(":-(")
    // }
    debug_assert!(players_that_can_have_this_card.len() > 0);

    let player_that_has_this_card: DoPlayer = players_that_can_have_this_card
        .into_iter()
        .choose(rng)
        .unwrap();

    distribute_card(
        player_that_has_this_card,
        current_card_to_assign,
        player_hands,
        player_hands_length,
        player_allowed_to_have,
        remaining_cards_as_vec,
    );

    return true;
}

pub fn sample_assignment_full(
    state: &DoState,
    observation: &DoObservation,

    rng: &mut SmallRng
) -> DoState {
    let tricks = &observation.tricks;

    let mut hand_lengths: [usize; 4] = [0,0,0,0];

    for i in 0..4 {
        hand_lengths[i] = hand_len(state.hands[i]);
    }

    let player_hands = sample_assignment(
        observation.wedding_player_if_wedding_announced,
        &tricks,
        observation.observing_player_hand,
        &hand_lengths,
        observation.observing_player,
        rng
    );

    let player_hands_arr = [
        hand_from_vec(player_hands[0].clone()),
        hand_from_vec(player_hands[1].clone()),
        hand_from_vec(player_hands[2].clone()),
        hand_from_vec(player_hands[3].clone()),
    ];

    return state.clone_with_different_hands(
        player_hands_arr
    );
}

pub fn sample_assignment(
    player_marriage: Option<DoPlayer>,

    // ToDo: Hochzeit testen??!
    tricks: &[Option<DoTrick>; 12],
    player_hand_observing_player: DoHand,
    player_hands_length: &[usize; 4],
    observing_player: DoPlayer,

    rng: &mut SmallRng
) -> [Vec<DoCard>; 4] {
    let remaining_cards = calc_remaining_cards(
        tricks,
        player_hand_observing_player,
    );

    let mut remaining_cards_as_vec = hand_to_vec(
        remaining_cards
    );

    let mut player_hands = [
        vec![],
        vec![],
        vec![],
        vec![]
    ];

    let mut player_allowed_to_have = player_allowed_to_have(
        player_marriage,
        tricks,
        player_hand_observing_player,
        observing_player
    );

    // println!("====");
    // println!("player_marriage: {:?}", player_marriage);
    // println!("tricks: {:?}", tricks);
    // println!("player_hand_observing_player: {:?}", player_hand_observing_player);
    // println!("player_hands_length: {:?}", player_hands_length);
    // println!("observing_player: {:?}", observing_player);

    let mut player_hands_length = player_hands_length.clone();

    // println!("===");
    // println!("remaining_cards_as_vec: {:?}", remaining_cards_as_vec);
    // println!("player_allowed_to_have: {:?}", player_allowed_to_have);
    // Wir verteilen alle Karten die auf die Spieler verteilt werden
    // müssen, bis keine Karten mehr übrig sind.
    loop {
        let result = distribute_single_cards(
            &mut remaining_cards_as_vec,
            &mut player_hands,
            &mut player_hands_length,
            &mut player_allowed_to_have,
        );

        if (result) {
            continue;
        }

        let result = distribute_exactly_as_per_hand(
            &mut player_hands,
            &mut player_hands_length,
            &mut player_allowed_to_have,
            &mut remaining_cards_as_vec,
            rng
        );

        if result {
            continue;
        }

        let result = distribute_single_card_randomly(
            &mut remaining_cards_as_vec,
            &mut player_hands,
            &mut player_hands_length,
            &mut player_allowed_to_have,
            rng
        );

        if !result {
            break;
        }
    }

    player_hands[observing_player] = hand_to_vec(player_hand_observing_player);

    return player_hands;
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ptr::hash;
    use rand::SeedableRng;
    use rs_doko::card::cards::DoCard::{ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondQueen, HeartAce, HeartKing, HeartNine, SpadeJack, SpadeKing};
    use rs_doko::hand::hand::hand_from_vec;
    use super::*;
    use rs_doko::player::player::{DoPlayer, PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};

    #[test]
    fn test_calc_remaining_cards() {
        // Ein Normalspiel ohne Stiche. Es sind also alle Karten, außer denen auf der Hand des Spielers, noch im Spiel.
        let result = player_allowed_to_have(
            None,
            &[
                None, None, None, None, None, None, None, None, None, None, None, None
            ],
            hand_from_vec(vec![
                DoCard::ClubQueen,
                DoCard::ClubJack,
                DoCard::ClubJack,
                DoCard::SpadeJack,
                DoCard::DiamondNine,

                DoCard::ClubTen,
                DoCard::ClubKing,
                DoCard::ClubKing,
                DoCard::ClubNine,

                DoCard::SpadeAce,
                DoCard::SpadeNine,
                DoCard::HeartNine
            ]),
            PLAYER_BOTTOM
        );

        assert_eq!(result[0], vec![]);
        assert_eq!(result[1], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubQueen, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[2], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubQueen, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[3], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubQueen, DoCard::ClubAce, DoCard::ClubAce]);

        // Eine angesagte Hochzeit, der Hochzeitsspieler ist der beobachtende Spieler.
        let result = player_allowed_to_have(
            Some(PLAYER_BOTTOM),
            &[
                None, None, None, None, None, None, None, None, None, None, None, None
            ],
            hand_from_vec(vec![
                DoCard::ClubQueen,
                DoCard::ClubQueen,
                DoCard::ClubJack,
                DoCard::SpadeJack,
                DoCard::DiamondNine,

                DoCard::ClubTen,
                DoCard::ClubKing,
                DoCard::ClubKing,
                DoCard::ClubNine,

                DoCard::SpadeAce,
                DoCard::SpadeNine,
                DoCard::HeartNine
            ]),
            PLAYER_BOTTOM
        );

        assert_eq!(result[0], vec![]);
        assert_eq!(result[1], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubJack, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[2], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubJack, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[3], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeJack, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubJack, DoCard::ClubAce, DoCard::ClubAce]);

        // Eine angesagte Hochzeit, der Hochzeitsspieler ist ein anderer als der beobachtende Spieler.
        let result = player_allowed_to_have(
            Some(PLAYER_TOP),
            &[
                None, None, None, None, None, None, None, None, None, None, None, None
            ],
            hand_from_vec(vec![
                DoCard::ClubJack,
                DoCard::ClubJack,
                DoCard::SpadeJack,
                DoCard::SpadeJack,
                DoCard::DiamondNine,

                DoCard::ClubTen,
                DoCard::ClubKing,
                DoCard::ClubKing,
                DoCard::ClubNine,

                DoCard::SpadeAce,
                DoCard::SpadeNine,
                DoCard::HeartNine
            ]),
            PLAYER_BOTTOM
        );

        assert_eq!(result[0], vec![]);
        assert_eq!(result[1], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[2], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubQueen, DoCard::ClubQueen, DoCard::ClubAce, DoCard::ClubAce]);
        assert_eq!(result[3], vec![DoCard::DiamondNine, DoCard::DiamondTen, DoCard::DiamondTen, DoCard::DiamondJack, DoCard::DiamondJack, DoCard::DiamondQueen, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::DiamondKing, DoCard::DiamondAce, DoCard::DiamondAce, DoCard::HeartNine, DoCard::HeartTen, DoCard::HeartTen, DoCard::HeartJack, DoCard::HeartJack, DoCard::HeartQueen, DoCard::HeartQueen, DoCard::HeartKing, DoCard::HeartKing, DoCard::HeartAce, DoCard::HeartAce, DoCard::SpadeNine, DoCard::SpadeTen, DoCard::SpadeTen, DoCard::SpadeQueen, DoCard::SpadeQueen, DoCard::SpadeKing, DoCard::SpadeKing, DoCard::SpadeAce, DoCard::ClubNine, DoCard::ClubTen, DoCard::ClubAce, DoCard::ClubAce]);

        // Ein Normalspiel, mit 9 Stichen und einem angefangenen Stich
        // https://www.online-doppelkopf.com/spiele/97.208.474
        let result = player_allowed_to_have(
            None,
            &[
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::DiamondKing,
                    DoCard::DiamondNine,
                    DoCard::SpadeQueen,
                    DoCard::ClubQueen
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubTen,
                    DoCard::ClubNine,
                    DoCard::ClubKing
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeKing,
                    DoCard::SpadeNine,
                    DoCard::SpadeNine
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeJack,
                    DoCard::SpadeQueen,
                    DoCard::DiamondNine,
                    DoCard::DiamondJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::ClubQueen,
                    DoCard::HeartJack,
                    DoCard::DiamondJack,
                    DoCard::ClubJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondKing,
                    DoCard::HeartJack,
                    DoCard::HeartQueen
                ])),
                // PLAYER_RIGHT hat garantiert kein Trumpf mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondTen,
                    DoCard::DiamondTen,
                    DoCard::SpadeTen
                ])),
                // PLAYER_TOP hat garantiert kein Herz mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartAce,
                    DoCard::HeartKing,
                    DoCard::DiamondAce,
                    DoCard::HeartNine
                ])),
                // PLAYER_BOTTOM hat garantiert kein Pik mehr
                // PLAYER_LEFT hat garantiert kein Pik mehr
                Some(DoTrick::existing(PLAYER_TOP, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeTen,
                    DoCard::DiamondQueen,
                    DoCard::HeartQueen
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubKing
                ])),
                // Es verbleiben Karten:
                // DoCard::HeartAce, DoCard::ClubNine,
                // DoCard::DiamondQueen, DoCard::ClubJack, DoCard::HeartKing, DoCard::SpadeJack
                // DoCard::DiamondAce, DoCard::SpadeKing, DoCard::HeartNine, DoCard::ClubTen
                None,
                None
            ],
            hand_from_vec(vec![
                DoCard::ClubNine,
                DoCard::SpadeJack,
                DoCard::ClubTen
            ]),
            PLAYER_BOTTOM
        );

        assert_eq!(result[0], vec![]);
        assert_eq!(result[1], vec![DiamondQueen, DiamondAce, HeartNine, HeartKing, HeartAce, ClubJack]);
        assert_eq!(result[2], vec![DiamondQueen, DiamondAce, SpadeKing, ClubJack]);
        assert_eq!(result[3], vec![HeartNine, HeartKing, HeartAce, SpadeKing]);
    }

    #[test]
    fn test_remaining_cards() {
        let remaining_cards = calc_remaining_cards(
            &[
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::DiamondKing,
                    DoCard::DiamondNine,
                    DoCard::SpadeQueen,
                    DoCard::ClubQueen
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubTen,
                    DoCard::ClubNine,
                    DoCard::ClubKing
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeKing,
                    DoCard::SpadeNine,
                    DoCard::SpadeNine
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeJack,
                    DoCard::SpadeQueen,
                    DoCard::DiamondNine,
                    DoCard::DiamondJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::ClubQueen,
                    DoCard::HeartJack,
                    DoCard::DiamondJack,
                    DoCard::ClubJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondKing,
                    DoCard::HeartJack,
                    DoCard::HeartQueen
                ])),
                // PLAYER_RIGHT hat garantiert kein Trumpf mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondTen,
                    DoCard::DiamondTen,
                    DoCard::SpadeTen
                ])),
                // PLAYER_TOP hat garantiert kein Herz mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartAce,
                    DoCard::HeartKing,
                    DoCard::DiamondAce,
                    DoCard::HeartNine
                ])),
                // PLAYER_BOTTOM hat garantiert kein Pik mehr
                // PLAYER_LEFT hat garantiert kein Pik mehr
                Some(DoTrick::existing(PLAYER_TOP, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeTen,
                    DoCard::DiamondQueen,
                    DoCard::HeartQueen
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubKing
                ])),
                // Es verbleiben Karten:
                // DoCard::HeartAce, DoCard::ClubNine,
                // DoCard::DiamondQueen, DoCard::ClubJack, DoCard::HeartKing, DoCard::SpadeJack
                // DoCard::DiamondAce, DoCard::SpadeKing, DoCard::HeartNine, DoCard::ClubTen
                None,
                None
            ],
            hand_from_vec(vec![
                DoCard::ClubNine,
                DoCard::SpadeJack,
                DoCard::ClubTen
            ]),
        );

        let remaining_cards_as_vec = hand_to_vec(
            remaining_cards
        );

        // DoCard::HeartAce, DoCard::ClubNine,
        // DoCard::DiamondQueen, DoCard::ClubJack, DoCard::HeartKing, DoCard::SpadeJack
        // DoCard::DiamondAce, DoCard::SpadeKing, DoCard::HeartNine, DoCard::ClubTen
        assert_eq!(remaining_cards_as_vec, vec![
            DiamondQueen, DiamondAce, HeartNine, HeartKing, HeartAce, ClubJack, SpadeKing
        ]);
    }

    fn brute_force_assignments(
        player_marriage: Option<DoPlayer>,

        tricks: &[Option<DoTrick>; 12],
        player_hand_observing_player: DoHand,
        player_hands_length: &[usize; 4],
        observing_player: DoPlayer,
        rng: &mut SmallRng
    ) -> Vec<[Vec<DoCard>; 4]> {
        let mut all_samples = vec![];

        for i in 0..1000 {
            let result = sample_assignment(
                player_marriage,
                tricks,
                player_hand_observing_player,
                player_hands_length,
                observing_player,
                rng
            );

            all_samples.push(result);
        }

        let unique_samples = all_samples
            .iter()
            .map(|player_cards| {
                let mut player_hands = player_cards.clone();

                player_hands
                    .map(|hand| {
                    let mut cards = hand.clone();
                    cards.sort_by_key(|card| *card as usize);
                    cards
                })
            })
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        return unique_samples;
    }

    #[test]
    fn test_sample_assignment_unique() {
        // https://www.online-doppelkopf.com/spiele/97.183.100
        let mut rng = SmallRng::seed_from_u64(42);

            // Ein Normalspiel, hier gibt es wenigstens eine eindeutige Zuordnung.
        let result = brute_force_assignments(
            None,
            &[
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::ClubAce,
                    DoCard::ClubTen,
                    DoCard::DiamondAce,
                    DoCard::ClubAce
                ])),
                Some(DoTrick::existing(PLAYER_TOP, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeTen,
                    DoCard::SpadeKing,
                    DoCard::DiamondKing,
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::DiamondJack,
                    DoCard::DiamondNine,
                    DoCard::DiamondQueen,
                    DoCard::DiamondNine
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::HeartAce,
                    DoCard::HeartNine,
                    DoCard::HeartJack,
                    DoCard::HeartKing
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::SpadeJack,
                    DoCard::DiamondKing,
                    DoCard::DiamondTen,
                    DoCard::ClubJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::DiamondAce,
                    DoCard::HeartTen,
                    DoCard::DiamondJack,
                    DoCard::DiamondTen
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::HeartQueen,
                    DoCard::HeartJack,
                    DoCard::ClubNine,
                    DoCard::SpadeJack,
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::HeartQueen,
                    DoCard::HeartTen,
                    DoCard::ClubTen,
                    DoCard::SpadeQueen
                ])),
                Some(DoTrick::existing(PLAYER_TOP, vec![
                    DoCard::SpadeNine,
                    DoCard::SpadeNine,
                    DoCard::SpadeAce,
                    DoCard::ClubKing
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::SpadeTen,
                    DoCard::SpadeQueen,
                    DoCard::HeartAce
                ])),
                // Es verbleiben Karten:
                // DoCard::SpadeKing
                // DoCard::ClubQueen, DoCard::ClubQueen, DoCard::ClubJack, DoCard::HeartNine
                // DoCard::HeartKing, DoCard::ClubNine, DoCard::DiamondQueen, DoCard::ClubKing
                None,
                None
            ],
            hand_from_vec(vec![
                DoCard::ClubQueen,
                DoCard::HeartKing
            ]),
            &[2, 2, 2, 3],
            PLAYER_BOTTOM,
            &mut rng
        );


        assert_eq!(result.len(), 9);
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![DiamondQueen, ClubNine], vec![ClubJack, ClubQueen], vec![HeartNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubNine, ClubJack], vec![DiamondQueen, ClubQueen], vec![HeartNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![DiamondQueen, ClubJack], vec![HeartNine, ClubQueen], vec![ClubNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubNine, ClubQueen], vec![DiamondQueen, ClubJack], vec![HeartNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubQueen, ClubKing], vec![DiamondQueen, ClubJack], vec![HeartNine, ClubNine, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubJack, ClubQueen], vec![DiamondQueen, HeartNine], vec![ClubNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubJack, ClubKing], vec![DiamondQueen, ClubQueen], vec![HeartNine, ClubNine, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![DiamondQueen, ClubQueen], vec![HeartNine, ClubJack], vec![ClubNine, ClubKing, SpadeKing]]));
        assert!(result.contains(&[vec![HeartKing, ClubQueen], vec![ClubQueen, ClubKing], vec![DiamondQueen, ClubJack], vec![HeartNine, ClubNine, SpadeKing]]));

    }

    #[test]
    fn test_sample_assignment() {
        // https://www.online-doppelkopf.com/spiele/97.208.474
        let mut rng = SmallRng::seed_from_u64(42);

        // Ein Normalspiel, hier gibt es aber keine eindeutige Zuordnung.
        let result = brute_force_assignments(
            None,
            &[
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::DiamondKing,
                    DoCard::DiamondNine,
                    DoCard::SpadeQueen,
                    DoCard::ClubQueen
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubTen,
                    DoCard::ClubNine,
                    DoCard::ClubKing
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeKing,
                    DoCard::SpadeNine,
                    DoCard::SpadeNine
                ])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![
                    DoCard::SpadeJack,
                    DoCard::SpadeQueen,
                    DoCard::DiamondNine,
                    DoCard::DiamondJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::ClubQueen,
                    DoCard::HeartJack,
                    DoCard::DiamondJack,
                    DoCard::ClubJack
                ])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondKing,
                    DoCard::HeartJack,
                    DoCard::HeartQueen
                ])),
                // PLAYER_RIGHT hat garantiert kein Trumpf mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartTen,
                    DoCard::DiamondTen,
                    DoCard::DiamondTen,
                    DoCard::SpadeTen
                ])),
                // PLAYER_TOP hat garantiert kein Herz mehr
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![
                    DoCard::HeartAce,
                    DoCard::HeartKing,
                    DoCard::DiamondAce,
                    DoCard::HeartNine
                ])),
                // PLAYER_BOTTOM hat garantiert kein Pik mehr
                // PLAYER_LEFT hat garantiert kein Pik mehr
                Some(DoTrick::existing(PLAYER_TOP, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeTen,
                    DoCard::DiamondQueen,
                    DoCard::HeartQueen
                ])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![
                    DoCard::ClubAce,
                    DoCard::ClubKing
                ])),
                // Es verbleiben Karten:
                // DoCard::HeartAce, DoCard::ClubNine,
                // DoCard::DiamondQueen, DoCard::ClubJack, DoCard::HeartKing, DoCard::SpadeJack
                // DoCard::DiamondAce, DoCard::SpadeKing, DoCard::HeartNine, DoCard::ClubTen
                None,
                None
            ],
            hand_from_vec(vec![
                DoCard::ClubNine,
                DoCard::SpadeJack,
                DoCard::ClubTen
            ]),
            &[3, 2, 2, 3],
            PLAYER_BOTTOM,
            &mut rng
        );

        assert_eq!(result.len(), 12);
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondQueen, DiamondAce], vec![ClubJack, SpadeKing], vec![HeartNine, HeartKing, HeartAce]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![HeartAce, ClubJack], vec![DiamondQueen, DiamondAce], vec![HeartNine, HeartKing, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![HeartNine, ClubJack], vec![DiamondQueen, DiamondAce], vec![HeartKing, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondAce, ClubJack], vec![DiamondQueen, SpadeKing], vec![HeartNine, HeartKing, HeartAce]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondAce, HeartKing], vec![DiamondQueen, ClubJack], vec![HeartNine, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondAce, HeartAce], vec![DiamondQueen, ClubJack], vec![HeartNine, HeartKing, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondQueen, HeartAce], vec![DiamondAce, ClubJack], vec![HeartNine, HeartKing, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondAce, HeartNine], vec![DiamondQueen, ClubJack], vec![HeartKing, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondQueen, HeartNine], vec![DiamondAce, ClubJack], vec![HeartKing, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondQueen, HeartKing], vec![DiamondAce, ClubJack], vec![HeartNine, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![HeartKing, ClubJack], vec![DiamondQueen, DiamondAce], vec![HeartNine, HeartAce, SpadeKing]]));
        assert!(result.contains(&[vec![ClubNine, ClubTen, SpadeJack], vec![DiamondQueen, ClubJack], vec![DiamondAce, SpadeKing], vec![HeartNine, HeartKing, HeartAce]]));
    }

    #[test]
    fn sample_assignment_fehlerfall() {
        let mut rng = SmallRng::seed_from_u64(42);

        let result = brute_force_assignments(
            None,
            &[
                Some(DoTrick::existing(2, vec![
                    DoCard::HeartNine,
                    DoCard::HeartTen,
                    DoCard::HeartNine,
                    DoCard::HeartKing
                ])),
                Some(DoTrick::existing(3, vec![
                    DoCard::DiamondTen,
                    DoCard::DiamondKing,
                    DoCard::HeartJack,
                    DoCard::DiamondTen
                ])),
                Some(DoTrick::existing(1, vec![
                    DoCard::ClubAce,
                    DoCard::SpadeJack,
                    DoCard::ClubKing,
                    DoCard::ClubTen
                ])),
                Some(DoTrick::existing(2, vec![
                    DoCard::SpadeAce,
                    DoCard::SpadeNine,
                    DoCard::SpadeAce,
                    DoCard::SpadeTen
                ])),
                Some(DoTrick::existing(2, vec![
                    DoCard::HeartAce,
                    DoCard::ClubNine,
                    DoCard::HeartAce,
                    DoCard::DiamondKing
                ])),
                Some(DoTrick::existing(1, vec![
                    DoCard::ClubKing,
                    DoCard::DiamondNine,
                    DoCard::ClubAce,
                    DoCard::SpadeQueen
                ])),
                Some(DoTrick::existing(0, vec![
                    DoCard::DiamondJack,
                    DoCard::HeartQueen,
                    DoCard::DiamondQueen,
                    DoCard::ClubQueen
                ])),
                Some(DoTrick::existing(3, vec![
                    DoCard::DiamondAce,
                    DoCard::ClubQueen,
                    DoCard::DiamondNine,
                    DoCard::DiamondQueen
                ])),
                Some(DoTrick::existing(0, vec![
                    DoCard::DiamondAce,
                    DoCard::SpadeQueen,
                    DoCard::HeartQueen,
                    DoCard::HeartTen
                ])),
                Some(DoTrick::existing(3, vec![
                    DoCard::SpadeTen,
                    DoCard::DiamondJack,
                    DoCard::SpadeNine,
                    DoCard::HeartKing
                ])),
                Some(DoTrick::existing(0, vec![
                    DoCard::ClubJack
                ])),
                None
            ],
            4198400,
            &[1, 2, 2, 2],
            PLAYER_LEFT,
            &mut rng
        );


    }

}