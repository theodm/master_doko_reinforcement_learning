use cli_table::{Cell, CellStruct, Style, Table};
use cli_table::format::{Align, Justify};
use crate::player::player_set::DoPlayerSet;
use crate::reservation::reservation::DoReservation;
use crate::card::cards::DoCard;
use crate::observation::observation::DoObservation;
use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
use crate::player::player_set::player_set_contains;
use crate::reservation::reservation::DoVisibleReservation;
use crate::state::state::DoState;
use crate::stats::stats::DoEndOfGameStats;
use crate::util::RotArr;

fn reservation_to_str(
    reservation: Option<DoReservation>
) -> &'static str {
    match reservation {
        Some(reservation) => {
            match reservation {
                DoReservation::Healthy => "G",
                DoReservation::Wedding => "H"
            }
        }
        None => "",
    }
}

fn party_to_str(
    re_players: Option<DoPlayerSet>,
    player_index: usize
) -> &'static str {
    match re_players {
        Some(re_players) => {
            if player_set_contains(re_players, player_index) {
                "Re"
            } else {
                "Kontra"
            }
        }
        None => "?",
    }
}

fn player_name(player_index: usize) -> &'static str {
    match player_index {
        PLAYER_BOTTOM => "Bottom",
        PLAYER_LEFT => "Left",
        PLAYER_TOP => "Top",
        PLAYER_RIGHT => "Right",
        _ => panic!("Invalid player index {}", player_index),
    }
}

//
fn trick_for_index(
    observation: &DoObservation,
    index: usize
) -> Vec<CellStruct> {
    let starting_player = observation.game_starting_player;

    let trick = observation
        .tricks[index]
        .as_ref();

    // Wenn es den Stich mit diesem Index noch nicht gibt, dann
    // geben wir eine leere Zeile zurück.
    if trick.is_none() {
        return (0..4)
            .map(|_| "".to_string())
            .map(|it| it.cell().justify(Justify::Center))
            .collect::<Vec<_>>();
    }

    let trick = trick
        .unwrap();

    let trick_starting_player = trick
        .start_player;

    return RotArr::RotArr::new_from_perspective(
        trick_starting_player,
        trick.cards.clone()
    )
        .map_indexed(|i, c| {
            match c {
                None => { return "".to_string(); }
                Some(card) => {
                    let card_as_str = card
                        .to_string();

                    if i == 0 {
                        // Die erste Karte im Stich wird mit einem * markiert.
                        format!("{card_as_str}*")
                    } else {
                        card_as_str
                    }
                }
            }
        })
        .rotate_to_perspective(starting_player)
        .extract()
        .iter()
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();
}

pub fn display_game(
    observation: DoObservation,
) {
    let players = RotArr::RotArr::new_from_0(
        PLAYER_BOTTOM,
        [PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_TOP, PLAYER_RIGHT]
    );

    // Die Tabelle beginnt aus der Perspektive des Spielers,
    // der das Spiel gestartet hat.    
    let starting_player = observation
        .game_starting_player;

    // Als erste Zeile zeigen wir die Namen an.
    //
    // +--------+------+--------+--------+
    // | Bottom | Left |  Top   | Right  |
    // +--------+------+--------+--------+
    let header_row = players
        .clone()
        .map(|it| player_name(it))
        .rotate_to_perspective(starting_player)
        .extract()
        .iter()
        .map(|it| it.cell().align(Align::Center).bold(true))
        .collect::<Vec<_>>();

    // Vorbehaltsangaben
    //
    // Vorbehalte sind schon in der richtigen Reihenfolge
    // da sie immer ausgehend vom Startspieler gelten.
    //
    // +--------+------+--------+--------+
    // |   G    |  G   |   G    |   G    |
    // +--------+------+--------+--------+
    let reservations = observation
        .phi_real_reservations
        .reservations
        .iter()
        .map(|r| reservation_to_str(*r))
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    // Angabe der Parteizugehörigkeit
    //
    // +--------+------+--------+--------+
    // |   Re   |  Re  | Kontra | Kontra |
    // +--------+------+--------+--------+
    let re_kontra_line = players
        .clone()
        .map(|player_index| party_to_str(observation.phi_re_players, player_index))
        .rotate_to_perspective(starting_player)
        .extract()
        .iter()
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    let eog_stats = observation
        .clone()
        .finished_observation;

    // Die bisher gesammelten Augen der Spieler werden angezeigt.
    //
    // +--------+------+--------+--------+
    // |   70   | 112  |   22   |   36   |
    // +--------+------+--------+--------+
    let eyes = RotArr::RotArr::new_from_0(starting_player, observation.player_eyes)
        .extract()
        .iter()
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    // Die Punktzahl, falls das Spiel beendet ist.
    //
    // +--------+------+--------+--------+
    // |   3    |  -3  |   -3   |   3    |
    // +--------+------+--------+--------+
    let points = match eog_stats {
        None => {
            (0..4)
                .map(|_| "".to_string())
                .map(|it| it.cell().justify(Justify::Center))
                .collect::<Vec<_>>()
        }
        Some(eog_stats) => {
            RotArr::RotArr::new_from_0(starting_player, eog_stats.player_points)
                .extract()
                .iter()
                .map(|it| it.cell().justify(Justify::Center))
                .collect::<Vec<_>>()
        }
    };

    let table = vec![
        reservations,
        re_kontra_line,
        trick_for_index(&observation, 0),
        trick_for_index(&observation, 1),
        trick_for_index(&observation, 2),
        trick_for_index(&observation, 3),
        trick_for_index(&observation, 4),
        trick_for_index(&observation, 5),
        trick_for_index(&observation, 6),
        trick_for_index(&observation, 7),
        trick_for_index(&observation, 8),
        trick_for_index(&observation, 9),
        trick_for_index(&observation, 10),
        trick_for_index(&observation, 11),
        eyes,
        points,
    ].table()
        .title(header_row);

    println!("{}", table.display().unwrap());

}

#[cfg(test)]
mod tests {
    use crate::action::allowed_actions::allowed_actions_from_vec;
    use crate::basic::phase::DoPhase;
    use crate::basic::team::DoTeam;
    use crate::card::cards::DoCard;
    use crate::hand::hand::hand_from_vec;
    use crate::observation::observation::DoObservation;
    use super::*;
    use crate::player::player::{PLAYER_BOTTOM, PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use crate::player::player_set::player_set_create;
    use crate::reservation::reservation::{DoReservation, DoVisibleReservation};
    use crate::reservation::reservation_round::DoReservationRound;
    use crate::state::state::DoState;
    use crate::stats::stats::DoEndOfGameStats;
    use crate::trick::trick::DoTrick;

    #[test]
    fn test_display_finished_game() {
        let obs = DoObservation {
            phase: DoPhase::Finished,
            observing_player: PLAYER_BOTTOM,
            current_player: None,
            allowed_actions_current_player: allowed_actions_from_vec(vec![]),
            game_starting_player: PLAYER_BOTTOM,
            wedding_player_if_wedding_announced: None,
            tricks: [
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::ClubQueen, DoCard::DiamondJack, DoCard::ClubJack, DoCard::HeartJack])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::DiamondJack, DoCard::ClubQueen, DoCard::DiamondNine, DoCard::SpadeQueen])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::SpadeKing, DoCard::SpadeTen, DoCard::SpadeAce, DoCard::SpadeAce])),
                Some(DoTrick::existing(PLAYER_RIGHT, vec![DoCard::HeartNine, DoCard::HeartAce, DoCard::HeartNine, DoCard::HeartKing])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::HeartTen, DoCard::SpadeQueen, DoCard::DiamondQueen, DoCard::HeartTen])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::DiamondTen, DoCard::DiamondNine, DoCard::HeartJack, DoCard::DiamondTen])),
                Some(DoTrick::existing(PLAYER_TOP, vec![DoCard::DiamondKing, DoCard::SpadeJack, DoCard::HeartQueen, DoCard::ClubAce])),
                Some(DoTrick::existing(PLAYER_BOTTOM, vec![DoCard::SpadeJack, DoCard::DiamondQueen, DoCard::DiamondKing, DoCard::SpadeKing])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::HeartQueen, DoCard::ClubTen, DoCard::HeartAce, DoCard::ClubKing])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::DiamondAce, DoCard::HeartKing, DoCard::ClubNine, DoCard::ClubAce])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::ClubJack, DoCard::SpadeNine, DoCard::ClubNine, DoCard::SpadeTen])),
                Some(DoTrick::existing(PLAYER_LEFT, vec![DoCard::DiamondAce, DoCard::ClubKing, DoCard::ClubTen, DoCard::SpadeNine]))
            ],
            visible_reservations: [Some(DoVisibleReservation::Healthy), Some(DoVisibleReservation::Healthy), Some(DoVisibleReservation::Healthy), Some(DoVisibleReservation::Healthy)],
            player_eyes: [70, 112, 22, 36],
            observing_player_hand: hand_from_vec(vec![]),
            finished_observation: Some(DoEndOfGameStats { winning_team: DoTeam::Re, re_players: 3, is_solo: false, player_eyes: [70, 112, 22, 36], re_eyes: 182, kontra_eyes: 58, re_points: 3, kontra_points: -3, player_points: [3, -3, -3, 3] }),
            phi_re_players: Some(player_set_create(vec![PLAYER_BOTTOM, PLAYER_LEFT])),
            phi_real_reservations: DoReservationRound::existing(PLAYER_BOTTOM, vec![DoReservation::Healthy, DoReservation::Healthy, DoReservation::Healthy, DoReservation::Healthy]),
            phi_real_hands: [
                hand_from_vec(vec![]),
                hand_from_vec(vec![]),
                hand_from_vec(vec![]),
                hand_from_vec(vec![])
            ],
            phi_team_eyes: [15, 15],
        };

        display_game(obs);
    }
}