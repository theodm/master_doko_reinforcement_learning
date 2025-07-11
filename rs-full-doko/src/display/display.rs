use crate::announcement::announcement::{FdoAnnouncement, FdoAnnouncementOccurrence};
use crate::basic::team::FdoTeam;
use crate::observation::observation::FdoObservation;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation::FdoReservation;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;
use cli_table::format::{Align, Justify};
use cli_table::{Cell, CellStruct, ColorChoice, Style, Table};

fn reservation_to_str(reservation: Option<FdoReservation>) -> &'static str {
    match reservation {
        Some(reservation) => match reservation {
            FdoReservation::Healthy => "G",
            FdoReservation::Wedding => "H",

            FdoReservation::DiamondsSolo => "♦",
            FdoReservation::HeartsSolo => "♥",
            FdoReservation::SpadesSolo => "♠",
            FdoReservation::ClubsSolo => "♣",

            FdoReservation::QueensSolo => "Q",
            FdoReservation::JacksSolo => "J",

            FdoReservation::TrumplessSolo => "T",
        },
        None => " ",
    }
}

fn party_to_str(re_players: Option<FdoPlayerSet>, player: FdoPlayer) -> &'static str {
    match re_players {
        Some(re_players) => {
            if re_players.contains(player) {
                "Re"
            } else {
                "Kontra"
            }
        }
        None => "?",
    }
}

fn player_name(player: FdoPlayer) -> &'static str {
    match player.index() {
        0 => "Bottom",
        1 => "Left",
        2 => "Top",
        3 => "Right",
        _ => panic!("Invalid player index"),
    }
}

fn map_announcement_str(announcement: FdoAnnouncement, player_team: FdoTeam) -> &'static str {
    return match player_team {
        FdoTeam::Re => match announcement {
            FdoAnnouncement::ReContra => "R",
            FdoAnnouncement::No90 => "R90",
            FdoAnnouncement::No60 => "R60",
            FdoAnnouncement::No30 => "R30",
            FdoAnnouncement::Black => "R0",
            FdoAnnouncement::CounterReContra => "R",
            _ => "",
        },
        FdoTeam::Kontra => match announcement {
            FdoAnnouncement::ReContra => "K",
            FdoAnnouncement::No90 => "K90",
            FdoAnnouncement::No60 => "K60",
            FdoAnnouncement::No30 => "K30",
            FdoAnnouncement::Black => "K0",
            FdoAnnouncement::CounterReContra => "K",
            _ => "",
        },
    };
}

fn map_announcement_occurence_str(
    occurence: &FdoAnnouncementOccurrence,
    re_players: FdoPlayerSet,
) -> String {
    let announcement_str =
        map_announcement_str(occurence.announcement, occurence.player.team(re_players));

    return format!(
        "{}: {}",
        player_name(occurence.player)
            .chars()
            .take(1)
            .collect::<String>(),
        announcement_str
    );
}

fn trick_announcements_for_index(
    observation: &FdoObservation,
    index: usize,
) -> Option<Vec<CellStruct>> {
    // Wenn es den Stich mit diesem Index noch nicht gibt, dann
    // geben wir eine leere Zeile zurück.
    if index >= observation.tricks.len() {
        return None;
    }

    let mut strs = vec![];
    for i in 0..4 {
        let card_index = (index * 4) + i;

        let announcements: Vec<&FdoAnnouncementOccurrence> = observation
            .announcements
            .iter()
            .filter(|a| a.card_index == card_index)
            .collect();

        let mut str: String = String::new();

        for announcement in announcements {
            let new_str = format!(
                "{}{}",
                map_announcement_occurence_str(announcement, observation.phi_re_players.unwrap()),
                "\n"
            );

            if new_str.len() > 0 {
                str.push_str(&new_str);
            }
        }

        if str.len() == 0 {
            str.push_str(" ");
        }

        strs.push(str);
    }

    if strs.join("").trim().is_empty() {
        return None;
    }

    return Some(
        strs.iter()
            .map(|it| it.cell().justify(Justify::Center))
            .collect::<Vec<_>>(),
    );
}

fn trick_for_index(observation: &FdoObservation, index: usize) -> Vec<CellStruct> {
    let starting_player = observation.game_starting_player;

    // Wenn es den Stich mit diesem Index noch nicht gibt, dann
    // geben wir eine leere Zeile zurück.
    if index >= observation.tricks.len() {
        return (0..4)
            .map(|_| " ".to_string())
            .map(|it| it.cell().justify(Justify::Center))
            .collect::<Vec<_>>();
    }

    let trick = observation.tricks[index].clone();

    let trick_starting_player = trick.starting_player();

    let mut card_index = 0;
    let trick_row = trick
        .cards
        .map(|card| {
            card_index += 1;
            (*card, card_index - 1)
        })
        .to_zero_array_remaining_option()
        .rotate_to(starting_player)
        .iter_with_player()
        .map(|(i, c)| {
            match c {
                None => {
                    return " ".to_string();
                }
                Some((card, _i)) => {
                    let card_as_str = card.to_string();

                    let mut card_as_str = if i == trick_starting_player {
                        // Die erste Karte im Stich wird mit einem * markiert.
                        format!("{card_as_str}*")
                    } else {
                        card_as_str
                    };

                    // card_as_str += format!(" {}", index * 4 + (_i)).as_str();

                    card_as_str
                }
            }
        })
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    return trick_row;
}

pub fn display_game(
    observation: FdoObservation
) -> String {
    let players = PlayerZeroOrientedArr::from_full([
        FdoPlayer::BOTTOM,
        FdoPlayer::LEFT,
        FdoPlayer::TOP,
        FdoPlayer::RIGHT,
    ]);

    // Die Tabelle beginnt aus der Perspektive des Spielers,
    // der das Spiel gestartet hat.
    let starting_player = observation.game_starting_player;

    // Als erste Zeile zeigen wir die Namen an.
    //
    // +--------+------+--------+--------+
    // | Bottom | Left |  Top   | Right  |
    // +--------+------+--------+--------+
    let header_row = players
        .map(|it| player_name(*it).to_string() + (if observation.current_player == Some(*it) { "*" } else { "" }))
        .rotate_to(starting_player)
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
        .to_array_remaining_option()
        .iter_with_player()
        .map(|r| reservation_to_str(*r.1))
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    // Angabe der Parteizugehörigkeit
    //
    // +--------+------+--------+--------+
    // |   Re   |  Re  | Kontra | Kontra |
    // +--------+------+--------+--------+
    let re_kontra_line = players
        .map(|player| party_to_str(observation.phi_re_players, *player))
        .rotate_to(starting_player)
        .iter()
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    let eog_stats = observation.clone().finished_stats;

    // Die bisher gesammelten Augen der Spieler werden angezeigt.
    //
    // +--------+------+--------+--------+
    // |   70   | 112  |   22   |   36   |
    // +--------+------+--------+--------+
    let eyes = observation
        .player_eyes
        .rotate_to(starting_player)
        .iter()
        .map(|it| it.cell().justify(Justify::Center))
        .collect::<Vec<_>>();

    // Die Punktzahl, falls das Spiel beendet ist.
    //
    // +--------+------+--------+--------+
    // |   3    |  -3  |   -3   |   3    |
    // +--------+------+--------+--------+
    let points = match eog_stats {
        None => (0..4)
            .map(|_| " ".to_string())
            .map(|it| it.cell().justify(Justify::Center))
            .collect::<Vec<_>>(),
        Some(eog_stats) => eog_stats
            .player_points
            .rotate_to(starting_player)
            .iter()
            .map(|it| it.cell().justify(Justify::Center))
            .collect::<Vec<_>>(),
    };

    let mut table = vec![];

    fn add(vec: &mut Vec<Vec<CellStruct>>, cell: Vec<CellStruct>) {
        vec.push(cell);
    }

    fn add_if_some(vec: &mut Vec<Vec<CellStruct>>, opt: Option<Vec<CellStruct>>) {
        match opt {
            Some(cell) => vec.push(cell),
            None => {}
        }
    }

    add(&mut table, reservations);
    add(&mut table, re_kontra_line);
    add_if_some(&mut table, trick_announcements_for_index(&observation, 0));
    add(&mut table, trick_for_index(&observation, 0));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 1));
    add(&mut table, trick_for_index(&observation, 1));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 2));
    add(&mut table, trick_for_index(&observation, 2));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 3));
    add(&mut table, trick_for_index(&observation, 3));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 4));
    add(&mut table, trick_for_index(&observation, 4));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 5));
    add(&mut table, trick_for_index(&observation, 5));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 6));
    add(&mut table, trick_for_index(&observation, 6));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 7));
    add(&mut table, trick_for_index(&observation, 7));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 8));
    add(&mut table, trick_for_index(&observation, 8));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 9));
    add(&mut table, trick_for_index(&observation, 9));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 10));
    add(&mut table, trick_for_index(&observation, 10));
    add_if_some(&mut table, trick_announcements_for_index(&observation, 11));
    add(&mut table, trick_for_index(&observation, 11));
    add(&mut table, eyes);
    add(&mut table, points);

    let _table = table.table().title(header_row)
        .color_choice(ColorChoice::Never);

    let hands = observation
        .phi_real_hands
        .rotate_to(starting_player)
        .iter_with_player()
        .map(|(player, hand)| {
            format!(
                "{}: {}",
                player_name(player),
                hand
                    .to_vec_sorted(observation.game_type)
                    .iter()
                    .map(|card| card.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    let str = format!("{}\n\n{}\n", _table
        .display()
        .unwrap()
        .to_string(),
        hands
    );

    return str;
}

#[cfg(test)]
mod tests {
    use crate::action::action::FdoAction::{
        AnnouncementBlack, AnnouncementNo30, AnnouncementNo60, AnnouncementNo90, NoAnnouncement,
        ReservationClubsSolo, ReservationDiamondsSolo, ReservationHealthy, ReservationHeartsSolo,
        ReservationJacksSolo, ReservationQueensSolo, ReservationSpadesSolo,
        ReservationTrumplessSolo,
    };
    use crate::action::allowed_actions::FdoAllowedActions;
    use crate::announcement::announcement::FdoAnnouncement::ReContra;
    use crate::announcement::announcement::{FdoAnnouncement, FdoAnnouncementOccurrence};
    use crate::basic::phase::FdoPhase;
    use crate::basic::phase::FdoPhase::Reservation;
    use crate::card::cards::FdoCard;
    use crate::card::cards::FdoCard::{
        ClubAce, ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondJack,
        DiamondKing, DiamondNine, DiamondQueen, DiamondTen, HeartAce, HeartJack, HeartKing,
        HeartNine, HeartQueen, HeartTen, SpadeAce, SpadeJack, SpadeKing, SpadeNine, SpadeQueen,
        SpadeTen,
    };
    use crate::display::display::display_game;
    use crate::hand::hand::FdoHand;
    use crate::observation::observation::FdoObservation;
    use crate::player::player::FdoPlayer;
    use crate::player::player_set::FdoPlayerSet;
    use crate::reservation::reservation::{FdoReservation, FdoVisibleReservation};
    use crate::reservation::reservation_round::FdoReservationRound;
    use crate::stats::stats::FdoEndOfGameStats;
    use crate::trick::trick::FdoTrick;
    use crate::util::po_arr::PlayerOrientedArr;
    use crate::util::po_vec::PlayerOrientedVec;
    use crate::util::po_zero_arr::PlayerZeroOrientedArr;

    #[test]
    fn test_display_finished_game() {
        let obs = FdoObservation {
            phase: FdoPhase::Finished,
            observing_player: FdoPlayer::BOTTOM,
            current_player: None,
            allowed_actions_current_player: FdoAllowedActions::from_vec(vec![]),
            game_starting_player: FdoPlayer::BOTTOM,
            wedding_player_if_wedding_announced: None,
            tricks: heapless::Vec::from_slice(&[
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::ClubQueen,
                        FdoCard::DiamondJack,
                        FdoCard::ClubJack,
                        FdoCard::HeartJack,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::DiamondJack,
                        FdoCard::ClubQueen,
                        FdoCard::DiamondNine,
                        FdoCard::SpadeQueen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::SpadeKing,
                        FdoCard::SpadeTen,
                        FdoCard::SpadeAce,
                        FdoCard::SpadeAce,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::RIGHT,
                    vec![
                        FdoCard::HeartNine,
                        FdoCard::HeartAce,
                        FdoCard::HeartNine,
                        FdoCard::HeartKing,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::HeartTen,
                        FdoCard::SpadeQueen,
                        FdoCard::DiamondQueen,
                        FdoCard::HeartTen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::DiamondTen,
                        FdoCard::DiamondNine,
                        FdoCard::HeartJack,
                        FdoCard::DiamondTen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::TOP,
                    vec![
                        FdoCard::DiamondKing,
                        FdoCard::SpadeJack,
                        FdoCard::HeartQueen,
                        FdoCard::ClubAce,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::BOTTOM,
                    vec![
                        FdoCard::SpadeJack,
                        FdoCard::DiamondQueen,
                        FdoCard::DiamondKing,
                        FdoCard::SpadeKing,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::HeartQueen,
                        FdoCard::ClubTen,
                        FdoCard::HeartAce,
                        FdoCard::ClubKing,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::DiamondAce,
                        FdoCard::HeartKing,
                        FdoCard::ClubNine,
                        FdoCard::ClubAce,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::ClubJack,
                        FdoCard::SpadeNine,
                        FdoCard::ClubNine,
                        FdoCard::SpadeTen,
                    ],
                ),
                FdoTrick::existing(
                    FdoPlayer::LEFT,
                    vec![
                        FdoCard::DiamondAce,
                        FdoCard::ClubKing,
                        FdoCard::ClubTen,
                        FdoCard::SpadeNine,
                    ],
                ),
            ])
            .unwrap(),
            visible_reservations: PlayerOrientedArr::from_full(
                FdoPlayer::BOTTOM,
                [
                    FdoVisibleReservation::Healthy,
                    FdoVisibleReservation::Healthy,
                    FdoVisibleReservation::Healthy,
                    FdoVisibleReservation::Healthy,
                ],
            ),
            announcements: heapless::Vec::from_slice(&[
                FdoAnnouncementOccurrence {
                    player: FdoPlayer::BOTTOM,
                    card_index: 5,
                    announcement: FdoAnnouncement::No60,
                },
                FdoAnnouncementOccurrence {
                    player: FdoPlayer::RIGHT,
                    card_index: 16,
                    announcement: FdoAnnouncement::No90,
                },
                FdoAnnouncementOccurrence {
                    player: FdoPlayer::RIGHT,
                    card_index: 7,
                    announcement: FdoAnnouncement::No60,
                },
            ])
            .unwrap(),
            player_eyes: PlayerZeroOrientedArr::from_full([70, 112, 22, 36]),
            observing_player_hand: FdoHand::from_vec(vec![]),
            finished_stats: Some(FdoEndOfGameStats {
                re_players: FdoPlayerSet::from_vec(vec![FdoPlayer::BOTTOM, FdoPlayer::LEFT]),
                is_solo: false,
                player_eyes: PlayerZeroOrientedArr::from_full([70, 112, 22, 36]),
                re_eyes: 182,
                kontra_eyes: 58,
                re_points: 3,
                kontra_points: -3,
                player_points: PlayerZeroOrientedArr::from_full([3, -3, -3, 3]),
                basic_winning_point_details: None,
                basic_draw_points_details: None,
                additional_points_details: None,
            }),
            phi_re_players: Some(FdoPlayerSet::from_vec(vec![
                FdoPlayer::BOTTOM,
                FdoPlayer::LEFT,
            ])),
            phi_real_reservations: FdoReservationRound::existing(
                FdoPlayer::BOTTOM,
                vec![
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                ],
            ),
            phi_real_hands: PlayerZeroOrientedArr::from_full([
                FdoHand::from_vec(vec![]),
                FdoHand::from_vec(vec![]),
                FdoHand::from_vec(vec![]),
                FdoHand::from_vec(vec![]),
            ]),
            phi_team_eyes: [15, 15],
            game_type: None,

            re_lowest_announcement: None,
            contra_lowest_announcement: None,
        };

        println!("{}", display_game(obs));
    }

    #[test]
    fn test_fehler_state() {
        let obs = FdoObservation {
            phase: Reservation,
            observing_player: FdoPlayer::RIGHT,
            current_player: Some(FdoPlayer::RIGHT),
            allowed_actions_current_player: FdoAllowedActions::from_vec(vec![
                ReservationHealthy,
                ReservationDiamondsSolo,
                ReservationHeartsSolo,
                ReservationSpadesSolo,
                ReservationClubsSolo,
                ReservationTrumplessSolo,
                ReservationQueensSolo,
                ReservationJacksSolo,
            ]),
            game_starting_player: FdoPlayer::RIGHT,
            wedding_player_if_wedding_announced: None,
            tricks: heapless::Vec::new(),
            visible_reservations: PlayerOrientedArr::from_full(
                FdoPlayer::RIGHT,
                [
                    FdoVisibleReservation::NoneYet,
                    FdoVisibleReservation::NoneYet,
                    FdoVisibleReservation::NoneYet,
                    FdoVisibleReservation::NoneYet,
                ],
            ),
            announcements: heapless::Vec::new(),
            player_eyes: PlayerZeroOrientedArr::from_full([0, 0, 0, 0]),
            observing_player_hand: FdoHand::from_vec(vec![
                ClubQueen,
                SpadeQueen,
                HeartQueen,
                HeartJack,
                DiamondJack,
                DiamondKing,
                ClubAce,
                ClubKing,
                SpadeTen,
                SpadeNine,
                SpadeNine,
                HeartKing,
            ]),
            finished_stats: None,
            phi_re_players: None,
            phi_real_reservations: FdoReservationRound {
                reservations: PlayerOrientedVec::empty(FdoPlayer::RIGHT),
            },
            phi_real_hands: PlayerZeroOrientedArr::from_full([
                FdoHand::from_vec(vec![
                    HeartTen,
                    ClubJack,
                    ClubJack,
                    DiamondAce,
                    DiamondNine,
                    DiamondNine,
                    ClubAce,
                    SpadeAce,
                    SpadeTen,
                    SpadeKing,
                    HeartAce,
                    HeartNine,
                ]),
                FdoHand::from_vec(vec![
                    ClubQueen,
                    DiamondQueen,
                    HeartJack,
                    DiamondAce,
                    DiamondTen,
                    ClubTen,
                    ClubTen,
                    ClubKing,
                    ClubNine,
                    SpadeKing,
                    HeartAce,
                    HeartNine,
                ]),
                FdoHand::from_vec(vec![
                    HeartTen,
                    SpadeQueen,
                    HeartQueen,
                    DiamondQueen,
                    SpadeJack,
                    SpadeJack,
                    DiamondJack,
                    DiamondTen,
                    DiamondKing,
                    ClubNine,
                    SpadeAce,
                    HeartKing,
                ]),
                FdoHand::from_vec(vec![
                    ClubQueen,
                    SpadeQueen,
                    HeartQueen,
                    HeartJack,
                    DiamondJack,
                    DiamondKing,
                    ClubAce,
                    ClubKing,
                    SpadeTen,
                    SpadeNine,
                    SpadeNine,
                    HeartKing,
                ]),
            ]),
            phi_team_eyes: [0, 0],
            game_type: None,
            re_lowest_announcement: None,
            contra_lowest_announcement: None,
        };

        println!("{}", display_game(obs));
    }

    #[test]
    fn test_weiter() {
        let obs = FdoObservation {
            game_type: None,
            phase: FdoPhase::Announcement,
            observing_player: FdoPlayer::BOTTOM,
            current_player: Some(FdoPlayer::BOTTOM),
            allowed_actions_current_player: FdoAllowedActions::from_vec(vec![
                AnnouncementNo90,
                AnnouncementNo60,
                AnnouncementNo30,
                AnnouncementBlack,
                NoAnnouncement,
            ]),
            game_starting_player: FdoPlayer::BOTTOM,
            wedding_player_if_wedding_announced: None,
            tricks: heapless::Vec::from_slice(&[FdoTrick::existing(
                FdoPlayer::BOTTOM,
                vec![HeartTen],
            )]).unwrap(),
            visible_reservations: PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
                FdoVisibleReservation::Healthy,
                FdoVisibleReservation::Healthy,
                FdoVisibleReservation::Healthy,
                FdoVisibleReservation::Healthy,
            ]),
            announcements: heapless::Vec::from_slice(&[FdoAnnouncementOccurrence {
                card_index: 1,
                player: FdoPlayer::RIGHT,
                announcement: ReContra,
            }])
            .unwrap(),
            player_eyes: PlayerZeroOrientedArr::from_full([0, 0, 0, 0]),
            observing_player_hand: FdoHand::from_vec(vec![
                ClubQueen,
                SpadeQueen,
                HeartQueen,
                ClubJack,
                HeartJack,
                DiamondAce,
                DiamondKing,
                DiamondKing,
                DiamondNine,
                SpadeTen,
                SpadeNine,
            ]),
            finished_stats: None,
            phi_re_players: Some(FdoPlayerSet::from_vec(vec![
                FdoPlayer::BOTTOM,
                FdoPlayer::RIGHT,
            ])),
            phi_real_reservations: FdoReservationRound {
                reservations: PlayerOrientedVec::from_full(FdoPlayer::BOTTOM, vec![
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                    FdoReservation::Healthy,
                ])
            },
            phi_real_hands: PlayerZeroOrientedArr::from_full([
                FdoHand::from_vec(vec![
                    ClubQueen,
                    SpadeQueen,
                    HeartQueen,
                    ClubJack,
                    HeartJack,
                    DiamondAce,
                    DiamondKing,
                    DiamondKing,
                    DiamondNine,
                    SpadeTen,
                    SpadeNine,
                ]),
                FdoHand::from_vec(vec![
                    SpadeQueen,
                    DiamondQueen,
                    SpadeJack,
                    SpadeJack,
                    HeartJack,
                    DiamondJack,
                    ClubAce,
                    ClubTen,
                    ClubNine,
                    SpadeAce,
                    HeartKing,
                    HeartNine,
                ]),
                FdoHand::from_vec(vec![
                    HeartQueen,
                    DiamondQueen,
                    ClubJack,
                    ClubAce,
                    ClubKing,
                    ClubNine,
                    SpadeAce,
                    SpadeKing,
                    SpadeKing,
                    SpadeNine,
                    HeartAce,
                    HeartNine,
                ]),
                FdoHand::from_vec(vec![
                    HeartTen,
                    ClubQueen,
                    DiamondJack,
                    DiamondAce,
                    DiamondTen,
                    DiamondTen,
                    DiamondNine,
                    ClubTen,
                    ClubKing,
                    SpadeTen,
                    HeartAce,
                    HeartKing,
                ]),
            ]),
            phi_team_eyes: [0, 0],
            re_lowest_announcement: None,
            contra_lowest_announcement: None,
        };

        println!("{}", display_game(obs));
    }
}
