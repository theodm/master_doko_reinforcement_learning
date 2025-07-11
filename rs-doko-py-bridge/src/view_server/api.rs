use std::cmp::max;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rs_full_doko::action::action::FdoAction;
use rs_full_doko::announcement::announcement::FdoAnnouncement;
use rs_full_doko::basic::phase::FdoPhase;
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::observation::observation::FdoObservation;
use rs_full_doko::player::player::FdoPlayer;
use rs_full_doko::player::player_set::FdoPlayerSet;
use rs_full_doko::reservation::reservation::FdoReservation;
use rs_full_doko::state::state::FdoState;
use rs_full_doko::trick::trick::FdoTrick;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Rank {
    #[serde(rename = "9")]
    Nine,
    #[serde(rename = "10")]
    Ten,
    J,
    Q,
    K,
    A,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApGameCard {
    pub suit: Suit,
    pub rank: Rank,
}

impl ApGameCard {
    fn create_from(card: FdoCard) -> ApGameCard {
        match card {
            FdoCard::DiamondNine => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::Nine,
            },
            FdoCard::DiamondTen => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::Ten,
            },
            FdoCard::DiamondJack => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::J,
            },
            FdoCard::DiamondQueen => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::Q,
            },
            FdoCard::DiamondKing => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::K,
            },
            FdoCard::DiamondAce => ApGameCard {
                suit: Suit::Diamonds,
                rank: Rank::A,
            },
            FdoCard::HeartNine => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::Nine,
            },
            FdoCard::HeartTen => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::Ten,
            },
            FdoCard::HeartJack => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::J,
            },
            FdoCard::HeartQueen => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::Q,
            },
            FdoCard::HeartKing => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::K,
            },
            FdoCard::HeartAce => ApGameCard {
                suit: Suit::Hearts,
                rank: Rank::A,
            },
            FdoCard::ClubNine => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::Nine,
            },
            FdoCard::ClubTen => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::Ten,
            },
            FdoCard::ClubJack => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::J,
            },
            FdoCard::ClubQueen => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::Q,
            },
            FdoCard::ClubKing => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::K,
            },
            FdoCard::ClubAce => ApGameCard {
                suit: Suit::Clubs,
                rank: Rank::A,
            },
            FdoCard::SpadeNine => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::Nine,
            },
            FdoCard::SpadeTen => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::Ten,
            },
            FdoCard::SpadeJack => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::J,
            },
            FdoCard::SpadeQueen => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::Q,
            },
            FdoCard::SpadeKing => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::K,
            },
            FdoCard::SpadeAce => ApGameCard {
                suit: Suit::Spades,
                rank: Rank::A,
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApGuessedCard {
    pub card: ApGameCard,
    pub probability: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApReservationType {
    Wedding,
    DiamondsSolo,
    HeartsSolo,
    SpadesSolo,
    ClubsSolo,
    TrumplessSolo,
    QueensSolo,
    JacksSolo,
    NoReservation,
}

impl ApReservationType {
    fn create_from(reservation: &FdoReservation) -> ApReservationType {
        match reservation {
            FdoReservation::Healthy => ApReservationType::NoReservation,
            FdoReservation::Wedding => ApReservationType::Wedding,
            FdoReservation::DiamondsSolo => ApReservationType::DiamondsSolo,
            FdoReservation::HeartsSolo => ApReservationType::HeartsSolo,
            FdoReservation::SpadesSolo => ApReservationType::SpadesSolo,
            FdoReservation::ClubsSolo => ApReservationType::ClubsSolo,
            FdoReservation::QueensSolo => ApReservationType::QueensSolo,
            FdoReservation::JacksSolo => ApReservationType::JacksSolo,
            FdoReservation::TrumplessSolo => ApReservationType::TrumplessSolo,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApTrick {
    pub startingIndex: u8,
    pub cards: Vec<ApGameCard>,
    pub startingPlayer: String,
    pub winner: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum AnnouncementAction {
    Re,
    Kontra,
    ReUnder90,
    ReUnder60,
    ReUnder30,
    ReBlack,
    KontraUnder90,
    KontraUnder60,
    KontraUnder30,
    KontraBlack,
    NoAnnouncement,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApAnnouncement {
    pub playerName: String,
    pub cardIndex: usize,
    pub announcement: AnnouncementAction,
}

impl AnnouncementAction {
    fn create_from(ano: &FdoAnnouncement, player_played: FdoPlayer, re_players: Option<FdoPlayerSet>) -> AnnouncementAction {
        let re_players = re_players
            .unwrap();

        match ano {
            FdoAnnouncement::ReContra => {
                if re_players.contains(player_played) {
                    AnnouncementAction::Re
                } else {
                    AnnouncementAction::Kontra
                }
            }
            FdoAnnouncement::No90 => {
                if re_players.contains(player_played) {
                    AnnouncementAction::ReUnder90
                } else {
                    AnnouncementAction::KontraUnder90
                }
            }
            FdoAnnouncement::No60 => {
                if re_players.contains(player_played) {
                    AnnouncementAction::ReUnder60
                } else {
                    AnnouncementAction::KontraUnder60
                }
            }
            FdoAnnouncement::No30 => {
                if re_players.contains(player_played) {
                    AnnouncementAction::ReUnder30
                } else {
                    AnnouncementAction::KontraUnder30
                }
            }
            FdoAnnouncement::Black => {
                if re_players.contains(player_played) {
                    AnnouncementAction::ReBlack
                } else {
                    AnnouncementAction::KontraBlack
                }
            }
            FdoAnnouncement::CounterReContra => {
                if re_players.contains(player_played) {
                    AnnouncementAction::Re
                } else {
                    AnnouncementAction::Kontra
                }
            }
            FdoAnnouncement::NoAnnouncement => {
                AnnouncementAction::NoAnnouncement
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApAnnouncements {
    pub announcements: Vec<ApAnnouncement>,
}

impl ApAnnouncements {
    fn create_from(
        state: &FdoState,
        observation: &FdoObservation
    ) -> ApAnnouncements {
        let mut result = vec![];
        for ano in observation.announcements.clone() {
            let player_name = player_index_to_name(ano.player as usize);
            let card_index = ano.card_index;
            let announcement = AnnouncementAction::create_from(
                &ano.announcement,
                ano.player,
                observation.phi_re_players
            );

            result.push(ApAnnouncement {
                playerName: player_name,
                cardIndex: card_index,
                announcement,
            });
        }

        ApAnnouncements {
            announcements: result,
        }
    }
}
impl GamePhase {
    pub fn create_from(phase: &FdoPhase) -> GamePhase {
        match phase {
            FdoPhase::Reservation => GamePhase::Reservations,
            FdoPhase::Announcement => GamePhase::Announcements,
            FdoPhase::PlayCard => GamePhase::Playing,
            FdoPhase::Finished => GamePhase::Finished,
        }
    }
}
impl ApTrick {
    pub fn create_from(
        trick: &FdoTrick,
        index: usize
    ) -> ApTrick {
        let cards: Vec<ApGameCard> = trick.cards
            .iter()
            .map(|card| ApGameCard::create_from(*card))
            .collect();

        let starting_player = player_index_to_name(trick.starting_player().index());
        let winner = trick.winning_player.map(|p| player_index_to_name(p.index()));

        ApTrick {
            startingIndex: index as u8,
            cards,
            startingPlayer: starting_player,
            winner,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum GamePhase {
    Reservations,
    Announcements,
    Playing,
    Finished,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApGameState {
    pub currentPlayer: Option<String>,
    pub currentPlayerActionValues: Option<HashMap<String, f32>>,
    pub playerHands: HashMap<String, Vec<ApGameCard>>,
    pub reservations: HashMap<String, ApReservationType>,
    pub announcements: ApAnnouncements,
    pub previousTricks: Vec<ApTrick>,
    pub currentTrick: Option<ApTrick>,
    pub currentPhase: GamePhase,
    pub isImportantState: bool,
    pub guessedHands: HashMap<String, Vec<ApGuessedCard>>,
    pub gameMode: String,
}

fn player_index_to_name(index: usize) -> String {
    match index {
        0 => "Bottom".to_string(),
        1 => "Left".to_string(),
        2 => "Top".to_string(),
        3 => "Right".to_string(),
        _ => panic!("Invalid player index"),
    }
}

impl ApGameState {
    pub(crate) fn create_from(
        state: &FdoState,
        observation: &FdoObservation,
        last_action: Option<FdoAction>
    ) -> ApGameState {

        let mut player_hands = HashMap::new();

        for (player, hand) in observation.phi_real_hands.iter_with_player() {
            let cards: Vec<ApGameCard> = hand
                .iter()
                .map(|card| ApGameCard::create_from(card))
                .collect();

            player_hands.insert(
                player_index_to_name(player as usize), cards);
        }
        let mut reservations = HashMap::new();

        for (player, reservation) in observation.phi_real_reservations.reservations.iter_with_player() {
            reservations.insert(
                player_index_to_name(player as usize),
                ApReservationType::create_from(reservation),

            );
        }

        // Tricks bis auf aktuellen
        let mut previous_tricks = vec![];

        for (trick_index, trick) in observation
            .tricks
            .iter()
            .enumerate()
            .take(max(observation.tricks.len() as i32 - 1i32, 0i32) as usize) {
            previous_tricks.push(
                ApTrick::create_from(trick, trick_index)
            );
        }

        let current_trick = observation.tricks.last().map(|trick| {
            ApTrick::create_from(trick, observation.tricks.len() - 1)
        });


        ApGameState {
            currentPlayer: observation.current_player.map(|p| p.to_string()),
            currentPlayerActionValues: None,
            playerHands: player_hands,
            reservations: reservations,
            announcements: ApAnnouncements::create_from(state, observation),
            previousTricks: previous_tricks,
            currentTrick: current_trick,
            currentPhase: GamePhase::create_from(&observation.phase),
            isImportantState: last_action != Some(FdoAction::NoAnnouncement),
            guessedHands: HashMap::new(),
            gameMode: "Normal".to_string() //observation.game_type.map(|game_type| game_type.to_string()).unwrap_or("Normal".to_string()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApPlayedGame {
    pub players: Vec<String>,
    pub states: Vec<ApGameState>,
}


#[cfg(test)]
mod tests {
    use rand::prelude::SmallRng;
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn test_complete_game() {
        let mut rng = SmallRng::seed_from_u64(0);

        let mut state = FdoState::new_game(&mut rng);

        let mut last_action = None;
        loop {
            let obs = state.observation_for_current_player();

            let game_state = ApGameState::create_from(&state, &obs, last_action);
            println!("{:?}", game_state);

            if obs.finished_stats.is_some() {
                break;
            }

            let action = obs
                .allowed_actions_current_player
                .random(&mut rng);

            state.play_action(action);

            last_action = Some(action);
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let test_card = ApGameCard {
            suit: Suit::Hearts,
            rank: Rank::Q,
        };

        let game_state = ApGameState {
            currentPlayer: Some("Alice".to_string()),
            currentPlayerActionValues: Some(HashMap::from([
                ("HEARTSQ".to_string(), 0.9),
                ("NO_ANNOUNCEMENT".to_string(), 0.1),
            ])),
            playerHands: HashMap::from([(
                "Alice".to_string(),
                vec![test_card.clone()],
            )]),
            reservations: HashMap::from([(
                "Alice".to_string(),
                ApReservationType::HeartsSolo,
            )]),
            announcements: ApAnnouncements {
                announcements: vec![ApAnnouncement {
                    playerName: "Alice".to_string(),
                    cardIndex: 0,
                    announcement: AnnouncementAction::Re,
                }],
            },
            previousTricks: vec![],
            currentTrick: Some(ApTrick {
                startingIndex: 0,
                cards: vec![test_card.clone()],
                startingPlayer: "Alice".to_string(),
                winner: Some("Alice".to_string()),
            }),
            currentPhase: GamePhase::Playing,
            isImportantState: true,
            guessedHands: HashMap::from([(
                "Bob".to_string(),
                vec![ApGuessedCard {
                    card: test_card.clone(),
                    probability: 0.25,
                }],
            )]),
            gameMode: "BY_COLOR".to_string(),
        };

        let full_game = ApPlayedGame {
            players: vec!["Alice".to_string(), "Bob".to_string()],
            states: vec![game_state.clone()],
        };

        let json = serde_json::to_string_pretty(&full_game).unwrap();
        println!("{}", json);

        let parsed: ApPlayedGame = serde_json::from_str(&json).unwrap();
        assert_eq!(full_game, parsed);
    }
}