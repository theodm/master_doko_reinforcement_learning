use log::debug;
use strum_macros::EnumIter;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use crate::hand::hand::{FdoHand};
use crate::matching::gather_impossible_colors::gather_impossible_colors;
use crate::matching::is_consistent::NotConsistentReason::{AlreadyDiscardedColor, HandSizeMismatch, HasClubQueenButAnnouncedKontra, HasClubQueenButSomeoneElseAnnouncedWedding, HasNoClubQueenButAnnouncedRe, NotInRemainingCards, RemainingCardsLeft, WrongReservation, WrongReservationClubQ};
use crate::observation::observation::FdoObservation;
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::reservation::reservation::{FdoReservation, FdoVisibleReservation};
use crate::state::state::FdoState;
use crate::trick::trick::FdoTrick;
use crate::util::po_arr::PlayerOrientedArr;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

macro_rules! debug_println {
    ($($arg:tt)*) => {
        // if cfg!(debug_assertions) {
        //     println!($($arg)*);
        // }
    };
}

#[derive(Debug, EnumIter, PartialEq, Eq, Clone)]
pub enum NotConsistentReason {
    HandSizeMismatch,
    NotInRemainingCards,
    RemainingCardsLeft,
    AlreadyDiscardedColor,
    HasClubQueenButSomeoneElseAnnouncedWedding,
    HasNoClubQueenButAnnouncedRe,
    HasClubQueenButAnnouncedKontra,
    WrongReservation,
    WrongReservationClubQ,
}

pub fn _is_consistent(
    _state: &FdoState,
    obs: &FdoObservation,

    assumed_hands: PlayerZeroOrientedArr<FdoHand>,
    assumed_reservations: PlayerOrientedArr<Option<FdoReservation>>,
) -> Option<NotConsistentReason> {
    let real_hands = obs.phi_real_hands;
    let game_type = obs.game_type;
    let re_players = obs.phi_re_players;
    let previous_tricks = obs.tricks.clone();
    let visible_reservations = obs.visible_reservations;
    let game_start_player = obs.game_starting_player;

    assert!(real_hands[obs.observing_player] == assumed_hands[obs.observing_player]);

    // Folgende Dinge können wir immer ableiten:
    //
    // Ein Spieler hat Trumpf nicht bekannt. -> Er hat (von den verbleibenden Karten) keinen Trumpf.
    // Ein Spieler hat Herz nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Herz.
    // Ein Spieler hat Pik nicht bekannt. -> Er hat (von den verbleibenden Karten) kein Pik.
    // Ein Spieler hat Kreuz nicht bekannt. -> Er hat (von den verbleibenden Karten) keine Kreuz.
    // Ein Spieler hat eine Hochzeit angesagt. -> Er hat beide Kreuz-Damen, alle anderen haben keine Kreuz-Dame.
    // Ein Spieler hat im Normalspiel eine Ansage als Re gemacht. -> Er hat eine Kreuz-Dame.
    // Ein Spieler hat im Normalspiel eine Ansage als Kontra gemacht. -> Er hat keine Kreuz-Dame.

    // Wenn die Anzahl der Karten in den Händen nicht übereinstimmt, sind die Hände nicht konsistent.
    for player in FdoPlayerSet::all().iter() {
        if assumed_hands[player].len() != real_hands[player].len() {
            debug_println!("Hand size mismatch for player {}", player);

            return Some(HandSizeMismatch);
        }
    }

    // Wenn jemand eine Karte auf der Hand hat, die es nicht mehr gibt, ist das Spiel nicht konsistent.
    let mut remaining_cards = real_hands[FdoPlayer::BOTTOM].clone();

    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::LEFT]);
    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::TOP]);
    remaining_cards = remaining_cards.plus_hand(real_hands[FdoPlayer::RIGHT]);

    for player in FdoPlayerSet::all().iter() {
        for card in assumed_hands[player].iter() {
            if !remaining_cards.contains(card) {
                debug_println!("Card {} is not in the remaining cards", card);

                return Some(NotInRemainingCards);
            }

            remaining_cards.remove(card);
        }
    }

    if remaining_cards.len() != 0 {
        debug_println!("Remaining cards left: {:?}", remaining_cards);

        return Some(RemainingCardsLeft);
    }

    // Wenn jemand eine Farbe hat, auf die er bereits abgeworfen hat, ist das Spiel nicht konsistent.
    if let Some(game_type) = game_type {
        // Wenn noch keine Karten gespielt wurden, können wir nichts ableiten.
        let impossible_colors = gather_impossible_colors(
            &previous_tricks,
            Some(game_type)
        );

        for player in FdoPlayerSet::all().iter() {
            let hand = assumed_hands[player];

            for ic in impossible_colors[player].iter() {
                if hand.contains_card_of_color(ic, game_type) {
                    debug_println!("Player {} has a card of color {:?} which he discarded", player, ic);
                    return Some(AlreadyDiscardedColor);
                }
            }
        }
    }

    let mut wedding_player = None;

    // Wenn jemand eine Hochzeit angesagt hat, muss er beide Kreuz-Damen haben (sofern sie noch verfügbar sind).
    for (player, reservation) in assumed_reservations.iter_with_player() {
        if *reservation == Some(FdoReservation::Wedding) {
            wedding_player = Some(player);
        }
    }

    // Es reicht aus, wenn wir überprüfen, dass die anderen Spieler keine Kreuz-Dame haben.
    if let Some(wedding_player) = wedding_player {
        for player in FdoPlayerSet::all().iter() {
            if player == wedding_player {
                continue;
            }

            if assumed_hands[player].contains(FdoCard::ClubQueen) {
                debug_println!("Player {} has a Club Queen but some one else announced a wedding", player);
                return Some(HasClubQueenButSomeoneElseAnnouncedWedding);
            }
        }
    }


    if let Some(re_players) = re_players {
        if game_type == Some(FdoGameType::Normal) {
            // Ansagen können nur gemacht sein, wenn das Spiel begonnen hat
            // und die Re-Spieler fest sind.
            let mut player_already_played_q_club = PlayerZeroOrientedArr::from_full([
                false,
                false,
                false,
                false
            ]);

            for trick in previous_tricks.iter() {
                for (player, card) in trick.iter_with_player() {
                    if *card == FdoCard::ClubQueen {
                        player_already_played_q_club[player] = true;
                    }
                }
            }

            for announcement in obs.announcements.iter() {
                // Wenn jemand im Normalspiel eine Ansage als
                // Re gemacht hat, muss er eine Kreuz-Dame haben oder
                // gehabt haben.
                if re_players.contains(announcement.player) {
                    let club_q_in_hand = assumed_hands[announcement.player]
                        .contains(FdoCard::ClubQueen);
                    let club_q_played = player_already_played_q_club[announcement.player];

                    if !club_q_in_hand && !club_q_played {
                        debug_println!("Player {} has no Club Queen but announced Re", announcement.player);
                        return Some(HasNoClubQueenButAnnouncedRe);
                    }
                } else {
                    // Wenn jemand im Normalspiel eine Ansage als
                    // Kontra gemacht hat, darf er keine Kreuz-Dame haben.
                    let club_q_in_hand = assumed_hands[announcement.player]
                        .contains(FdoCard::ClubQueen);

                    if club_q_in_hand {
                        debug_println!("Player {} has a Club Queen but announced Kontra", announcement.player);
                        return Some(HasClubQueenButAnnouncedKontra);
                    }
                }
            }
        }
    }

    // Ein Hochzeitsvorbehalt ist nur konsistent, wenn wir davon ausgehen,
    // dass der Gegner beide Kreuz-Damen hat
    let mut played_or_assumed_q_club = PlayerZeroOrientedArr::from_full([
        0,
        0,
        0,
        0
    ]);

    // Hat er die Kreuz-Dame gespielt?
    for trick in previous_tricks.iter() {
        for (player, card) in trick.iter_with_player() {
            if *card == FdoCard::ClubQueen {
                played_or_assumed_q_club[player] += 1;

                debug_println!("Player {} has played the Club Queen", player);
            }
        }
    }

    // Gehen wir davon aus, dass er die Kreuz-Dame hat?
    for player in FdoPlayerSet::all().iter() {
        let hand = assumed_hands[player];

        if hand.contains_both(FdoCard::ClubQueen) {
            played_or_assumed_q_club[player] += 2;

            // println!("Player {} has both Club Queens", i);
        } else if hand.contains(FdoCard::ClubQueen) {
            played_or_assumed_q_club[player] += 1;

            // println!("Player {} has one Club Queen", i);
        }
    }

    for player in FdoPlayerSet::all().iter() {
        // Wenn er tatsächlich keine Ansage bisher gemacht hat und
        // wir davon ausgehen, dass er eine gemacht hat ist das inkonsistent.
        if visible_reservations[player] != FdoVisibleReservation::NoneYet && assumed_reservations[player].is_none() {
            debug_println!("Player {} has not made a reservation but we assume he has", player);
            return Some(WrongReservation);
        }

        // Wenn er tatsächlich noch keine Ansage gemacht hat und
        // wir davon ausgehen, dass er eine gemacht hat ist das inkonsistent.
        if visible_reservations[player] == FdoVisibleReservation::NoneYet && assumed_reservations[player].is_some() {
            debug_println!("Player {} has made a reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        // Wenn der sichtbare Vorbehalt mit dem angenommenen Vorbehalt nicht übereinstimmt,
        // ist das Spiel inkonsistent.
        if visible_reservations[player] == FdoVisibleReservation::Healthy && assumed_reservations[player] != Some(FdoReservation::Healthy) {
            debug_println!("Player {} has a healthy reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::Wedding && assumed_reservations[player] != Some(FdoReservation::Wedding) {
            debug_println!("Player {} has a wedding reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::DiamondsSolo && assumed_reservations[player] != Some(FdoReservation::DiamondsSolo) {
            debug_println!("Player {} has a diamonds solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::HeartsSolo && assumed_reservations[player] != Some(FdoReservation::HeartsSolo) {
            debug_println!("Player {} has a hearts solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::SpadesSolo && assumed_reservations[player] != Some(FdoReservation::SpadesSolo) {
            debug_println!("Player {} has a spades solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::ClubsSolo && assumed_reservations[player] != Some(FdoReservation::ClubsSolo) {
            debug_println!("Player {} has a clubs solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::TrumplessSolo && assumed_reservations[player] != Some(FdoReservation::TrumplessSolo) {
            debug_println!("Player {} has a trumpless solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::JacksSolo && assumed_reservations[player] != Some(FdoReservation::JacksSolo) {
            debug_println!("Player {} has a jacks solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        if visible_reservations[player] == FdoVisibleReservation::QueensSolo && assumed_reservations[player] != Some(FdoReservation::QueensSolo) {
            debug_println!("Player {} has a queens solo reservation but we assume he has not", player);
            return Some(WrongReservation);
        }

        // Wenn jemand eine Hochzeit angesagt hat, muss er beide Kreuz-Damen haben.
        if let Some(assumed_reservation) = assumed_reservations[player] {
            if assumed_reservation == FdoReservation::Wedding {
                if played_or_assumed_q_club[player] < 2 {
                    debug_println!("i: {}", player);
                    debug_println!("game_start_player: {:?}", game_start_player);
                    debug_println!("visible_reservations: {:?}", visible_reservations);
                    debug_println!("assumed_reservations: {:?}", assumed_reservations);
                    debug_println!("played_or_assumed_q_club: {:?}", played_or_assumed_q_club);
                    debug_println!("assumed_hands: {:?}", assumed_hands);
                    debug_println!("Player {} has announced a wedding but does not have both Club Queens", player);

                    return Some(WrongReservationClubQ);
                }
            }
        }
    }

    None
}

pub fn is_consistent(
    _state: &FdoState,
    obs: &FdoObservation,

    assumed_hands: PlayerZeroOrientedArr<FdoHand>,
    assumed_reservations: PlayerOrientedArr<Option<FdoReservation>>,
) -> bool {
    return _is_consistent(_state, obs, assumed_hands, assumed_reservations).is_none();
}

#[cfg(test)]
mod tests {
    use crate::action::action::FdoAction;
    use crate::card::cards::FdoCard::{ClubAce, ClubJack, ClubKing, ClubNine, ClubQueen, ClubTen, DiamondAce, DiamondJack, DiamondKing, DiamondNine, DiamondQueen, DiamondTen, HeartAce, HeartJack, HeartKing, HeartNine, HeartQueen, HeartTen, SpadeAce, SpadeJack, SpadeKing, SpadeNine, SpadeQueen, SpadeTen};
    use crate::hand::hand::FdoHand;
    use crate::player::player::FdoPlayer;
    use crate::reservation::reservation::FdoReservation;
    use crate::state::state::FdoState;
    use crate::util::po_arr::PlayerOrientedArr;
    use crate::util::po_vec::PlayerOrientedVec;
    use crate::util::po_zero_arr::PlayerZeroOrientedArr;

    #[test]
    pub fn test_w() {
        // https://www.online-doppelkopf.com/spiele/99.585.453
        let mut state = FdoState::new_game_from_hand_and_start_player(
            PlayerZeroOrientedArr::from_full([
                // Bottom
                FdoHand::from_vec(vec![HeartTen, DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
                // Left
                FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
                // Top
                FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
                // Right
                FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
            ]),
            FdoPlayer::BOTTOM,
        );

        state.play_action(FdoAction::ReservationHealthy);
        state.play_action(FdoAction::ReservationHealthy);
        state.play_action(FdoAction::ReservationHealthy);

        // Wir testen ob die tatsächliche Verteilung konsistent ist. (Sollte sie ja immer sein)
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![HeartTen, DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Wir testen, ob eine Karte die insgesamt 3 mal vorkommen kann.
        // Das sollte nicht möglich sein.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // DiamondAce -> HeartTen
            FdoHand::from_vec(vec![HeartTen, HeartTen, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        // Wir testen, ob eine Karte zu wenig als nicht konsistent erkannt wird.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        state.play_action(FdoAction::ReservationHealthy);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);

        state.play_action(FdoAction::CardHeartTen);

        // Wir testen nochmal ob das reale Spiel konsistent ist.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Wir testen, ob wir die gespielte Herz 10 auch niemanden nun zuordnen.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            // DiamondAce -> HeartTen
            FdoHand::from_vec(vec![HeartTen, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        // Wir testen einmal ob ein komplettes Tauschen noch konsistent ist.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top (mit Right getauscht)
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
            // Right (mit Top getauscht)
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Wir testen ob einzelne Karten getauscht werden können.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            // DiamondAce -> ClubJack
            FdoHand::from_vec(vec![ClubJack, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top (ClubJack -> DiamondAce)
            FdoHand::from_vec(vec![DiamondAce, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Wir testen ob einzelne Kreuz-Damen getauscht werden können.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom (DiamondNine -> ClubQueen)
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, ClubQueen, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![DiamondJack, SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right (ClubQueen -> DiamondNine)
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, DiamondNine, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::AnnouncementReContra);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);

        state.play_action(FdoAction::CardDiamondJack);

        // Spieler 4 hat Re angesagt, ihm muss nun immer eine Kreuz-Dame zugeordnet sein.

        // Wir testen ob das richtige Spiel konsistent ist.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            // DiamondJack -> Fehlt
            FdoHand::from_vec(vec![SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Es ist inkonsistent, wenn wir dem rechten Spieler die Kreuz-Dame wegnehmen.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom (DiamondNine -> ClubQueen)
            // HeartTen -> Fehlt
            FdoHand::from_vec(vec![DiamondAce, SpadeTen, ClubQueen, ClubQueen, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            // DiamondJack -> Fehlt
            FdoHand::from_vec(vec![SpadeQueen, HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![ClubJack, DiamondQueen, HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right (ClubQueen -> DiamondNine)
            FdoHand::from_vec(vec![DiamondAce, HeartTen, HeartAce, DiamondTen, DiamondNine, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);

        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardClubJack);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardDiamondAce);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardDiamondAce);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardSpadeQueen);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardDiamondQueen);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardHeartTen);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardHeartAce);

        state.play_action(FdoAction::AnnouncementNo90);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardSpadeTen);

        // So jetzt ist bekannt, dass Spieler 1 und 4 als Re zusammenspielen.

        // Wir testen ob das richtige Spiel konsistent ist.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![DiamondNine, ClubQueen, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![HeartNine, HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![DiamondTen, ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Als Spieler 2 müssen wir davon ausgehen, dass Spieler 1 und 4 die Kreuz-Damen haben
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![DiamondNine, ClubQueen, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![HeartNine, HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top (HeartQueen -> ClubQueen)
            FdoHand::from_vec(vec![HeartNine, ClubQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right (ClubQueen -> HeartQueen)
            FdoHand::from_vec(vec![DiamondTen, HeartQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardHeartNine);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardHeartNine);

        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardDiamondTen);

        state.play_action(FdoAction::AnnouncementNo60);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::NoAnnouncement);
        state.play_action(FdoAction::CardClubQueen);

        // Weiterhin bekannt, dass Spieler 1 und 4 die Kreuz-Damen haben.
        // Einziger Unterschied: Spieler 1 hat seine jetzt schon gespielt.

        // Wir testen ob das richtige Spiel konsistent ist.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![DiamondNine, SpadeQueen, DiamondKing, DiamondKing, HeartJack, HeartQueen, ClubJack, SpadeNine]),
            // Left
            FdoHand::from_vec(vec![HeartJack, SpadeJack, SpadeJack, DiamondQueen, ClubAce, ClubNine, HeartKing, ClubTen, SpadeAce]),
            // Top
            FdoHand::from_vec(vec![HeartQueen, SpadeNine, SpadeKing, ClubKing, ClubAce, ClubNine, SpadeKing, SpadeAce, HeartAce]),
            // Right
            FdoHand::from_vec(vec![ClubQueen, DiamondNine, DiamondJack, ClubKing, DiamondTen, SpadeTen, ClubTen, HeartKing]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);
    }

    #[test]
    pub fn test_w2() {
        // Hochzeit
        // https://www.online-doppelkopf.com/spiele/100.445.789
        let mut state = FdoState::new_game_from_hand_and_start_player(
            PlayerZeroOrientedArr::from_full([
                // Bottom
                FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
                // Left
                FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, ClubQueen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, ClubQueen, HeartTen]),
                // Top
                FdoHand::from_vec(vec![DiamondTen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
                // Right
                FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
            ]),
            FdoPlayer::BOTTOM,
        );

        // Am Anfang können die Kreuz-Damen noch beliebig verteilt sein.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left (ClubQueen -> DiamondTen) (ClubQueen -> SpadeKing)
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, DiamondTen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, SpadeKing, HeartTen]),
            // Top
            FdoHand::from_vec(vec![ClubQueen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![ClubQueen, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            None,
            None,
            None,
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        state.play_action(FdoAction::ReservationHealthy);
        state.play_action(FdoAction::ReservationWedding);
        state.play_action(FdoAction::ReservationHealthy);

        // Auch nach der Asage können die Kreuz-Damen woanders liegen. (denn der Vorbehalt muss ja noch offenbart werden)
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![ClubQueen, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left (ClubQueen -> DiamondTen) (ClubQueen -> SpadeAce)
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, DiamondTen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, SpadeAce, HeartTen]),
            // Top
            FdoHand::from_vec(vec![ClubQueen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        // Wenn wir aber davon ausgehen, kann der Vorbehalt ein beliebiger aber keine Hochzeit sein.
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::TrumplessSolo),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Hier ist inkonsistent, da wir nicht von einer Hochzeit ausgehen können, wenn wir verschiedenen Spielern die Kreuz-Damen zuordnen.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![ClubQueen, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left (ClubQueen -> DiamondTen) (ClubQueen -> SpadeAce)
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, DiamondTen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, SpadeAce, HeartTen]),
            // Top
            FdoHand::from_vec(vec![ClubQueen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        // Wenn wir aber davon ausgehen, kann der Vorbehalt ein beliebiger aber keine Hochzeit sein.
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Wedding),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);
        // Wenn wir aber wieder davon ausgehen, dass beide Kreuz-Damen bei Spieler 2 liegen, dann können wir
        // davon ausgehen, dass er eine Hochzeit angesagt hat.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, ClubQueen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, ClubQueen, HeartTen]),
            // Top
            FdoHand::from_vec(vec![DiamondTen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Wedding),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        // Wir müssen aber nicht:
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, ClubQueen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, ClubQueen, HeartTen]),
            // Top
            FdoHand::from_vec(vec![DiamondTen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::ClubsSolo),
            Some(FdoReservation::Healthy),
            None,
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), true);

        state.play_action(FdoAction::ReservationHealthy);

        // Nun wurde die Hochzeit aber aufgedeckt, jetzt können wir die Kreuz-Damen den anderen nicht mehr zuordnen
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left (ClubQueen -> DiamondTen) (ClubQueen -> SpadeKing)
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, DiamondTen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, SpadeKing, HeartTen]),
            // Top
            FdoHand::from_vec(vec![ClubQueen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![ClubQueen, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Wedding),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy),
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        // Wir können auch nichts anderes als eine Hochzeit annehmen.
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, ClubQueen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, ClubQueen, HeartTen]),
            // Top
            FdoHand::from_vec(vec![DiamondTen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::Healthy),
            Some(FdoReservation::ClubsSolo),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy)
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

        // Wir können einem Gesunden auch nichts anderes unterschieden
        let hands = PlayerZeroOrientedArr::from_full([
            // Bottom
            FdoHand::from_vec(vec![SpadeAce, DiamondNine, ClubNine, SpadeQueen, DiamondQueen, ClubKing, SpadeNine, SpadeKing, SpadeTen, SpadeTen, ClubKing, ClubTen]),
            // Left
            FdoHand::from_vec(vec![SpadeNine, HeartKing, ClubNine, ClubQueen, DiamondJack, DiamondAce, DiamondAce, SpadeJack, DiamondQueen, HeartQueen, ClubQueen, HeartTen]),
            // Top
            FdoHand::from_vec(vec![DiamondTen, HeartAce, ClubAce, DiamondKing, DiamondJack, ClubTen, HeartKing, SpadeJack, HeartQueen, HeartAce, ClubJack, SpadeQueen]),
            // Right
            FdoHand::from_vec(vec![SpadeKing, HeartNine, ClubAce, DiamondNine, DiamondTen, HeartJack, HeartNine, DiamondKing, HeartTen, SpadeAce, HeartJack, ClubJack]),
        ]);
        let reservations = PlayerOrientedArr::from_full(FdoPlayer::BOTTOM, [
            Some(FdoReservation::ClubsSolo),
            Some(FdoReservation::Wedding),
            Some(FdoReservation::Healthy),
            Some(FdoReservation::Healthy)
        ]);
        assert_eq!(super::is_consistent(&state, &state.observation_for_current_player(), hands, reservations), false);

    }
}