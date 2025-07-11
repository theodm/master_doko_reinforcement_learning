use crate::basic::color::FdoColor;
use crate::card::card_color_masks::get_color_masks_for_game_type;
use crate::card::cards::FdoCard;
use crate::game_type::game_type::FdoGameType;
use rand::prelude::{SliceRandom, SmallRng};
use rs_game_utils::bit_flag::Bitflag;
use std::fmt::{Debug, Display, Formatter};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, IntoEnumIterator};
use crate::player::player::FdoPlayer;
use crate::player::player_set::FdoPlayerSet;
use crate::util::po_zero_arr::PlayerZeroOrientedArr;

/// Die Hand eines Spielers.
///
/// Wird als Bitflag gespeichert, welches die Werte von DoCard zweimal für beide
/// Vorkommen enthält. Einmal an der Position 0 und einmal an der Position 24 (SECOND_CARD_SHIFT_OFFSET).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
// Es gibt von jeder Karte zwei Instanzen.
pub struct FdoHand(pub(crate) Bitflag<{ FdoCard::COUNT * 2 }>);

impl FdoHand {
    pub(crate) fn contains_card_of_color(&self, color: FdoColor, game_type: FdoGameType) -> bool {
        let (
            trump_color_mask,
            diamond_color_mask,
            hearts_color_mask,
            spades_color_mask,
            clubs_color_mask,
        ) = get_color_masks_for_game_type(game_type);

        let color_mask = match color {
            FdoColor::Diamond => diamond_color_mask,
            FdoColor::Heart => hearts_color_mask,
            FdoColor::Spade => spades_color_mask,
            FdoColor::Club => clubs_color_mask,
            FdoColor::Trump => trump_color_mask,
        };

        self.0.0 & color_mask != 0 || self.0.0 & (color_mask << SECOND_CARD_SHIFT_OFFSET) != 0
    }

    /// Entfernt alle Karten einer Farbe aus der Hand
    pub fn remove_color(&mut self, color: FdoColor, game_type: FdoGameType) {
        let (
            trump_color_mask,
            diamond_color_mask,
            hearts_color_mask,
            spades_color_mask,
            clubs_color_mask,
        ) = get_color_masks_for_game_type(game_type);

        let color_mask = match color {
            FdoColor::Diamond => diamond_color_mask,
            FdoColor::Heart => hearts_color_mask,
            FdoColor::Spade => spades_color_mask,
            FdoColor::Club => clubs_color_mask,
            FdoColor::Trump => trump_color_mask,
        };

        self.0 .0 &= !(color_mask | (color_mask << SECOND_CARD_SHIFT_OFFSET));
    }
}

impl FdoHand {
    /// Erstellt eine Vereinigung der beiden Hände. Es sei
    /// angemerkt, dass das nicht allein auf Bitflag-Ebene funktioniert,
    /// da die Karten doppelt vorhanden sind.
    pub fn plus_hand(&self, other: FdoHand) -> FdoHand {
        let mut new_hand = self.clone();

        for card in FdoCard::iter() {
            if other.contains_both(card) {
                new_hand.add(card);
                new_hand.add(card);
            } else if other.contains(card) {
                new_hand.add(card);
            }
        }

        new_hand
    }

    // ToDo: Testen
    pub fn minus_hand(
        &self,
        other: FdoHand
    ) -> FdoHand {
        let mut new_hand = self.clone();

        for card in FdoCard::iter() {
            if other.contains_both(card) {
                new_hand.remove_both(card);
            } else if other.contains(card) {
                new_hand.remove(card);
            }
        }

        new_hand
    }
}

pub const SECOND_CARD_SHIFT_OFFSET: u64 = 24;

macro_rules! duplicate_array {
    ($($elem:expr),*) => {
        [
            $($elem, $elem),*
        ]
    };
}
const AVAILABLE_CARDS: [FdoCard; 48] = duplicate_array![
    FdoCard::DiamondNine,
    FdoCard::DiamondTen,
    FdoCard::DiamondJack,
    FdoCard::DiamondQueen,
    FdoCard::DiamondKing,
    FdoCard::DiamondAce,
    FdoCard::HeartNine,
    FdoCard::HeartTen,
    FdoCard::HeartJack,
    FdoCard::HeartQueen,
    FdoCard::HeartKing,
    FdoCard::HeartAce,
    FdoCard::ClubNine,
    FdoCard::ClubTen,
    FdoCard::ClubJack,
    FdoCard::ClubQueen,
    FdoCard::ClubKing,
    FdoCard::ClubAce,
    FdoCard::SpadeNine,
    FdoCard::SpadeTen,
    FdoCard::SpadeJack,
    FdoCard::SpadeQueen,
    FdoCard::SpadeKing,
    FdoCard::SpadeAce
];

impl Display for FdoHand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cards = self.to_vec_sorted(None);

        write!(f, "[")?;
        for i in 0..cards.len() {
            write!(f, "{}", cards[i])?;
            if i < cards.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl Debug for FdoHand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cards = self.to_vec_sorted(None);

        write!(f, "FdoHand::from_vec(vec![")?;
        for i in 0..cards.len() {
            write!(f, "{:?}", cards[i])?;
            if i < cards.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "])")
    }
}

impl FdoHand {
    pub fn empty() -> FdoHand {
        FdoHand(Bitflag::new())
    }

    pub(crate) fn randomly_distributed(mut rng: &mut SmallRng) -> PlayerZeroOrientedArr<FdoHand> {
        let mut cards_to_distribute = AVAILABLE_CARDS;

        // Shuffle the available cards
        cards_to_distribute.shuffle(&mut rng);

        let mut hands = PlayerZeroOrientedArr::from_full([FdoHand::empty(); 4]);
        for player in FdoPlayerSet::all().iter() {
            for j in 0..12 {
                hands[player].add(cards_to_distribute[player.index() * 12 + j]);
            }
        }

        hands
    }

    /// Nur für Test-Zwecke.
    pub fn from_vec(cards: Vec<FdoCard>) -> FdoHand {
        let mut hand = FdoHand::empty();

        for card in cards {
            hand.add(card);
        }

        hand
    }

    /// Nur für Testzwecke!
    pub(crate) fn from_str(s: &str) -> FdoHand {
        FdoHand::from_vec(FdoCard::vec_from_str(s))
    }

    /// Gibt zurück, ob die Hand die Karte [card] mindestens einmal enthält.
    pub fn contains(&self, card: FdoCard) -> bool {
        self.0.contains(card as u64) || self.0.contains((card as u64) << SECOND_CARD_SHIFT_OFFSET)
    }

    /// Gibt zurück, ob die Hand beide Exemplare der Karte [card] enthält.
    pub fn contains_both(&self, card: FdoCard) -> bool {
        self.0.contains(card as u64) && self.0.contains((card as u64) << SECOND_CARD_SHIFT_OFFSET)
    }

    /// Fügt die Karte [card] zur Hand hinzu.
    pub fn add(&mut self, card: FdoCard) {
        debug_assert!(
            !self.contains_both(card),
            "Karte {:?} bereits doppelt in der Hand {:?}",
            card,
            self
        );

        if self.0.contains(card as u64) {
            self.0.add((card as u64) << SECOND_CARD_SHIFT_OFFSET)
        } else {
            self.0.add(card as u64)
        }
    }

    pub fn add_ignore(&mut self, card: FdoCard) {
        if !self.contains_both(card) {
            self.add(card);
        }
    }

    /// Entfernt eine Instanz der Karte [card] von der Hand.
    pub fn remove(&mut self, card: FdoCard) {
        if self.0.contains((card as u64) << SECOND_CARD_SHIFT_OFFSET) {
            self.0.remove((card as u64) << SECOND_CARD_SHIFT_OFFSET)
        } else if self.0.contains(card as u64) {
            self.0.remove(card as u64)
        } else {
            panic!("Karte {:?} nicht in Hand", card);
        }
    }

    /// Entfernt eine Instanz der Karte [card] von der Hand. Wenn die Karte nicht in der Hand ist, passiert nichts.
    pub fn remove_ignore(&mut self, card: FdoCard) {
        if self.contains(card) {
            self.remove(card);
        }
    }

    pub fn remove_both(&mut self, card: FdoCard) {
        if self.contains(card) {
            self.remove(card);

            if self.contains(card) {
                self.remove(card);
            }
        } else {
            panic!("Karte {:?} nicht in Hand", card);
        }
    }

    /// Gibt die Anzahl der Karten in der Hand zurück.
    pub fn len(&self) -> usize {
        self.0.number_of_ones() as usize
    }

    /// Gibt die Karten in der Hand als sortierten Vektor (je nach aktuellem Spieltyp zurück.) zurück. Ist
    /// noch kein Spieltyp bekannt, wird der Vektor nach der Sortierung im Normalspiel zurückgegeben.
    pub fn to_vec_sorted(&self, game_type: Option<FdoGameType>) -> heapless::Vec<FdoCard, 48> {
        macro_rules! add_cards {
        ($hand:expr, $target:expr, $($card:expr),*) => {
            $(
                unsafe {
                    if ($hand.contains_both($card)) {
                        $target.push_unchecked($card);
                        $target.push_unchecked($card);
                    } else if ($hand.contains($card)) {
                        $target.push_unchecked($card);
                    }
                }
            )*
        };
    }

        fn sorting_normal(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::HeartTen,
                FdoCard::ClubQueen,
                FdoCard::SpadeQueen,
                FdoCard::HeartQueen,
                FdoCard::DiamondQueen,
                FdoCard::ClubJack,
                FdoCard::SpadeJack,
                FdoCard::HeartJack,
                FdoCard::DiamondJack,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondNine,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeNine,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartNine
            );

            return cards;
        }

        fn sorting_heart_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::HeartTen,
                FdoCard::ClubQueen,
                FdoCard::SpadeQueen,
                FdoCard::HeartQueen,
                FdoCard::DiamondQueen,
                FdoCard::ClubJack,
                FdoCard::SpadeJack,
                FdoCard::HeartJack,
                FdoCard::DiamondJack,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartNine,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondNine
            );

            return cards;
        }

        fn sorting_spade_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::HeartTen,
                FdoCard::ClubQueen,
                FdoCard::SpadeQueen,
                FdoCard::HeartQueen,
                FdoCard::DiamondQueen,
                FdoCard::ClubJack,
                FdoCard::SpadeJack,
                FdoCard::HeartJack,
                FdoCard::DiamondJack,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeNine,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubNine,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondNine
            );

            return cards;
        }

        fn sorting_clubs_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::HeartTen,
                FdoCard::ClubQueen,
                FdoCard::SpadeQueen,
                FdoCard::HeartQueen,
                FdoCard::DiamondQueen,
                FdoCard::ClubJack,
                FdoCard::SpadeJack,
                FdoCard::HeartJack,
                FdoCard::DiamondJack,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeNine,
                FdoCard::HeartAce,
                FdoCard::HeartKing,
                FdoCard::HeartNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondNine
            );

            return cards;
        }

        fn sorting_trumpless_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubQueen,
                FdoCard::ClubJack,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeQueen,
                FdoCard::SpadeJack,
                FdoCard::SpadeNine,
                FdoCard::HeartAce,
                FdoCard::HeartTen,
                FdoCard::HeartKing,
                FdoCard::HeartQueen,
                FdoCard::HeartJack,
                FdoCard::HeartNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondQueen,
                FdoCard::DiamondJack,
                FdoCard::DiamondNine
            );

            return cards;
        }

        fn sorting_queen_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::ClubQueen,
                FdoCard::SpadeQueen,
                FdoCard::HeartQueen,
                FdoCard::DiamondQueen,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubJack,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeJack,
                FdoCard::SpadeNine,
                FdoCard::HeartAce,
                FdoCard::HeartTen,
                FdoCard::HeartKing,
                FdoCard::HeartJack,
                FdoCard::HeartNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondJack,
                FdoCard::DiamondNine
            );

            return cards;
        }

        fn sorting_jack_solo(hand: &FdoHand) -> heapless::Vec<FdoCard, 48> {
            let mut cards: heapless::Vec<FdoCard, 48> = heapless::Vec::new();

            add_cards!(
                hand,
                cards,
                FdoCard::ClubJack,
                FdoCard::SpadeJack,
                FdoCard::HeartJack,
                FdoCard::DiamondJack,
                FdoCard::ClubAce,
                FdoCard::ClubTen,
                FdoCard::ClubKing,
                FdoCard::ClubQueen,
                FdoCard::ClubNine,
                FdoCard::SpadeAce,
                FdoCard::SpadeTen,
                FdoCard::SpadeKing,
                FdoCard::SpadeQueen,
                FdoCard::SpadeNine,
                FdoCard::HeartAce,
                FdoCard::HeartTen,
                FdoCard::HeartKing,
                FdoCard::HeartQueen,
                FdoCard::HeartNine,
                FdoCard::DiamondAce,
                FdoCard::DiamondTen,
                FdoCard::DiamondKing,
                FdoCard::DiamondQueen,
                FdoCard::DiamondNine
            );

            return cards;
        }

        return match game_type {
            None => sorting_normal(self),
            Some(game_type) => match game_type {
                FdoGameType::Normal => sorting_normal(self),
                FdoGameType::Wedding => sorting_normal(self),
                FdoGameType::DiamondsSolo => sorting_normal(self),
                FdoGameType::HeartsSolo => sorting_heart_solo(self),
                FdoGameType::SpadesSolo => sorting_spade_solo(self),
                FdoGameType::ClubsSolo => sorting_clubs_solo(self),
                FdoGameType::TrumplessSolo => sorting_trumpless_solo(self),
                FdoGameType::QueensSolo => sorting_queen_solo(self),
                FdoGameType::JacksSolo => sorting_jack_solo(self),
            },
        };
    }
}

mod tests {
    use super::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_distribute_cards() {
        let mut rng = SmallRng::seed_from_u64(42);

        let hands = FdoHand::randomly_distributed(&mut rng);

        // print hands in binary
        for player in FdoPlayerSet::all().iter() {
            println!("{:024b}", hands[player].0 .0);
        }

        assert_eq!(
            hands[FdoPlayer::BOTTOM].0 .0,
            0b0000000010000000000000110100100110000110110001
        );
        assert_eq!(
            hands[FdoPlayer::LEFT].0 .0,
            0b0001000000100001000000000001010101100001100101
        );
        assert_eq!(
            hands[FdoPlayer::TOP].0 .0,
            0b1000000000000000001000011110110000000000011110
        );
        assert_eq!(
            hands[FdoPlayer::RIGHT].0 .0,
            0b0000001000011000000000100010001001011110000010
        );
    }

    #[test]
    fn test_empty_hand() {
        let hand = FdoHand::empty();
        assert_eq!(
            hand.len(),
            0,
            "Eine leere Hand sollte keine Karten enthalten."
        );
    }

    #[test]
    fn test_add_and_contains() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);
        assert!(
            hand.contains(card),
            "Die Karte sollte nach dem Hinzufügen enthalten sein."
        );
        assert!(
            !hand.contains_both(card),
            "Die Karte sollte nicht doppelt enthalten sein."
        );
    }

    #[test]
    fn test_add_same_card_twice() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);
        hand.add(card);

        assert!(hand.contains(card), "Die Karte sollte enthalten sein.");
        assert!(
            hand.contains_both(card),
            "Die Karte sollte doppelt enthalten sein."
        );
        assert_eq!(
            hand.len(),
            2,
            "Die Hand sollte genau zwei Karten enthalten."
        );
    }

    #[test]
    #[should_panic]
    fn test_add_more_than_two_instances() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);
        hand.add(card);
        hand.add(card); // Sollte einen Panic auslösen
    }

    #[test]
    fn test_remove_card() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);
        hand.add(card);

        hand.remove(card);
        assert!(
            hand.contains(card),
            "Eine Instanz der Karte sollte noch vorhanden sein."
        );
        assert!(
            !hand.contains_both(card),
            "Die Karte sollte nicht doppelt enthalten sein."
        );
        assert_eq!(hand.len(), 1, "Die Hand sollte genau eine Karte enthalten.");

        hand.remove(card);
        assert!(
            !hand.contains(card),
            "Die Karte sollte nach dem Entfernen nicht mehr enthalten sein."
        );
        assert_eq!(hand.len(), 0, "Die Hand sollte leer sein.");
    }

    #[test]
    fn test_remove_both() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);
        hand.add(card);

        hand.remove_both(card);

        assert!(
            !hand.contains(card),
            "Die Karte sollte nach dem Entfernen nicht mehr enthalten sein."
        );

        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.add(card);

        hand.remove_both(card);

        assert!(
            !hand.contains(card),
            "Die Karte sollte nach dem Entfernen nicht mehr enthalten sein."
        );
    }

    #[test]
    #[should_panic]
    fn test_remove_nonexistent_card() {
        let mut hand = FdoHand::empty();
        let card = FdoCard::HeartQueen;

        hand.remove(card); // Sollte einen Panic auslösen
    }

    #[test]
    fn test_len() {
        let mut hand = FdoHand::empty();
        assert_eq!(hand.len(), 0, "Eine leere Hand sollte 0 Karten enthalten.");

        hand.add(FdoCard::HeartQueen);
        assert_eq!(
            hand.len(),
            1,
            "Eine Hand mit einer Karte sollte 1 Karte enthalten."
        );

        hand.add(FdoCard::HeartNine);
        assert_eq!(
            hand.len(),
            2,
            "Eine Hand mit zwei Karten sollte 2 Karten enthalten."
        );
    }

    #[test]
    fn test_hand_to_vec() {
        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(None)
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♦9 ♣10 ♣9 ♠10 ♠K ♠9 ♥A ♥9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(None)
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♦A ♦10 ♦K ♣A ♣K ♠A ♥K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::Normal))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♦9 ♣10 ♣9 ♠10 ♠K ♠9 ♥A ♥9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::Normal))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♦A ♦10 ♦K ♣A ♣K ♠A ♥K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::Wedding))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♦9 ♣10 ♣9 ♠10 ♠K ♠9 ♥A ♥9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::Wedding))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♦A ♦10 ♦K ♣A ♣K ♠A ♥K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::DiamondsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♦9 ♣10 ♣9 ♠10 ♠K ♠9 ♥A ♥9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::DiamondsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♦A ♦10 ♦K ♣A ♣K ♠A ♥K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::HeartsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♥A ♥9 ♣10 ♣9 ♠10 ♠K ♠9 ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::HeartsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♥K ♣A ♣K ♠A ♦A ♦10 ♦K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::SpadesSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♠10 ♠K ♠9 ♣10 ♣9 ♥A ♥9 ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::SpadesSolo))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♠A ♣A ♣K ♥K ♦A ♦10 ♦K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::ClubsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣J ♠J ♦J ♣10 ♣9 ♠10 ♠K ♠9 ♥A ♥9 ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::ClubsSolo))
                .to_vec(),
            FdoCard::vec_from_str("♥10 ♠Q ♥Q ♦Q ♥J ♣A ♣K ♠A ♥K ♦A ♦10 ♦K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::TrumplessSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣10 ♣Q ♣J ♣9 ♠10 ♠K ♠J ♠9 ♥A ♥9 ♦J ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::TrumplessSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣A ♣K ♠A ♠Q ♥10 ♥K ♥Q ♥J ♦A ♦10 ♦K ♦Q")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::QueensSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣Q ♣10 ♣J ♣9 ♠10 ♠K ♠J ♠9 ♥A ♥9 ♦J ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::QueensSolo))
                .to_vec(),
            FdoCard::vec_from_str("♠Q ♥Q ♦Q ♣A ♣K ♠A ♥10 ♥K ♥J ♦A ♦10 ♦K")
        );

        assert_eq!(
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10")
                .to_vec_sorted(Some(FdoGameType::JacksSolo))
                .to_vec(),
            FdoCard::vec_from_str("♣J ♠J ♦J ♣10 ♣Q ♣9 ♠10 ♠K ♠9 ♥A ♥9 ♦9")
        );
        assert_eq!(
            FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J")
                .to_vec_sorted(Some(FdoGameType::JacksSolo))
                .to_vec(),
            FdoCard::vec_from_str("♥J ♣A ♣K ♠A ♠Q ♥10 ♥K ♥Q ♦A ♦10 ♦K ♦Q")
        );
    }

    #[test]
    fn test_union() {
        let hand1 = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        let hand2 = FdoHand::from_str("♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J");

        let union = hand1.plus_hand(hand2);

        assert_eq!(
            union.len(),
            24,
            "Die Vereinigung sollte 24 Karten enthalten."
        );
        assert_eq!(
            union.0,
            FdoHand::from_str(
                "♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10 ♥K ♦K ♣A ♥10 ♠A ♣K ♠Q ♦A ♥Q ♦Q ♦10 ♥J"
            )
            .0
        );

        let hand1 = FdoHand::from_str("♣Q ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q ♣10 ♥9");

        let union = hand1.plus_hand(hand2);

        assert_eq!(union.len(), 6);
        assert_eq!(union.0, FdoHand::from_str("♣Q ♣10 ♥9 ♣Q ♣10 ♥9").0);

        let hand1 = FdoHand::from_str("♣Q ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q ♣9 ♥10");

        let union = hand1.plus_hand(hand2);

        assert_eq!(union.len(), 6);
        assert_eq!(union.0, FdoHand::from_str("♣Q ♣Q ♣10 ♥9 ♣9 ♥10").0);
    }

    #[test]
    fn test_minus_hand() {
        let hand1 = FdoHand::from_str("♣Q ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q ♣10 ♥9");

        let diff = hand1.minus_hand(hand2);

        assert_eq!(diff.len(), 0);

        let hand1 = FdoHand::from_str("♣Q ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q");

        let diff = hand1.minus_hand(hand2);

        assert_eq!(diff.len(), 2);
        assert_eq!(diff.0, FdoHand::from_str("♣10 ♥9").0);

        let hand1 = FdoHand::from_str("♣Q ♣Q ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q ♣10 ♥9");

        let diff = hand1.minus_hand(hand2);

        assert_eq!(diff.len(), 1);
        assert_eq!(diff.0, FdoHand::from_str("♣Q").0);

        let hand1 = FdoHand::from_str("♣Q ♣Q ♣10 ♣10 ♥9");
        let hand2 = FdoHand::from_str("♣Q ♣10 ♣10");

        let diff = hand1.minus_hand(hand2);

        assert_eq!(diff.len(), 2);
        assert_eq!(diff.0, FdoHand::from_str("♣Q ♥9").0);




    }

    #[test]
    fn test_remove_color() {
        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9");
        hand.remove_color(FdoColor::Club, FdoGameType::Normal);
        assert_eq!(hand.0, FdoHand::from_str("♣Q ♥9").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9");
        hand.remove_color(FdoColor::Club, FdoGameType::ClubsSolo);
        assert_eq!(hand.0, FdoHand::from_str("♣Q ♣10 ♥9").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9");
        hand.remove_color(FdoColor::Club, FdoGameType::TrumplessSolo);
        assert_eq!(hand.0, FdoHand::from_str("♥9").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9");
        hand.remove_color(FdoColor::Club, FdoGameType::HeartsSolo);
        assert_eq!(hand.0, FdoHand::from_str("♣Q ♥9").0);

        let mut hand = FdoHand::from_str("♣Q ♣Q ♣10 ♣10 ♥9");
        hand.remove_color(FdoColor::Club, FdoGameType::Normal);
        assert_eq!(hand.0, FdoHand::from_str("♣Q ♣Q ♥9").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        hand.remove_color(FdoColor::Trump, FdoGameType::ClubsSolo);
        assert_eq!(hand.0, FdoHand::from_str("♥9 ♠9 ♦9 ♠K ♥A ♠10").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        hand.remove_color(FdoColor::Heart, FdoGameType::Normal);
        assert_eq!(
            hand.0,
            FdoHand::from_str("♣Q ♣10 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♠10").0
        );

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        hand.remove_color(FdoColor::Spade, FdoGameType::Normal);
        assert_eq!(hand.0, FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♦J ♥A").0);

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        hand.remove_color(FdoColor::Diamond, FdoGameType::Normal);
        assert_eq!(
            hand.0,
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10").0
        );

        let mut hand = FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♦9 ♠9 ♦J ♠K ♥A ♠10");
        hand.remove_color(FdoColor::Diamond, FdoGameType::SpadesSolo);
        assert_eq!(
            hand.0,
            FdoHand::from_str("♣Q ♣10 ♥9 ♣J ♠J ♣9 ♠9 ♦J ♠K ♥A ♠10").0
        );
    }

    #[test]
    fn test_contains_card_of_color() {
        let hand = FdoHand::from_str("♣Q ♣10 ♥9");

        assert_eq!(hand.contains_card_of_color(FdoColor::Diamond, FdoGameType::Normal), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Heart, FdoGameType::Normal), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Club, FdoGameType::Normal), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Spade, FdoGameType::Normal), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Trump, FdoGameType::Normal), true);

        let hand = FdoHand::from_str("♣Q ♣10 ♥9");

        assert_eq!(hand.contains_card_of_color(FdoColor::Diamond, FdoGameType::ClubsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Heart, FdoGameType::ClubsSolo), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Club, FdoGameType::ClubsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Spade, FdoGameType::ClubsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Trump, FdoGameType::ClubsSolo), true);

        let hand = FdoHand::from_str("♣Q ♣10 ♥9");

        assert_eq!(hand.contains_card_of_color(FdoColor::Diamond, FdoGameType::TrumplessSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Heart, FdoGameType::TrumplessSolo), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Club, FdoGameType::TrumplessSolo), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Spade, FdoGameType::TrumplessSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Trump, FdoGameType::TrumplessSolo), false);

        let hand = FdoHand::from_str("♣Q ♣10 ♥9");

        assert_eq!(hand.contains_card_of_color(FdoColor::Diamond, FdoGameType::HeartsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Heart, FdoGameType::HeartsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Club, FdoGameType::HeartsSolo), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Spade, FdoGameType::HeartsSolo), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Trump, FdoGameType::HeartsSolo), true);

        let hand = FdoHand::from_str("♣Q ♣Q ♣10 ♣10 ♥9");

        assert_eq!(hand.contains_card_of_color(FdoColor::Diamond, FdoGameType::Normal), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Heart, FdoGameType::Normal), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Club, FdoGameType::Normal), true);
        assert_eq!(hand.contains_card_of_color(FdoColor::Spade, FdoGameType::Normal), false);
        assert_eq!(hand.contains_card_of_color(FdoColor::Trump, FdoGameType::Normal), true);



    }


}
