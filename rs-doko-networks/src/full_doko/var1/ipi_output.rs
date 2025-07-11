use std::fmt::{Display, Formatter};
use array_concat::concat_arrays;
use enumset::EnumSet;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIs, EnumProperty};
use rs_full_doko::card::cards::FdoCard;
use rs_full_doko::hand::hand::FdoHand;
use rs_full_doko::reservation::reservation::FdoReservation;

// Das sind die Werte zwischen denen das neuronale Netz für die
// imperfekte Information unterscheiden soll.
#[derive(EnumCount, Debug)]
pub enum ImperfectInformationOutput {
    Healthy = 0,
    Wedding = 1,

    DiamondsSolo = 2,
    HeartsSolo = 3,
    SpadesSolo = 4,
    ClubsSolo = 5,

    QueensSolo = 6,
    JacksSolo = 7,

    TrumplessSolo = 8,

    DiamondNine = 9,
    DiamondTen = 10,
    DiamondJack = 11,
    DiamondQueen = 12,
    DiamondKing = 13,
    DiamondAce = 14,

    HeartNine = 15,
    HeartTen = 16,
    HeartJack = 17,
    HeartQueen = 18,
    HeartKing = 19,
    HeartAce = 20,

    ClubNine = 21,
    ClubTen = 22,
    ClubJack = 23,
    ClubQueen = 24,
    ClubKing = 25,
    ClubAce = 26,

    SpadeNine = 27,
    SpadeTen = 28,
    SpadeJack = 29,
    SpadeQueen = 30,
    SpadeKing = 31,
    SpadeAce = 32,
}

impl ImperfectInformationOutput {
    pub fn mask_for_reservation(p0: EnumSet<FdoReservation>) -> [bool; ImperfectInformationOutput::COUNT] {
        let mut mask = [false; ImperfectInformationOutput::COUNT];

        for reservation in p0.iter() {
            let imp = ImperfectInformationOutput::from_reservation(
                reservation
            );
            mask[imp as usize] = true;
        }

        return mask;
    }
}

impl ImperfectInformationOutput {
    pub fn mask_from_hand(hand: FdoHand) -> [bool; ImperfectInformationOutput::COUNT] {
        let mut mask = [false; ImperfectInformationOutput::COUNT];

        for card in hand.iter() {
            let imp = ImperfectInformationOutput::from_card(&card);
            mask[imp as usize] = true;
        }

        return mask;
    }
}

impl ImperfectInformationOutput {
    pub fn arr_from_hand(
        hand: FdoHand
    ) -> [f32; ImperfectInformationOutput::COUNT] {
        let mut arr = [0.0; ImperfectInformationOutput::COUNT];

        let num_of_cards = hand.len();
        let float_value = 1.0 / num_of_cards as f32;

        for card in hand.iter() {
            let imp = ImperfectInformationOutput::from_card(&card);

            match imp {
                ImperfectInformationOutput::Healthy => arr[0] = float_value,
                ImperfectInformationOutput::Wedding => arr[1] = float_value,

                ImperfectInformationOutput::DiamondsSolo => arr[2] = float_value,
                ImperfectInformationOutput::HeartsSolo => arr[3] = float_value,
                ImperfectInformationOutput::SpadesSolo => arr[4] = float_value,
                ImperfectInformationOutput::ClubsSolo => arr[5] = float_value,

                ImperfectInformationOutput::QueensSolo => arr[6] = float_value,
                ImperfectInformationOutput::JacksSolo => arr[7] = float_value,

                ImperfectInformationOutput::TrumplessSolo => arr[8] = float_value,

                ImperfectInformationOutput::DiamondNine => arr[9] = float_value,
                ImperfectInformationOutput::DiamondTen => arr[10] = float_value,
                ImperfectInformationOutput::DiamondJack => arr[11] = float_value,
                ImperfectInformationOutput::DiamondQueen => arr[12] = float_value,
                ImperfectInformationOutput::DiamondKing => arr[13] = float_value,
                ImperfectInformationOutput::DiamondAce => arr[14] = float_value,

                ImperfectInformationOutput::HeartNine => arr[15] = float_value,
                ImperfectInformationOutput::HeartTen => arr[16] = float_value,
                ImperfectInformationOutput::HeartJack => arr[17] = float_value,
                ImperfectInformationOutput::HeartQueen => arr[18] = float_value,
                ImperfectInformationOutput::HeartKing => arr[19] = float_value,
                ImperfectInformationOutput::HeartAce => arr[20] = float_value,

                ImperfectInformationOutput::ClubNine => arr[21] = float_value,
                ImperfectInformationOutput::ClubTen => arr[22] = float_value,
                ImperfectInformationOutput::ClubJack => arr[23] = float_value,
                ImperfectInformationOutput::ClubQueen => arr[24] = float_value,
                ImperfectInformationOutput::ClubKing => arr[25] = float_value,
                ImperfectInformationOutput::ClubAce => arr[26] = float_value,

                ImperfectInformationOutput::SpadeNine => arr[27] = float_value,
                ImperfectInformationOutput::SpadeTen => arr[28] = float_value,
                ImperfectInformationOutput::SpadeJack => arr[29] = float_value,
                ImperfectInformationOutput::SpadeQueen => arr[30] = float_value,
                ImperfectInformationOutput::SpadeKing => arr[31] = float_value,
                ImperfectInformationOutput::SpadeAce => arr[32] = float_value,
            }
        }

        return arr;
    }
}

impl Display for ImperfectInformationOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ImperfectInformationOutput::Healthy => "Healthy".fmt(f),
            ImperfectInformationOutput::Wedding => "Wedding".fmt(f),

            ImperfectInformationOutput::DiamondsSolo => "DiamondsSolo".fmt(f),
            ImperfectInformationOutput::HeartsSolo => "HeartsSolo".fmt(f),
            ImperfectInformationOutput::SpadesSolo => "SpadesSolo".fmt(f),
            ImperfectInformationOutput::ClubsSolo => "ClubsSolo".fmt(f),

            ImperfectInformationOutput::QueensSolo => "QueensSolo".fmt(f),
            ImperfectInformationOutput::JacksSolo => "JacksSolo".fmt(f),

            ImperfectInformationOutput::TrumplessSolo => "TrumplessSolo".fmt(f),

            ImperfectInformationOutput::DiamondNine => "♦9".fmt(f),
            ImperfectInformationOutput::DiamondTen => "♦10".fmt(f),
            ImperfectInformationOutput::DiamondJack => "♦J".fmt(f),
            ImperfectInformationOutput::DiamondQueen => "♦Q".fmt(f),
            ImperfectInformationOutput::DiamondKing => "♦K".fmt(f),
            ImperfectInformationOutput::DiamondAce => "♦A".fmt(f),

            ImperfectInformationOutput::HeartNine => "♥9".fmt(f),
            ImperfectInformationOutput::HeartTen => "♥10".fmt(f),
            ImperfectInformationOutput::HeartJack => "♥J".fmt(f),
            ImperfectInformationOutput::HeartQueen => "♥Q".fmt(f),
            ImperfectInformationOutput::HeartKing => "♥K".fmt(f),
            ImperfectInformationOutput::HeartAce => "♥A".fmt(f),

            ImperfectInformationOutput::ClubNine => "♣9".fmt(f),
            ImperfectInformationOutput::ClubTen => "♣10".fmt(f),
            ImperfectInformationOutput::ClubJack => "♣J".fmt(f),
            ImperfectInformationOutput::ClubQueen => "♣Q".fmt(f),
            ImperfectInformationOutput::ClubKing => "♣K".fmt(f),
            ImperfectInformationOutput::ClubAce => "♣A".fmt(f),

            ImperfectInformationOutput::SpadeNine => "♠9".fmt(f),
            ImperfectInformationOutput::SpadeTen => "♠10".fmt(f),
            ImperfectInformationOutput::SpadeJack => "♠J".fmt(f),
            ImperfectInformationOutput::SpadeQueen => "♠Q".fmt(f),
            ImperfectInformationOutput::SpadeKing => "♠K".fmt(f),
            ImperfectInformationOutput::SpadeAce => "♠A".fmt(f),
        }
    }
}

impl ImperfectInformationOutput {

}

impl ImperfectInformationOutput {

    pub fn stringify_logits(logits: [f32; ImperfectInformationOutput::COUNT]) -> String {
        let mut indices: Vec<usize> = (0..ImperfectInformationOutput::COUNT).collect();

        indices.sort_by(|&i, &j| {
            logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let lines: Vec<String> = indices.into_iter().map(|i| {
            let variant: ImperfectInformationOutput = unsafe { std::mem::transmute(i as u8) };

            format!("{}: {:.2} %", variant, logits[i] * 100.0)
        }).collect();

        lines.join(", ")
    }

    pub fn from_index(index: usize) -> ImperfectInformationOutput {
        match index {
            0 => ImperfectInformationOutput::Healthy,
            1 => ImperfectInformationOutput::Wedding,

            2 => ImperfectInformationOutput::DiamondsSolo,
            3 => ImperfectInformationOutput::HeartsSolo,
            4 => ImperfectInformationOutput::SpadesSolo,
            5 => ImperfectInformationOutput::ClubsSolo,

            6 => ImperfectInformationOutput::QueensSolo,
            7 => ImperfectInformationOutput::JacksSolo,

            8 => ImperfectInformationOutput::TrumplessSolo,

            9 => ImperfectInformationOutput::DiamondNine,
            10 => ImperfectInformationOutput::DiamondTen,
            11 => ImperfectInformationOutput::DiamondJack,
            12 => ImperfectInformationOutput::DiamondQueen,
            13 => ImperfectInformationOutput::DiamondKing,
            14 => ImperfectInformationOutput::DiamondAce,

            15 => ImperfectInformationOutput::HeartNine,
            16 => ImperfectInformationOutput::HeartTen,
            17 => ImperfectInformationOutput::HeartJack,
            18 => ImperfectInformationOutput::HeartQueen,
            19 => ImperfectInformationOutput::HeartKing,
            20 => ImperfectInformationOutput::HeartAce,

            21 => ImperfectInformationOutput::ClubNine,
            22 => ImperfectInformationOutput::ClubTen,
            23 => ImperfectInformationOutput::ClubJack,
            24 => ImperfectInformationOutput::ClubQueen,
            25 => ImperfectInformationOutput::ClubKing,
            26 => ImperfectInformationOutput::ClubAce,

            27 => ImperfectInformationOutput::SpadeNine,
            28 => ImperfectInformationOutput::SpadeTen,
            29 => ImperfectInformationOutput::SpadeJack,
            30 => ImperfectInformationOutput::SpadeQueen,
            31 => ImperfectInformationOutput::SpadeKing,
            32 => ImperfectInformationOutput::SpadeAce,

            _ => panic!("Invalid index: {}", index)
        }
    }
}

impl ImperfectInformationOutput {

    pub fn card_only_mask() -> [bool; ImperfectInformationOutput::COUNT] {
        return concat_arrays!([false; 9], [true; 24]);
    }

    pub fn reservation_only_mask() -> [bool; ImperfectInformationOutput::COUNT] {
        return concat_arrays!([true; 9], [false; 24]);
    }

    pub fn from_reservation(
        reservation: FdoReservation
    ) -> ImperfectInformationOutput {
        match reservation {
            FdoReservation::Healthy => ImperfectInformationOutput::Healthy,
            FdoReservation::Wedding => ImperfectInformationOutput::Wedding,

            FdoReservation::DiamondsSolo => ImperfectInformationOutput::DiamondsSolo,
            FdoReservation::HeartsSolo => ImperfectInformationOutput::HeartsSolo,
            FdoReservation::SpadesSolo => ImperfectInformationOutput::SpadesSolo,
            FdoReservation::ClubsSolo => ImperfectInformationOutput::ClubsSolo,

            FdoReservation::QueensSolo => ImperfectInformationOutput::QueensSolo,
            FdoReservation::JacksSolo => ImperfectInformationOutput::JacksSolo,

            FdoReservation::TrumplessSolo => ImperfectInformationOutput::TrumplessSolo,
        }
    }


    pub fn from_card(
        card: &FdoCard
    ) -> ImperfectInformationOutput {
        match card {
            FdoCard::DiamondNine => ImperfectInformationOutput::DiamondNine,
            FdoCard::DiamondTen => ImperfectInformationOutput::DiamondTen,
            FdoCard::DiamondJack => ImperfectInformationOutput::DiamondJack,
            FdoCard::DiamondQueen => ImperfectInformationOutput::DiamondQueen,
            FdoCard::DiamondKing => ImperfectInformationOutput::DiamondKing,
            FdoCard::DiamondAce => ImperfectInformationOutput::DiamondAce,

            FdoCard::HeartNine => ImperfectInformationOutput::HeartNine,
            FdoCard::HeartTen => ImperfectInformationOutput::HeartTen,
            FdoCard::HeartJack => ImperfectInformationOutput::HeartJack,
            FdoCard::HeartQueen => ImperfectInformationOutput::HeartQueen,
            FdoCard::HeartKing => ImperfectInformationOutput::HeartKing,
            FdoCard::HeartAce => ImperfectInformationOutput::HeartAce,

            FdoCard::ClubNine => ImperfectInformationOutput::ClubNine,
            FdoCard::ClubTen => ImperfectInformationOutput::ClubTen,
            FdoCard::ClubJack => ImperfectInformationOutput::ClubJack,
            FdoCard::ClubQueen => ImperfectInformationOutput::ClubQueen,
            FdoCard::ClubKing => ImperfectInformationOutput::ClubKing,
            FdoCard::ClubAce => ImperfectInformationOutput::ClubAce,

            FdoCard::SpadeNine => ImperfectInformationOutput::SpadeNine,
            FdoCard::SpadeTen => ImperfectInformationOutput::SpadeTen,
            FdoCard::SpadeJack => ImperfectInformationOutput::SpadeJack,
            FdoCard::SpadeQueen => ImperfectInformationOutput::SpadeQueen,
            FdoCard::SpadeKing => ImperfectInformationOutput::SpadeKing,
            FdoCard::SpadeAce => ImperfectInformationOutput::SpadeAce,
        }
    }

    pub fn to_card(&self) -> Option<FdoCard> {
        match self {
            ImperfectInformationOutput::Healthy => None,
            ImperfectInformationOutput::Wedding => None,

            ImperfectInformationOutput::DiamondsSolo => None,
            ImperfectInformationOutput::HeartsSolo => None,
            ImperfectInformationOutput::SpadesSolo => None,
            ImperfectInformationOutput::ClubsSolo => None,

            ImperfectInformationOutput::QueensSolo => None,
            ImperfectInformationOutput::JacksSolo => None,

            ImperfectInformationOutput::TrumplessSolo => None,

            ImperfectInformationOutput::DiamondNine => Some(FdoCard::DiamondNine),
            ImperfectInformationOutput::DiamondTen => Some(FdoCard::DiamondTen),
            ImperfectInformationOutput::DiamondJack => Some(FdoCard::DiamondJack),
            ImperfectInformationOutput::DiamondQueen => Some(FdoCard::DiamondQueen),
            ImperfectInformationOutput::DiamondKing => Some(FdoCard::DiamondKing),
            ImperfectInformationOutput::DiamondAce => Some(FdoCard::DiamondAce),

            ImperfectInformationOutput::HeartNine => Some(FdoCard::HeartNine),
            ImperfectInformationOutput::HeartTen => Some(FdoCard::HeartTen),
            ImperfectInformationOutput::HeartJack => Some(FdoCard::HeartJack),
            ImperfectInformationOutput::HeartQueen => Some(FdoCard::HeartQueen),
            ImperfectInformationOutput::HeartKing => Some(FdoCard::HeartKing),
            ImperfectInformationOutput::HeartAce => Some(FdoCard::HeartAce),

            ImperfectInformationOutput::ClubNine => Some(FdoCard::ClubNine),
            ImperfectInformationOutput::ClubTen => Some(FdoCard::ClubTen),
            ImperfectInformationOutput::ClubJack => Some(FdoCard::ClubJack),
            ImperfectInformationOutput::ClubQueen => Some(FdoCard::ClubQueen),
            ImperfectInformationOutput::ClubKing => Some(FdoCard::ClubKing),
            ImperfectInformationOutput::ClubAce => Some(FdoCard::ClubAce),

            ImperfectInformationOutput::SpadeNine => Some(FdoCard::SpadeNine),
            ImperfectInformationOutput::SpadeTen => Some(FdoCard::SpadeTen),
            ImperfectInformationOutput::SpadeJack => Some(FdoCard::SpadeJack),
            ImperfectInformationOutput::SpadeQueen => Some(FdoCard::SpadeQueen),
            ImperfectInformationOutput::SpadeKing => Some(FdoCard::SpadeKing),
            ImperfectInformationOutput::SpadeAce => Some(FdoCard::SpadeAce),
        }
    }

    pub fn to_reservation(&self) -> Option<FdoReservation> {
        match self {
            ImperfectInformationOutput::Healthy => Some(FdoReservation::Healthy),
            ImperfectInformationOutput::Wedding => Some(FdoReservation::Wedding),

            ImperfectInformationOutput::DiamondsSolo => Some(FdoReservation::DiamondsSolo),
            ImperfectInformationOutput::HeartsSolo => Some(FdoReservation::HeartsSolo),
            ImperfectInformationOutput::SpadesSolo => Some(FdoReservation::SpadesSolo),
            ImperfectInformationOutput::ClubsSolo => Some(FdoReservation::ClubsSolo),

            ImperfectInformationOutput::QueensSolo => Some(FdoReservation::QueensSolo),
            ImperfectInformationOutput::JacksSolo => Some(FdoReservation::JacksSolo),

            ImperfectInformationOutput::TrumplessSolo => Some(FdoReservation::TrumplessSolo),

            ImperfectInformationOutput::DiamondNine => None,
            ImperfectInformationOutput::DiamondTen => None,
            ImperfectInformationOutput::DiamondJack => None,
            ImperfectInformationOutput::DiamondQueen => None,
            ImperfectInformationOutput::DiamondKing => None,
            ImperfectInformationOutput::DiamondAce => None,

            ImperfectInformationOutput::HeartNine => None,
            ImperfectInformationOutput::HeartTen => None,
            ImperfectInformationOutput::HeartJack => None,
            ImperfectInformationOutput::HeartQueen => None,
            ImperfectInformationOutput::HeartKing => None,
            ImperfectInformationOutput::HeartAce => None,

            ImperfectInformationOutput::ClubNine => None,
            ImperfectInformationOutput::ClubTen => None,
            ImperfectInformationOutput::ClubJack => None,
            ImperfectInformationOutput::ClubQueen => None,
            ImperfectInformationOutput::ClubKing => None,
            ImperfectInformationOutput::ClubAce => None,

            ImperfectInformationOutput::SpadeNine => None,
            ImperfectInformationOutput::SpadeTen => None,
            ImperfectInformationOutput::SpadeJack => None,
            ImperfectInformationOutput::SpadeQueen => None,
            ImperfectInformationOutput::SpadeKing => None,
            ImperfectInformationOutput::SpadeAce => None,
        }
    }

    pub fn to_arr(&self) -> [f32; ImperfectInformationOutput::COUNT] {
        let mut result = [0.0; ImperfectInformationOutput::COUNT];

        match self {
            ImperfectInformationOutput::Healthy => result[0] = 1.0,
            ImperfectInformationOutput::Wedding => result[1] = 1.0,

            ImperfectInformationOutput::DiamondsSolo => result[2] = 1.0,
            ImperfectInformationOutput::HeartsSolo => result[3] = 1.0,
            ImperfectInformationOutput::SpadesSolo => result[4] = 1.0,
            ImperfectInformationOutput::ClubsSolo => result[5] = 1.0,

            ImperfectInformationOutput::QueensSolo => result[6] = 1.0,
            ImperfectInformationOutput::JacksSolo => result[7] = 1.0,

            ImperfectInformationOutput::TrumplessSolo => result[8] = 1.0,

            ImperfectInformationOutput::DiamondNine => result[9] = 1.0,
            ImperfectInformationOutput::DiamondTen => result[10] = 1.0,
            ImperfectInformationOutput::DiamondJack => result[11] = 1.0,
            ImperfectInformationOutput::DiamondQueen => result[12] = 1.0,
            ImperfectInformationOutput::DiamondKing => result[13] = 1.0,
            ImperfectInformationOutput::DiamondAce => result[14] = 1.0,

            ImperfectInformationOutput::HeartNine => result[15] = 1.0,
            ImperfectInformationOutput::HeartTen => result[16] = 1.0,
            ImperfectInformationOutput::HeartJack => result[17] = 1.0,
            ImperfectInformationOutput::HeartQueen => result[18] = 1.0,
            ImperfectInformationOutput::HeartKing => result[19] = 1.0,
            ImperfectInformationOutput::HeartAce => result[20] = 1.0,

            ImperfectInformationOutput::ClubNine => result[21] = 1.0,
            ImperfectInformationOutput::ClubTen => result[22] = 1.0,
            ImperfectInformationOutput::ClubJack => result[23] = 1.0,
            ImperfectInformationOutput::ClubQueen => result[24] = 1.0,
            ImperfectInformationOutput::ClubKing => result[25] = 1.0,
            ImperfectInformationOutput::ClubAce => result[26] = 1.0,

            ImperfectInformationOutput::SpadeNine => result[27] = 1.0,
            ImperfectInformationOutput::SpadeTen => result[28] = 1.0,
            ImperfectInformationOutput::SpadeJack => result[29] = 1.0,
            ImperfectInformationOutput::SpadeQueen => result[30] = 1.0,
            ImperfectInformationOutput::SpadeKing => result[31] = 1.0,
            ImperfectInformationOutput::SpadeAce => result[32] = 1.0,
        }

        result
    }

}