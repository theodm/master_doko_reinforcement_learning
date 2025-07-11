use strum::EnumCount;
use rs_game_utils::bit_flag::Bitflag;
use crate::card::cards::FdoCard;
use crate::hand::hand::FdoHand;
use crate::reservation::reservation::FdoReservation;

struct FdoAllowedReservations(Bitflag<{ FdoReservation::COUNT }>);

impl FdoAllowedReservations {
    pub fn from_hand(
        hand: &FdoHand
    ) -> FdoAllowedReservations {
        let mut allowed_reservations: Bitflag<{ FdoReservation::COUNT }> = Bitflag::new();

        // Hochzeit ist nur erlaubt, falls der Spieler beide Kreuz-Damen auf der Hand hat.
        if hand.contains_both(FdoCard::ClubQueen) {
            allowed_reservations.add(FdoReservation::Wedding as u64);
        }

        // Alle anderen Vorbehalte sind immer erlaubt.
        allowed_reservations.add(FdoReservation::Healthy as u64);
        allowed_reservations.add(FdoReservation::DiamondsSolo as u64);
        allowed_reservations.add(FdoReservation::JacksSolo as u64);
        allowed_reservations.add(FdoReservation::QueensSolo as u64);
        allowed_reservations.add(FdoReservation::TrumplessSolo as u64);

        allowed_reservations.add(FdoReservation::DiamondsSolo as u64);
        allowed_reservations.add(FdoReservation::HeartsSolo as u64);
        allowed_reservations.add(FdoReservation::SpadesSolo as u64);
        allowed_reservations.add(FdoReservation::ClubsSolo as u64);

        return FdoAllowedReservations(allowed_reservations);
    }
}

// ToDo: Test fehlen noch
