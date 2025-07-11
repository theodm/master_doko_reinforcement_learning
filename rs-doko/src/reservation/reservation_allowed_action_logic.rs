use crate::card::cards::DoCard;
use crate::hand::hand::{DoHand, hand_contains_both};
use crate::player::player::DoPlayer;
use crate::reservation::reservation::DoReservation;
use crate::reservation::reservation_round::DoReservationRound;
use crate::util::bitflag::bitflag::bitflag_add;

pub type DoAllowedReservations = usize;

fn calc_allowed_reservation_actions(
    player_hand: DoHand
) -> DoAllowedReservations {
    let mut allowed_reservations: DoAllowedReservations = 0;

    // Hochzeit ist nur erlaubt, falls der Spiler beide Kreuz-Damen auf der Hand hat.
    if hand_contains_both(player_hand, DoCard::ClubQueen) {
        allowed_reservations = bitflag_add::<DoAllowedReservations, 2>(allowed_reservations, DoReservation::Wedding as usize)
    }

    // Gesund ist immer erlaubt.
    allowed_reservations = bitflag_add::<DoAllowedReservations, 2>(allowed_reservations, DoReservation::Healthy as usize);

    allowed_reservations
}

