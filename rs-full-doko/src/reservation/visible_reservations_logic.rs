use crate::player::player::{FdoPlayer};
use crate::reservation::reservation::{FdoReservation, FdoVisibleReservation};
use crate::reservation::reservation_round::{ FdoReservationRound};
use crate::util::po_arr::PlayerOrientedArr;

impl FdoReservationRound {
    pub fn get_visible_reservations(
        &self,
        observing_player: FdoPlayer,
    ) -> PlayerOrientedArr<FdoVisibleReservation> {
        let mut visible_reservations = PlayerOrientedArr::from_full(
            self.reservations.starting_player,
            [
                FdoVisibleReservation::NoneYet,
                FdoVisibleReservation::NoneYet,
                FdoVisibleReservation::NoneYet,
                FdoVisibleReservation::NoneYet,
            ]
        );

        // Ist die Vorbehaltsrunde bereits abgeschlossen? Wenn nein, dann werden alle Vorbehalte noch
        // nicht angezeigt.
        let is_reservation_round_completed = self.is_completed();

        // Ist die Vorberhaltsrunde abgeschlossen, werden alle Vorbehalte nach dem höchsten Vorbehalt (Solo)
        // nicht angezeigt.
        let mut is_higher_reservation_made = false;

        for (current_player, reservation) in self.reservations.iter_with_player() {
            match reservation {
                FdoReservation::Healthy => {
                    visible_reservations[current_player] = FdoVisibleReservation::Healthy;
                }

                FdoReservation::Wedding => {
                    if is_reservation_round_completed || current_player == observing_player {
                        visible_reservations[current_player] = FdoVisibleReservation::Wedding;
                    } else {
                        visible_reservations[current_player] = FdoVisibleReservation::NotRevealed;
                    }
                }

                // Alle Solis
                reservation => {
                    debug_assert!(*reservation != FdoReservation::Healthy);
                    debug_assert!(*reservation != FdoReservation::Wedding);

                    if is_reservation_round_completed && !is_higher_reservation_made {
                        visible_reservations[current_player] = match reservation {
                            FdoReservation::DiamondsSolo => FdoVisibleReservation::DiamondsSolo,
                            FdoReservation::HeartsSolo => FdoVisibleReservation::HeartsSolo,
                            FdoReservation::SpadesSolo => FdoVisibleReservation::SpadesSolo,
                            FdoReservation::ClubsSolo => FdoVisibleReservation::ClubsSolo,
                            FdoReservation::QueensSolo => FdoVisibleReservation::QueensSolo,
                            FdoReservation::JacksSolo => FdoVisibleReservation::JacksSolo,
                            FdoReservation::TrumplessSolo => FdoVisibleReservation::TrumplessSolo,
                            _ => panic!("should not happen"),
                        };

                        is_higher_reservation_made = true;
                    } else {
                        visible_reservations[current_player] = FdoVisibleReservation::NotRevealed;
                    }
                }
            }
        }

        visible_reservations
    }
}

#[cfg(test)]
mod tests {
    use crate::player::player::{FdoPlayer};
    use crate::reservation::reservation::{FdoReservation, FdoVisibleReservation};
    use crate::reservation::reservation_round::{FdoReservationRound};

    #[test]
    fn test_open_round() {
        let reservation_round = FdoReservationRound::existing(FdoPlayer::TOP, vec![
            FdoReservation::Healthy,
            FdoReservation::SpadesSolo,
            FdoReservation::Wedding,
        ]);

        let visible_reservations = reservation_round.get_visible_reservations(FdoPlayer::LEFT);

        assert_eq!(visible_reservations[FdoPlayer::TOP], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::RIGHT], FdoVisibleReservation::NotRevealed);
        assert_eq!(visible_reservations[FdoPlayer::BOTTOM], FdoVisibleReservation::NotRevealed);
        assert_eq!(visible_reservations[FdoPlayer::LEFT], FdoVisibleReservation::NoneYet);
    }

    #[test]
    fn test_all_healthy() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Healthy,
            ],
        );

        let visible_reservations = reservation_round.get_visible_reservations(FdoPlayer::TOP);

        assert_eq!(visible_reservations[FdoPlayer::TOP], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::RIGHT], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::BOTTOM], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::LEFT], FdoVisibleReservation::Healthy);
    }

    #[test]
    fn test_wedding() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::TOP,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Healthy,
                FdoReservation::Wedding,
                FdoReservation::Healthy,
            ],
        );

        let visible_reservations = reservation_round.get_visible_reservations(FdoPlayer::TOP);

        assert_eq!(visible_reservations[FdoPlayer::TOP], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::RIGHT], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::BOTTOM], FdoVisibleReservation::Wedding);
        assert_eq!(visible_reservations[FdoPlayer::LEFT], FdoVisibleReservation::Healthy);
    }

    #[test]
    fn test_wedding_overriden() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
                FdoReservation::Wedding,
                FdoReservation::SpadesSolo,
                FdoReservation::TrumplessSolo,
            ],
        );

        let visible_reservations = reservation_round.get_visible_reservations(FdoPlayer::TOP);

        assert_eq!(visible_reservations[FdoPlayer::LEFT], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::TOP], FdoVisibleReservation::Wedding);
        assert_eq!(visible_reservations[FdoPlayer::RIGHT], FdoVisibleReservation::SpadesSolo);
        assert_eq!(visible_reservations[FdoPlayer::BOTTOM], FdoVisibleReservation::NotRevealed);
    }

    #[test]
    fn test_two_solis() {
        let reservation_round = FdoReservationRound::existing(
            FdoPlayer::LEFT,
            vec![
                FdoReservation::Healthy,
                FdoReservation::SpadesSolo,
                FdoReservation::TrumplessSolo,
                FdoReservation::Healthy,
            ],
        );

        let visible_reservations = reservation_round.get_visible_reservations(FdoPlayer::TOP);

        assert_eq!(visible_reservations[FdoPlayer::LEFT], FdoVisibleReservation::Healthy);
        assert_eq!(visible_reservations[FdoPlayer::TOP], FdoVisibleReservation::SpadesSolo);
        assert_eq!(visible_reservations[FdoPlayer::RIGHT], FdoVisibleReservation::NotRevealed);
        assert_eq!(visible_reservations[FdoPlayer::BOTTOM], FdoVisibleReservation::Healthy);

    }
}
