use std::fmt::{Debug, Formatter};

/// Repräsentiert einen gemachten Vorbehalt. Aber es handelt sich immer
/// um den geheimen Vorbehalt, der zunächst nur für den Spieler sichtbar ist.
/// (Andere Spieler können bis zur entsprechenden Aufklärung erst nur zwischen Vorbehalt und gesund
/// unterscheiden)
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoReservation {
    Wedding = 0,
    Healthy = 1
}

/// Repräsentiert einen Vorbehalt eines Spielers aus seiner aktuellen Sicht,
/// denn nicht alle Vorbehalte müssen angezeigt werden. Verfällt ein Vorbehalt
/// eines Spielers, weil ein vorheriger Spieler einen höheren Vorbehalt angesagt hat,
/// so wird dieser Vorbehalt nicht aufgedeckt.
#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoVisibleReservation {
    Wedding = 0,
    Healthy = 1,
    NotRevealed = 2
}

impl Debug for DoVisibleReservation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DoVisibleReservation::Wedding => write!(f, "DoVisibleReservation::Wedding"),
            DoVisibleReservation::Healthy => write!(f, "DoVisibleReservation::Healthy"),
            DoVisibleReservation::NotRevealed => write!(f, "DoVisibleReservation::NotRevealed")
        }
    }
}