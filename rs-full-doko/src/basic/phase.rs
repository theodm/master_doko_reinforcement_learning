use std::fmt::{Debug, Formatter};
use serde::{Deserialize, Serialize};

#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FdoPhase {
    Reservation = 0,
    Announcement = 1,
    PlayCard = 2,
    Finished = 3
}

impl Debug for FdoPhase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FdoPhase::Reservation => write!(f, "FdoPhase::Reservation"),
            FdoPhase::Announcement => write!(f, "FdoPhase::Announcement"),
            FdoPhase::PlayCard => write!(f, "FdoPhase::PlayCard"),
            FdoPhase::Finished => write!(f, "FdoPhase::Finished")
        }
    }
}