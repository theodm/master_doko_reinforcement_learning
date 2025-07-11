
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoPhase {
    Reservation = 0,
    PlayCard = 1,
    Finished = 2
}