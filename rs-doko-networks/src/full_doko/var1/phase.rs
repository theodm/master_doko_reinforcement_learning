use rs_full_doko::basic::phase::FdoPhase;

/// Anzahl der Phasen im Spiel (für die Embedding-Größe).
pub const PHASE_COUNT: i64 = 3;

/// Kodiert die Phase des Spiels. Wird
/// als Embedding innerhalb des neuronalen
/// Netzwerkes verwendet.
pub fn encode_phase(
    phase: FdoPhase
) -> [i64; 1] {
    let phase_num = phase as i64;

    debug_assert!(phase_num < PHASE_COUNT);
    debug_assert!(phase_num >= 0);

    [phase_num]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_phase() {
        assert_eq!(encode_phase(FdoPhase::Reservation), [0]);
        assert_eq!(encode_phase(FdoPhase::Announcement), [1]);
        assert_eq!(encode_phase(FdoPhase::PlayCard), [2]);
    }
}