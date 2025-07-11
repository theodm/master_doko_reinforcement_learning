

pub fn encode_subposition_card(
    subposition: Option<usize>
) -> [i64; 1] {
    match subposition {
        None => [0],
        Some(0) => [11],
        Some(1) => [12],
        _ => panic!("should not happen")
    }
}

pub fn encode_subposition_pos(
    pos: Option<usize>
) -> [i64; 1] {
    match pos {
        None => [0],
        Some(pos) => [(pos + 1) as i64],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_subposition_card() {
        assert_eq!(encode_subposition_card(None), [0]);
        assert_eq!(encode_subposition_card(Some(0)), [11]);
        assert_eq!(encode_subposition_card(Some(1)), [12]);
    }

    #[test]
    #[should_panic(expected = "should not happen")]
    fn test_encode_subposition_card_invalid() {
        encode_subposition_card(Some(2)); // This should panic
    }

    #[test]
    fn test_encode_subposition_pos() {
        assert_eq!(encode_subposition_pos(None), [0]);
        assert_eq!(encode_subposition_pos(Some(0)), [1]);
        assert_eq!(encode_subposition_pos(Some(1)), [2]);
        assert_eq!(encode_subposition_pos(Some(10)), [11]);
    }
}
