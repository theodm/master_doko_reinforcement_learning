
pub fn encode_bool(
    value: bool
) -> [i64; 1] {
    if value {
        [1]
    } else {
        [0]
    }
}