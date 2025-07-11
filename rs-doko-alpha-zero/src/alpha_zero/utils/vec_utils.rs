
pub fn flatten_nested_heapless_vec_of_array<
    T: Clone + Copy,
    const N: usize,
    const B: usize
>(
    nested_vec: &heapless::Vec<[T; N], B>,
) -> Vec<T> {
    nested_vec
        .iter()
        .flat_map(|x| x.iter())
        .copied()
        .collect()
}

pub fn flatten_nested_vec_of_array<
    T: Clone + Copy,
    const N: usize
>(
    nested_vec: &Vec<[T; N]>,
) -> Vec<T> {
    nested_vec
        .iter()
        .flat_map(|x| x.iter())
        .copied()
        .collect()
}