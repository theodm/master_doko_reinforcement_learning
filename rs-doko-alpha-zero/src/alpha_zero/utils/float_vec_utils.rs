use rand_distr::num_traits::Zero;

/// Führt die Softmax-Funktion auf den gegebenen Vektor aus.
pub fn softmax_inplace<const N: usize>(
    data: &mut heapless::Vec<f32, N>
) {
    for i in 0..data.len() {
        data[i] = data[i].exp();
    }

    let sum: f32 = data.iter().sum();

    for i in 0..data.len() {
        data[i] /= sum;
    }
}


pub fn vec_with_non_indices_removed<const N: usize>(
    data_in_memory: &heapless::Vec<f32, N>,

    indices: &heapless::Vec<usize, N>,
) -> heapless::Vec<f32, N> {
    let mut data = heapless::Vec::new();

    for &i in indices.iter() {
        data
            .push(data_in_memory[i])
            .unwrap();
    }

    data
}

