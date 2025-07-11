use std::fmt::Debug;
use rand_distr::num_traits::Zero;

/// Setzt die Werte des übergebenen Arrays [`data`] an den Stellen,
/// die in [`indices`] NICHT angegeben sind, auf den Minimalwert. Also
/// werden die [`indices`] als "erlaubte" Indizes interpretiert und
/// die Daten maskiert.
pub fn set_non_indices_to_min<const N: usize>(
    data: &mut [f32; N],

    indices: &heapless::Vec<usize, N>,
) {
    for index in 0..N {
        if !indices.contains(&index) {
            data[index] = f32::MIN;
        }
    }
}

/// Berechnet den Index des größten Wertes im übergebenen Array [`data`].
pub fn argmax<const N: usize>(
    data: &[f32; N],
) -> usize {
    let mut max_index = 0;
    let mut max_value = f32::MIN;

    for (index, &value) in data
        .iter()
        .enumerate() {
        if value > max_value {
            max_value = value;
            max_index = index;
        }
    }

    max_index
}


/// Konvertiert das übergebene Array [`data_in_memory`] in ein Vec, wobei
/// die Indizes, die in [`indices`] angegeben sind, dabei entfernt werden.
pub fn to_vec_with_indices_removed<
    const N: usize
>(
    data_in_memory: &[f32; N],
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

pub fn divide_by<const N: usize>(
    data: &mut [f32; N],
    divisor: f32,
) {
    for i in 0..N {
        data[i] /= divisor;
    }
}