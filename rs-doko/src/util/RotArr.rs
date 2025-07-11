use std::array::from_fn;
use std::mem::MaybeUninit;
use std::ops;
use std::ops::{Add, Div};
use num_traits::Zero;

#[derive(Clone, Debug)]
pub struct RotArr<D: Clone, const N: usize> {
    pub(crate) pov_i: usize,

    // Das Array, das von der Perspektive pov_i aus hier
    // gespeichert wird.
    data: [D; N],
}

pub fn index_for_i(pov_i: usize, i: usize, N: usize) -> usize {
    ((N - pov_i) + i) % N
}

impl<D: Clone, const N: usize> RotArr<D, N> {
    pub fn new_from_0(
        pov_i: usize,
        data_from_0: [D; N]
    ) -> Self {
        // Will this be optimized?
        let mut new_data: [D; N] = from_fn(|i| data_from_0[index_for_i(pov_i, i, N)].clone());

        RotArr {
            pov_i,
            data: new_data,
        }
    }

    pub fn new_from_perspective(
        perspective_of_index: usize,
        data_from_pov: [D; N]
    ) -> Self {
        RotArr {
            pov_i: perspective_of_index,
            data: data_from_pov,
        }
    }

    pub fn map<F, R: Clone>(self, f: F) -> RotArr<R, N>
    where
        F: FnMut(D) -> R,
    {
        let new_data = self.data.map(f);

        RotArr {
            pov_i: self.pov_i,
            data: new_data,
        }
    }

    pub fn map_indexed<F, R: Clone>(self, mut f: F) -> RotArr<R, N>
    where
        F: FnMut(usize, D) -> R,
    {
        let mut new_data = from_fn(|i| f(i, self.data[i].clone()));

        RotArr {
            pov_i: self.pov_i,
            data: new_data,
        }
    }

    pub fn extract(&self) -> [D; N] {
        self.data.clone()
    }

    pub fn rotate_to_perspective(&self, target_perpective_index: usize) -> RotArr<D, N> {
        let new_data = from_fn(|i| self.get_element_for_perspective(i));

        return RotArr::new_from_0(target_perpective_index, new_data);
    }

    pub fn get_element_for_perspective(&self, i: usize) -> D {
        self.data[index_for_i(self.pov_i, i, N)].clone()
    }

    pub fn from_pov_index(&self, i: usize) -> D {
        return self.data[i].clone()
    }

}

impl<
    D: Copy + Add<Output = D> + Zero + Div<Output = D>,
    const N: usize
> Div<D> for RotArr<D, N> {
    type Output = RotArr<D, N>;

    fn div(self, rhs: D) -> Self::Output {
        let mut new_data = [D::zero(); N];

        for i in 0..N {
            new_data[i] = self.data[i] / rhs;
        }

        RotArr {
            pov_i: self.pov_i,
            data: new_data,
        }
    }
}


impl<D: Copy + Add + Zero, const N: usize> RotArr<D, N> {
    pub fn zeros(
        pov_i: usize
    ) -> Self {
        RotArr {
            pov_i,
            data: [D::zero(); N],
        }
    }


    pub fn add_other_in_place(&mut self, other: &RotArr<D, N>) {
        for i in 0..N {
            let i_self = index_for_i(self.pov_i, i, N);
            self.data[i_self] = self.data[i_self] + other.get_element_for_perspective(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_0() {
        let rot_arr = RotArr::new_from_0(0, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_0(1, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_0(2, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_0(3, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
    }

    #[test]
    fn test_rot_arr() {
        let rot_arr = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_perspective(1, [1, 2, 3, 0]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_perspective(2, [2, 3, 0, 1]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);

        let rot_arr = RotArr::new_from_perspective(3, [3, 0, 1, 2]);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
    }

    #[test]
    fn test_rot_arr_add() {
        let mut rot_arr = RotArr::new_from_perspective(0, [0, 1, 2, 3]);
        let other = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(1, [1, 2, 3, 0]);
        let other = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(2, [2, 3, 0, 1]);
        let other = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_perspective(1, [1, 2, 3, 0]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_perspective(2, [2, 3, 0, 1]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);

        let mut rot_arr = RotArr::new_from_perspective(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_perspective(3, [3, 0, 1, 2]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 2);
        assert_eq!(rot_arr.get_element_for_perspective(2), 4);
        assert_eq!(rot_arr.get_element_for_perspective(3), 6);
    }

    #[test]
    fn test_rotated_for_i() {
        let org_rot_arr = RotArr::new_from_perspective(0, [0, 1, 2, 3]);

        let rot_arr = org_rot_arr.rotate_to_perspective(0);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
        assert_eq!(rot_arr.extract(), [0, 1, 2, 3]);

        let rot_arr = org_rot_arr.rotate_to_perspective(1);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
        assert_eq!(rot_arr.extract(), [1, 2, 3, 0]);

        let rot_arr = org_rot_arr.rotate_to_perspective(2);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
        assert_eq!(rot_arr.extract(), [2, 3, 0, 1]);

        let rot_arr = org_rot_arr.rotate_to_perspective(3);

        assert_eq!(rot_arr.get_element_for_perspective(0), 0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 1);
        assert_eq!(rot_arr.get_element_for_perspective(2), 2);
        assert_eq!(rot_arr.get_element_for_perspective(3), 3);
        assert_eq!(rot_arr.extract(), [3, 0, 1, 2]);
    }

    #[test]
    fn test_rot_arr_div() {
        let rot_arr = RotArr::new_from_perspective(0, [0.0, 1.0, 2.0, 3.0]);

        let rot_arr = rot_arr / 2.0;

        assert_eq!(rot_arr.get_element_for_perspective(0), 0.0);
        assert_eq!(rot_arr.get_element_for_perspective(1), 0.5);
        assert_eq!(rot_arr.get_element_for_perspective(2), 1.0);
        assert_eq!(rot_arr.get_element_for_perspective(3), 1.5);
    }
}
