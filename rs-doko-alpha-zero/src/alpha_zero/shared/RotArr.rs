use std::ops;
use rand_distr::num_traits::Zero;
use std::ops::{Add, Div};

#[derive(Debug)]
pub struct RotArr<D: Copy + Add<Output = D> + Zero, const N: usize> {
    pub(crate) pov_i: usize,

    // Das Array, das von der Perspektive pov_i aus hier
    // gespeichert wird.
    data: [D; N],
}

pub fn index_for_i(pov_i: usize, i: usize, N: usize) -> usize {
    ((N - pov_i) + i) % N
}

impl<D: Copy + Add<Output = D> + Zero, const N: usize> RotArr<D, N> {
    pub fn zeros(
        pov_i: usize
    ) -> Self {
        RotArr {
            pov_i,
            data: [D::zero(); N],
        }
    }

    pub fn new_from_0(pov_i: usize, data_from_0: [D; N]) -> Self {
        let mut new_data: [D; N] = [D::zero(); N];

        for i in 0..N {
            new_data[index_for_i(pov_i, i, N)] = data_from_0[i];
        }

        RotArr {
            pov_i: pov_i,
            data: new_data,
        }
    }


    pub fn new_from_pov(pov_i: usize, data_from_pov: [D; N]) -> Self {
        RotArr {
            pov_i: pov_i,
            data: data_from_pov,
        }
    }

    pub fn extract(&self) -> [D; N] {
        self.data
    }

    pub fn rotated_for_i(&self, i: usize) -> RotArr<D, N> {
        let mut new_data = [D::zero(); N];

        for player in 0..N {
            new_data[player] = self.get_for_i(player);
        }

        return RotArr::new_from_0(i, new_data);
    }

    pub fn get_for_i(&self, i: usize) -> D {
        self.data[index_for_i(self.pov_i, i, N)]
    }

    pub fn from_pov_index(&self, i: usize) -> D {
        return self.data[i];
    }

    pub fn add_other_in_place(&mut self, other: &RotArr<D, N>) {
        for i in 0..N {
            let i_self = index_for_i(self.pov_i, i, N);
            self.data[i_self] = self.data[i_self] + other.get_for_i(i);
        }
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_0() {
        let rot_arr = RotArr::new_from_0(0, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_0(1, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_0(2, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_0(3, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
    }

    #[test]
    fn test_rot_arr() {
        let rot_arr = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_pov(1, [1, 2, 3, 0]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_pov(2, [2, 3, 0, 1]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);

        let rot_arr = RotArr::new_from_pov(3, [3, 0, 1, 2]);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
    }

    #[test]
    fn test_rot_arr_add() {
        let mut rot_arr = RotArr::new_from_pov(0, [0, 1, 2, 3]);
        let other = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(1, [1, 2, 3, 0]);
        let other = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(2, [2, 3, 0, 1]);
        let other = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_pov(1, [1, 2, 3, 0]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_pov(2, [2, 3, 0, 1]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);

        let mut rot_arr = RotArr::new_from_pov(3, [3, 0, 1, 2]);
        let other = RotArr::new_from_pov(3, [3, 0, 1, 2]);

        rot_arr.add_other_in_place(&other);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 2);
        assert_eq!(rot_arr.get_for_i(2), 4);
        assert_eq!(rot_arr.get_for_i(3), 6);
    }

    #[test]
    fn test_rotated_for_i() {
        let org_rot_arr = RotArr::new_from_pov(0, [0, 1, 2, 3]);

        let rot_arr = org_rot_arr.rotated_for_i(0);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
        assert_eq!(rot_arr.extract(), [0, 1, 2, 3]);

        let rot_arr = org_rot_arr.rotated_for_i(1);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
        assert_eq!(rot_arr.extract(), [1, 2, 3, 0]);

        let rot_arr = org_rot_arr.rotated_for_i(2);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
        assert_eq!(rot_arr.extract(), [2, 3, 0, 1]);

        let rot_arr = org_rot_arr.rotated_for_i(3);

        assert_eq!(rot_arr.get_for_i(0), 0);
        assert_eq!(rot_arr.get_for_i(1), 1);
        assert_eq!(rot_arr.get_for_i(2), 2);
        assert_eq!(rot_arr.get_for_i(3), 3);
        assert_eq!(rot_arr.extract(), [3, 0, 1, 2]);
    }

    #[test]
    fn test_rot_arr_div() {
        let rot_arr = RotArr::new_from_pov(0, [0.0, 1.0, 2.0, 3.0]);

        let rot_arr = rot_arr / 2.0;

        assert_eq!(rot_arr.get_for_i(0), 0.0);
        assert_eq!(rot_arr.get_for_i(1), 0.5);
        assert_eq!(rot_arr.get_for_i(2), 1.0);
        assert_eq!(rot_arr.get_for_i(3), 1.5);
    }
}
