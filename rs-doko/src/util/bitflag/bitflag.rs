use std::mem::transmute;
use std::ops::{BitAnd, BitOr};
use num_traits::PrimInt;
use rand::Rng;
use rand::rngs::SmallRng;
use crate::util::bitflag::select_rank::select_by_rank;

fn debug_assert_bit<T, const Size: usize>(bit: T)
where
    T: PrimInt,
{
    debug_assert!(bit > T::zero());
    debug_assert!(bit <= T::one() << Size);
}

fn debug_assert_bitflag<T, const Size: usize>(bitflag: T)
where
    T: PrimInt,
{
    // ToDo: Stattdessen 111111 verwenden
    debug_assert!(bitflag <= T::one() << (Size + 1));
}

pub fn bitflag_contains<
    T,
    const Size: usize
>(bitflag: T, bit: T) -> bool
where
    T: PrimInt + BitAnd<Output = T> + PartialEq,
{
    debug_assert_bit::<T, Size>(bit);
    debug_assert_bitflag::<T, Size>(bitflag);

    bitflag & bit == bit
}

pub fn bitflag_add<
    T,
    const Size: usize
>(bitflag: T, bit: T) -> T
where
    T: PrimInt + BitOr<Output = T>,
{
    debug_assert_bit::<T, Size>(bit);
    debug_assert_bitflag::<T, Size>(bitflag);

    bitflag | bit
}

pub fn bitflag_remove<
    T,
    const Size: usize
>(bitflag: T, bit: T) -> T
where
    T: PrimInt + BitAnd<Output = T>,
{
    debug_assert_bit::<T, Size>(bit);
    debug_assert_bitflag::<T, Size>(bitflag);

    bitflag & !bit
}

pub fn bitflag_number_of_ones<
    T,
    const Size: usize
>(
    bitflag: T
) -> u32
where
    T: PrimInt,
{
    debug_assert_bitflag::<T, Size>(bitflag);

    bitflag.count_ones()
}

pub fn bitflag_to_vec<
    T,
    const Size: usize
>(bitflag: T) -> heapless::Vec<T, Size>
where
    T: PrimInt,
{
    let mut vec = heapless::Vec::new();
    let mut bit: T = T::one();

    while bit <= bitflag {
        if bitflag_contains::<T, Size>(bitflag, bit) {
            vec
                .push(bit);
        }

        bit = bit << 1;
    }

    vec
}


pub fn bitflag_random_single(
    bitflag: u64,
    mut random: &mut SmallRng
) -> u64 {
   let index = random.gen_range(0..bitflag.count_ones());

    return select_by_rank(bitflag, index as u64);
}


pub fn bitflag_random_single_slow<
    T,
    const Size: usize
>(
    bitflag: T,
    mut random: &mut SmallRng
) -> T
where
    T: PrimInt
{
    let vec = bitflag_to_vec::<T, Size>(bitflag);

    if vec.is_empty() {
        panic!("bitflag_random_single: bitflag is empty");
    }

    let index = random.gen_range(0..vec.len());

    vec[index]
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use strum::EnumCount;
    use crate::action::action::DoAction;
    use super::*;

    #[test]
    fn test_bitflag_contains() {
        assert!(bitflag_contains::<usize, 4>(0b1001, 0b0001));
        assert!(bitflag_contains::<usize, 4>(0b0010, 0b0010));
        assert!(bitflag_contains::<usize, 4>(0b0110, 0b0100));
        assert!(bitflag_contains::<usize, 4>(0b1001, 0b1000));

        assert!(!bitflag_contains::<usize, 4>(0b0101, 0b0010));
        assert!(!bitflag_contains::<usize, 4>(0b0011, 0b0100));
        assert!(!bitflag_contains::<usize, 4>(0b0101, 0b1000));
        assert!(!bitflag_contains::<usize, 4>(0b1110, 0b0001));
    }

    #[test]
    fn test_bitflag_add() {
        assert_eq!(bitflag_add::<usize, 4>(0b1001, 0b0001), 0b1001);
        assert_eq!(bitflag_add::<usize, 4>(0b0010, 0b0001), 0b0011);
    }

    #[test]
    fn test_bitflag_remove() {
        assert_eq!(bitflag_remove::<usize, 4>(0b1001, 0b0001), 0b1000);
        assert_eq!(bitflag_remove::<usize, 4>(0b0011, 0b0001), 0b0010);
    }

    #[test]
    fn test_bitflag_number_of_ones() {
        assert_eq!(bitflag_number_of_ones::<usize, 4>(0b1001), 2);
        assert_eq!(bitflag_number_of_ones::<usize, 4>(0b0010), 1);
        assert_eq!(bitflag_number_of_ones::<usize, 4>(0b0000), 0);
    }

    #[test]
    fn test_bitflag_to_vec() {
        assert_eq!(bitflag_to_vec::<usize, 4>(0b1001).to_vec(),  vec![0b0001, 0b1000]);
        assert_eq!(bitflag_to_vec::<usize, 4>(0b0010).to_vec(), vec![0b0010]);

        assert_eq!(bitflag_to_vec::<usize, { DoAction::COUNT }>(540337330).to_vec(), vec![
            DoAction::CardDiamondTen as usize,
            DoAction::CardDiamondKing as usize,
            DoAction::CardHeartAce as usize,
            DoAction::CardHeartTen as usize,
            DoAction::CardHeartKing as usize,
            DoAction::CardClubTen as usize,
            DoAction::CardClubJack as usize,
            DoAction::CardClubQueen as usize,
            DoAction::CardSpadeNine as usize,
            DoAction::CardSpadeJack as usize,
            DoAction::CardSpadeQueen as usize,
        ])
    }

    #[test]
    fn test_bitflag_random_single() {
        let mut rng = SmallRng::seed_from_u64(0);
        let bitflag: usize = 0b1001;

        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b1000);
        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b1000);
        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b1000);
        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b1000);
        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b1000);
        assert_eq!(bitflag_random_single_slow::<usize, 4>(bitflag, &mut rng), 0b0001);

        assert_eq!(bitflag_random_single(bitflag as u64, &mut rng), 0b0001);
        assert_eq!(bitflag_random_single(bitflag as u64, &mut rng), 0b0001);
        assert_eq!(bitflag_random_single(bitflag as u64, &mut rng), 0b0001);
        assert_eq!(bitflag_random_single(bitflag as u64, &mut rng), 0b1000);
    }
}
