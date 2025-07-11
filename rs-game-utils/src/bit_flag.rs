use std::ops::{BitAnd, BitOr, Not};
use num_traits::{Num, Zero};
use num_traits::One;
use rand::prelude::SmallRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

macro_rules! assert_valid_bit {
    ($bit:expr, $size:expr, $type:ty) => {
        debug_assert!($bit > <$type>::zero(), "Bit must be greater than zero.");
        debug_assert!($bit <= <$type>::one() << $size, "Bit exceeds maximum size.");
    };
}
macro_rules! assert_valid_bitflag {
    ($bitflag:expr, $size:expr, $type:ty) => {
        debug_assert!($bitflag <= <$type>::one() << ($size + 1), "Bitflag exceeds maximum size.");
    };
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitflag<const NumberOfEntries: usize>(pub u64);


impl<const NumberOfEntries: usize> Bitflag<NumberOfEntries> {
    pub fn new() -> Self {
        Bitflag(0)
    }

    pub fn inverse(&self) -> Bitflag<NumberOfEntries> {
        Bitflag(!self.0 & ((1 << NumberOfEntries) - 1) as u64)
    }

    /// Prüft, ob ein bestimmtes Bit im Bitflag enthalten ist.
    pub fn contains(&self, bit: u64) -> bool {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);
        assert_valid_bit!(bit, NumberOfEntries, u64);

        self.0 & bit == bit
    }

    /// Fügt ein Bit zum Bitflag hinzu. Überschreibt das Bitflag, falls das Bit bereits enthalten ist.
    pub fn add(&mut self, bit: u64) {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);
        assert_valid_bit!(bit, NumberOfEntries, u64);

        self.0 = self.0 | bit;
    }


    /// Entfernt ein Bit aus dem Bitflag.
    pub fn remove(&mut self, bit: u64) {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);
        assert_valid_bit!(bit, NumberOfEntries, u64);

        self.0 = self.0 & !bit;
    }


    /// Gibt die Anzahl der gesetzten Bits im Bitflag zurück.
    pub fn number_of_ones(&self) -> u32 {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);

        self.0.count_ones()
    }

    /// Gibt die gesetzten Bits als Vektor zurück.
    pub fn to_vec(&self) -> heapless::Vec<u64, NumberOfEntries> {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);

        let mut vec = heapless::Vec::new();
        let mut bit = 1;

        while bit <= self.0 {
            if self.contains(bit) {
                vec.push(bit).unwrap();
            }

            bit = bit << 1;
        }

        vec
    }


    /// Wählt ein zufälliges gesetztes Bit aus.
    pub fn random_single(&self, random: &mut SmallRng) -> u64 {
        assert_valid_bitflag!(self.0, NumberOfEntries, u64);

        let index = random.gen_range(0..self.number_of_ones());

        select_by_rank(self.0, index as u64)
    }

}

/// Wählt das Bit mit einem bestimmten Rang aus einer gegebenen Zahl aus.
///
/// # Parameter
/// - `v`: Die Eingabezahl, aus der ein Bit ausgewählt werden soll.
/// - `r`: Der Rang des gewünschten Bits (0-basiert).
///
/// # Rückgabewert
/// Gibt das Bit mit dem angegebenen Rang zurück.
fn select_by_rank(
    v: u64,

    r: u64,
) -> u64 {
    // http://graphics.stanford.edu/~seander/bithacks.html#SelectPosFromMSBRank

    // Output: Resulting position of bit with rank r [1-64]
    let mut s: u64;

    // Bit count temporary.
    let mut t;

    let mut a;
    let mut b;
    let mut c;
    let mut d;

    let mut v = v;
    let mut r = r + 1;

    a =  v - ((v >> 1) & (!0) / 3);

    // b = (a & 0x3333...) + ((a >> 2) & 0x3333...);
    b = (a & !0 / 5) + ((a >> 2) & !0 / 5);

    // c = (b & 0x0f0f...) + ((b >> 4) & 0x0f0f...);
    c = (b + (b >> 4)) & !0 / 0x11;

    // d = (c & 0x00ff...) + ((c >> 8) & 0x00ff...);
    d = (c + (c >> 8)) & !0 / 0x101;

    t = (d >> 32) + (d >> 48);

    // Now do branchless select!
    s  = 64;

    // if (r > t) {s -= 32; r -= t;}
    s -= (u64::wrapping_sub(t, r) & 256) >> 3;
    r -= (t & (u64::wrapping_sub(t, r) >> 8));
    t  = (d >> u64::wrapping_sub(s, 16)) & 0xff;

    // if (r > t) {s -= 16; r -= t;}
    s -= (u64::wrapping_sub(t, r) & 256) >> 4;
    r -= (t & (u64::wrapping_sub(t, r) >> 8));
    t  = (c >> u64::wrapping_sub(s, 8)) & 0xf;

    // if (r > t) {s -= 8; r -= t;}
    s -= (u64::wrapping_sub(t, r) & 256) >> 5;
    r -= (t & (u64::wrapping_sub(t, r) >> 8));
    t  = (b >> u64::wrapping_sub(s, 4)) & 0x7;

    // if (r > t) {s -= 4; r -= t;}
    s -= (u64::wrapping_sub(t, r) & 256) >> 6;
    r -= (t & (u64::wrapping_sub(t, r) >> 8));
    t  = (a >> u64::wrapping_sub(s, 2)) & 0x3;

    // if (r > t) {s -= 2; r -= t;}
    s -= (u64::wrapping_sub(t, r) & 256) >> 7;
    r -= (t & (u64::wrapping_sub(t, r) >> 8));
    t  = (v >> u64::wrapping_sub(s, 1)) & 0x1;

    // if (r > t) s--;
    s -= (u64::wrapping_sub(t, r) & 256) >> 8;
    //s = 65 - s;

    1 << (s-1)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn test_inverse() {
        let bitflag = Bitflag::<8>(0b10101010);

        let inverse = bitflag.inverse();
        assert_eq!(inverse.0, 0b01010101);
    }

    #[test]
    fn test_select_by_rank() {
        assert_eq!(select_by_rank(0b10101010, 0), 0b10000000);
        assert_eq!(select_by_rank(0b10101010, 1), 0b00100000);
        assert_eq!(select_by_rank(0b10101010, 2), 0b00001000);
        assert_eq!(select_by_rank(0b10101010, 3), 0b00000010);
    }

    #[test]
    fn test_bitflag_add_contains_remove() {
        let mut bitflag: Bitflag<8> = Bitflag::new();

        bitflag.add(0b00000010);
        assert!(bitflag.contains(0b00000010));

        bitflag.add(0b00000100);
        assert!(bitflag.contains(0b00000100));
        assert_eq!(bitflag.0, 0b00000110);

        bitflag.remove(0b00000010);
        assert!(!bitflag.contains(0b00000010));
        assert_eq!(bitflag.0, 0b00000100);
    }


    fn test_number_of_ones() {
        let mut bitflag: Bitflag<8> = Bitflag::new();

        bitflag.add(0b00000010);
        bitflag.add(0b00000100);

        assert_eq!(bitflag.number_of_ones(), 2);
    }

    #[test]
    fn test_to_vec() {
        let mut bitflag: Bitflag<8> = Bitflag::new();

        bitflag.add(0b00000010);
        bitflag.add(0b00000100);
        let vec = bitflag.to_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&0b00000010));
        assert!(vec.contains(&0b00000100));
    }

    #[test]
    fn test_random_single() {
        let mut bitflag: Bitflag<8> = Bitflag::new();
        bitflag.add(0b00000010);
        bitflag.add(0b00000100);

        let mut rng = SmallRng::seed_from_u64(42);
        let random_bit = bitflag.random_single(&mut rng);

        assert!(random_bit == 0b00000010 || random_bit == 0b00000100);
    }
}