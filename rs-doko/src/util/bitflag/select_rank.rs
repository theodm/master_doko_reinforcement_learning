
pub fn select_by_rank(
    v: u64,

    r: u64,


) -> u64 {
    // http://graphics.stanford.edu/~seander/bithacks.html#SelectPosFromMSBRank

    let mut s: u64;

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
    use super::*;

    #[test]
    fn test_select_by_rank() {
        assert_eq!(select_by_rank(0b10101010, 0), 0b10000000);
        assert_eq!(select_by_rank(0b10101010, 1), 0b00100000);
        assert_eq!(select_by_rank(0b10101010, 2), 0b00001000);
        assert_eq!(select_by_rank(0b10101010, 3), 0b00000010);

    }
}