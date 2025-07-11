
pub struct DoBasicDrawPointsDetails {
    // 7.1.4 DKV-TR
    // (a)
    winning_under_90_re: i32,
    winning_under_60_re: i32,
    winning_under_30_re: i32,

    winning_under_90_kontra: i32,
    winning_under_60_kontra: i32,
    winning_under_30_kontra: i32,

    // (e)
    re_reached_120_against_no_90: i32,
    re_reached_90_against_no_60: i32,
    re_reached_60_against_no_30: i32,
    re_reached_30_against_black: i32,

    // (f)
    kontra_reached_120_against_no_90: i32,
    kontra_reached_90_against_no_60: i32,
    kontra_reached_60_against_no_30: i32,
    kontra_reached_30_against_black: i32,
}