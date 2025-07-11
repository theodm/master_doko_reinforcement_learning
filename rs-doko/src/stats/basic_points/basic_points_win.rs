pub struct DoBasicWinningPointsDetails {
    // 7.1.4 DKV-TR
    // (a)
    winning_points: i32,
    winning_under_90: i32,
    winning_under_60: i32,
    winning_under_30: i32,
    winning_black: i32,

    // (b)
    re_announcement: i32,
    kontra_announcement: i32,

    // (c)
    re_under_90_announcement: i32,
    re_under_60_announcement: i32,
    re_under_30_announcement: i32,
    re_black_announcement: i32,

    // (d)
    kontra_under_90_announcement: i32,
    kontra_under_60_announcement: i32,
    kontra_under_30_announcement: i32,
    kontra_black_announcement: i32,

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
