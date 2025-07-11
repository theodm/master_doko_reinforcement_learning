use crate::basic::team::DoTeam;
use crate::player::player_set::{DoPlayerSet, player_set_contains, player_set_len};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DoEndOfGameStats {
    pub winning_team: DoTeam,

    pub re_players: DoPlayerSet,

    pub is_solo: bool,

    pub player_eyes: [u32; 4],

    pub re_eyes: u32,
    pub kontra_eyes: u32,

    pub re_points: i32,
    pub kontra_points: i32,

    pub player_points: [i32; 4],

    //basic_winning_point_details: Option<DoBasicWinningPointsDetails>,
}

pub fn calculate_end_of_game_stats(
    re_players: DoPlayerSet,

    player_eyes: [u32; 4],
    player_num_tricks: [u32; 4]
) -> DoEndOfGameStats {
    let mut re_tricks = 0;
    let mut kontra_tricks = 0;

    let mut re_eyes = 0;
    let mut kontra_eyes = 0;

    // Wir rechnen die Punkte der Teams zusammen
    // und die Anzahl der Stiche, die sie gemacht haben.
    for player in 0..4 {
        if player_set_contains(re_players, player) {
            re_eyes += player_eyes[player];
            re_tricks += player_num_tricks[player];
        } else {
            kontra_eyes += player_eyes[player];
            kontra_tricks += player_num_tricks[player];
        }
    }

    // Wir bestimmen dann das Gewinner-Team.
    let winning_team = if re_eyes > kontra_eyes {
        DoTeam::Re
    } else {
        DoTeam::Kontra
    };

    // Ist das Spiel ein Solo?
    let is_solo = player_set_len(re_players) == 1;


    let winning_team_eyes = if winning_team == DoTeam::Re {
        re_eyes
    } else {
        kontra_eyes
    };
    let winning_team_tricks = if winning_team == DoTeam::Re {
        re_tricks
    } else {
        kontra_tricks
    };

    let mut winning_team_points = 0;

    if winning_team_eyes >= 120 {
        winning_team_points = winning_team_points + 1;
    }

    if winning_team_eyes >= 150 {
        winning_team_points = winning_team_points + 1;
    }

    if winning_team_eyes >= 180 {
        winning_team_points = winning_team_points + 1;
    }

    if winning_team_eyes >= 210 {
        winning_team_points = winning_team_points + 1;
    }

    if winning_team_tricks == 12 {
        winning_team_points = winning_team_points + 1;
    }

    let mut re_points = if winning_team == DoTeam::Re {
        winning_team_points
    } else {
        -winning_team_points
    };
    let mut kontra_points = if winning_team == DoTeam::Kontra {
        winning_team_points
    } else {
        -winning_team_points
    };

    if is_solo {
        re_points = re_points * 3;
    }

    let mut player_points = [0, 0, 0, 0];

    for player in 0..4 {
        if player_set_contains(re_players, player) {
            player_points[player] = re_points;
        } else {
            player_points[player] = kontra_points;
        }
    }

    return DoEndOfGameStats {
        winning_team: winning_team,

        re_players: re_players,

        is_solo: is_solo,

        player_eyes: player_eyes,

        re_eyes: re_eyes,
        kontra_eyes: kontra_eyes,

        re_points: re_points,
        kontra_points: kontra_points,

        player_points: player_points
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::player::player::{PLAYER_LEFT, PLAYER_RIGHT, PLAYER_TOP};
    use crate::player::player_set::player_set_create;

    #[test]
    fn test_calculate_end_of_game_stats_re_win() {
        let player_eyes = [0, 130, 110, 0];
        let player_num_tricks = [0, 7, 5, 0];

        let re_players = player_set_create(vec![PLAYER_LEFT, PLAYER_TOP]);

        let end_of_game_stats = calculate_end_of_game_stats(
            re_players,
            player_eyes,
            player_num_tricks
        );

        assert_eq!(end_of_game_stats.winning_team, DoTeam::Re);
        assert_eq!(end_of_game_stats.re_players, re_players);
        assert_eq!(end_of_game_stats.is_solo, false);
        assert_eq!(end_of_game_stats.player_eyes, [0, 130, 110, 0]);
        assert_eq!(end_of_game_stats.re_eyes, 240);
        assert_eq!(end_of_game_stats.kontra_eyes, 0);
        assert_eq!(end_of_game_stats.re_points, 5);
        assert_eq!(end_of_game_stats.kontra_points, -5);
        assert_eq!(end_of_game_stats.player_points, [-5, 5, 5, -5]);
    }

    #[test]
    fn test_calculate_end_of_game_stats_kontra_win() {
        let player_eyes = [130, 0, 0, 110];
        let player_num_tricks = [7, 0, 0, 5];

        let re_players = player_set_create(vec![PLAYER_LEFT, PLAYER_TOP]);

        let end_of_game_stats = calculate_end_of_game_stats(
            re_players,
            player_eyes,
            player_num_tricks
        );

        assert_eq!(end_of_game_stats.winning_team, DoTeam::Kontra);
        assert_eq!(end_of_game_stats.re_players, re_players);
        assert_eq!(end_of_game_stats.is_solo, false);
        assert_eq!(end_of_game_stats.player_eyes, [130, 0, 0, 110]);
        assert_eq!(end_of_game_stats.re_eyes, 0);
        assert_eq!(end_of_game_stats.kontra_eyes, 240);
        assert_eq!(end_of_game_stats.re_points, -5);
        assert_eq!(end_of_game_stats.kontra_points, 5);
        assert_eq!(end_of_game_stats.player_points, [5, -5, -5, 5]);
    }

    #[test]
    fn test_calculate_end_of_game_stats_kontra_win_in_solo() {
        let player_eyes = [130, 50, 60, 0];
        let player_num_tricks = [7, 2, 3, 0];

        let re_players = player_set_create(vec![PLAYER_RIGHT]);

        let end_of_game_stats = calculate_end_of_game_stats(
            re_players,
            player_eyes,
            player_num_tricks
        );

        assert_eq!(end_of_game_stats.winning_team, DoTeam::Kontra);
        assert_eq!(end_of_game_stats.re_players, re_players);
        assert_eq!(end_of_game_stats.is_solo, true);
        assert_eq!(end_of_game_stats.player_eyes, [130, 50, 60, 0]);
        assert_eq!(end_of_game_stats.re_eyes, 0);
        assert_eq!(end_of_game_stats.kontra_eyes, 240);
        assert_eq!(end_of_game_stats.re_points, -15);
        assert_eq!(end_of_game_stats.kontra_points, 5);
        assert_eq!(end_of_game_stats.player_points, [5, 5, 5, -15]);
    }

    #[test]
    fn test_calculate_end_of_game_stats_re_win_in_solo() {
        let player_eyes = [0, 0, 0, 240];
        let player_num_tricks = [0, 0, 0, 12];

        let re_players = player_set_create(vec![PLAYER_RIGHT]);

        let end_of_game_stats = calculate_end_of_game_stats(
            re_players,
            player_eyes,
            player_num_tricks
        );

        assert_eq!(end_of_game_stats.winning_team, DoTeam::Re);
        assert_eq!(end_of_game_stats.re_players, re_players);
        assert_eq!(end_of_game_stats.is_solo, true);
        assert_eq!(end_of_game_stats.player_eyes, [0, 0, 0, 240]);
        assert_eq!(end_of_game_stats.re_eyes, 240);
        assert_eq!(end_of_game_stats.kontra_eyes, 0);
        assert_eq!(end_of_game_stats.re_points, 15);
        assert_eq!(end_of_game_stats.kontra_points, -5);
        assert_eq!(end_of_game_stats.player_points, [-5, -5, -5, 15]);
    }



}