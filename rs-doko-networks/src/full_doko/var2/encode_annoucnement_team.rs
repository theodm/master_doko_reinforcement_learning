use rs_full_doko::basic::team::FdoTeam;

pub fn encode_announcement_team(
    team: Option<FdoTeam>
) -> [i64; 1] {
    match team {
        None => [0],
        Some(team) => match team {
            FdoTeam::Re => [1],
            FdoTeam::Kontra => [2]
        }
    }
}