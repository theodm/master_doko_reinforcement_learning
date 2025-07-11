use crate::alpha_zero::utils::rot_arr::RotArr;

pub fn text_color_for_player(
    player: usize
) -> String {
    return match player {
        // Rot
        0 => "#ef4444".to_string(),
        // Grün
        1 => "#22c55e".to_string(),
        // Blau
        2 => "#3b82f6".to_string(),
        // Orange,
        3 => "#f97316".to_string(),
        _ => panic!("More than 4 players not supported")
    };
}

pub fn color_label(
    label: &str,
    color: &str
) -> String {
    return format!("<FONT COLOR=\"{}\">{}</FONT>", color, label);
}

pub fn color_label_for_player(
    label: &str,
    player: usize
) -> String {
    return color_label(label, text_color_for_player(player).as_str());
}

pub fn pretty_print_values_array<
    const N: usize
>(
    player_array: RotArr<f32, N>
) -> String {
    let mut result = String::new();

    result.push_str("[");

    // ToDo: Schöner
    for i in player_array.pov_i .. N + player_array.pov_i {
        let current_player = i % N;

        let value_for_current_player = format!("{:.2}", player_array.get_for_i(current_player));

        result.push_str(
            color_label_for_player(
                &value_for_current_player,
                current_player
            ).as_str()
        );

        if i < N + player_array.pov_i - 1 {
            result.push_str(", ");
        }
    }

    result.push_str("]");

    return result;
}