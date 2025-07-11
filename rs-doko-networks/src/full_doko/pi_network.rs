// use crate::full_doko::var1::announcement::{
//     ANNOUNCEMENT_OR_NONE_COUNT, NUM_CARD_POSITIONS, NUM_POSTIONS_IN_CARD_POSITION,
// };
// use crate::full_doko::var1::card::CARD_OR_NONE_COUNT;
// use crate::full_doko::var1::game_type::GAME_TYPE_OR_NONE_COUNT;
// use crate::full_doko::var1::phase::PHASE_COUNT;
// use crate::full_doko::var1::player::PLAYER_OR_NONE_COUNT;
// use crate::full_doko::var1::reservation::RESERVATION_OR_NONE_COUNT;
// use rs_full_doko::action::action::FdoAction;
// use strum::EnumCount;
// use tch::nn::{Module, VarStore};
// use tch::{nn, Device, IndexOp, Kind, Tensor};
// use crate::full_doko::utils::create_mlp_without_out;
//
// unsafe impl Sync for FullDokoPerfectInformationNetwork {}
// #[derive(Debug)]
// pub struct FullDokoPerfectInformationNetwork {
//     pub var_store: VarStore,
//     pub config: FullDokoPerfectInformationNetworkConfiguration,
//
//     hands_layers: nn::Sequential,
//     trick_layers: nn::Sequential,
//
//     pub phase_embeddings: nn::Embedding,
//     pub player_input_embeddings: nn::Embedding,
//     pub reservations_embeddings: nn::Embedding,
//     pub announcement_embeddings: nn::Embedding,
//     pub game_type_embeddings: nn::Embedding,
//     pub card_embeddings: nn::Embedding,
//
//     // Geteilte Schichten
//     shared: nn::Sequential,
//
//     // Policy head
//     policy_head: nn::Sequential,
//
//     // Value head
//     value_head: nn::Sequential,
// }
//
// #[derive(Debug)]
// pub struct FullDokoPerfectInformationNetworkConfiguration {
//     shared_layers: &'static [i64],
//
//     policy_head_layers: &'static [i64],
//     value_head_layers: &'static [i64],
//
//     trick_layers: &'static [i64],
//     hands_layers: &'static [i64],
//
//     phase_embeddings_dim_size: i64,
//     player_input_embeddings_dim_size: i64,
//     reservations_embeddings_dim_size: i64,
//     announcement_embeddings_dim_size: i64,
//     game_type_embeddings_dim_size: i64,
//     card_embeddings_dim_size: i64,
//
//     device: Device,
// }
//
// fn create_mlp(vs: &nn::Path, in_dim: i64, hidden_dims: &[i64], out_dim: i64) -> nn::Sequential {
//     let mut net = nn::seq();
//     let mut current_in_dim = in_dim;
//
//     // Hidden layers
//     for (i, &dim) in hidden_dims.iter().enumerate() {
//         let linear = nn::linear(
//             vs / format!("layer_{i}"),
//             current_in_dim,
//             dim,
//             Default::default(),
//         );
//         net = net.add(linear).add_fn(|x| x.relu());
//         current_in_dim = dim;
//     }
//
//     let final_linear = nn::linear(
//         vs / "final_out",
//         current_in_dim,
//         out_dim,
//         Default::default(),
//     );
//     net.add(final_linear)
// }
//
// impl FullDokoPerfectInformationNetwork {
//     pub fn new(
//         config: FullDokoPerfectInformationNetworkConfiguration,
//     ) -> FullDokoPerfectInformationNetwork {
//         let var_store = nn::VarStore::new(config.device);
//         let vs = &var_store.root();
//
//         let phase_embeddings = nn::embedding(
//             vs / "phase_embeddings",
//             PHASE_COUNT as i64,
//             config.phase_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let player_input_embeddings = nn::embedding(
//             vs / "player_input_embeddings",
//             PLAYER_OR_NONE_COUNT as i64,
//             config.player_input_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let reservations_embeddings = nn::embedding(
//             vs / "reservations_embeddings",
//             RESERVATION_OR_NONE_COUNT as i64,
//             config.reservations_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let announcement_embeddings = nn::embedding(
//             vs / "announcement_embeddings",
//             ANNOUNCEMENT_OR_NONE_COUNT as i64,
//             config.announcement_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let game_type_embeddings = nn::embedding(
//             vs / "game_type_embeddings",
//             GAME_TYPE_OR_NONE_COUNT,
//             config.game_type_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let card_embeddings = nn::embedding(
//             vs / "card_embeddings",
//             CARD_OR_NONE_COUNT,
//             config.card_embeddings_dim_size,
//             Default::default(),
//         );
//
//         let trick_layers = create_mlp_without_out(
//             &(vs / "trick"),
//             1 * config.player_input_embeddings_dim_size + 4 * config.card_embeddings_dim_size,
//             config.trick_layers
//         );
//
//         let hands_layers = create_mlp_without_out(
//             &(vs / "hands"),
//             48,
//             config.hands_layers
//         );
//
//         let output_size = config.phase_embeddings_dim_size * 1
//             + config.player_input_embeddings_dim_size * 23
//             + 10
//             + 10
//             + config.announcement_embeddings_dim_size * 2
//             + config.game_type_embeddings_dim_size * 1
//             + config.reservations_embeddings_dim_size * 4
//             + config.card_embeddings_dim_size * 96;
//
//         println!(
//             "Embedding concatenation size (input to first dense): {}",
//             output_size
//         );
//         let shared = create_mlp(
//             &(vs / "shared"),
//             output_size,
//             config.shared_layers,
//             if config.shared_layers.is_empty() {
//                 output_size
//             } else {
//                 *config.shared_layers.last().unwrap()
//             },
//         );
//
//         let final_shared_dim = if config.shared_layers.is_empty() {
//             output_size
//         } else {
//             *config.shared_layers.last().unwrap()
//         };
//
//         let policy_head = create_mlp(
//             &(vs / "policy_head"),
//             final_shared_dim,
//             config.policy_head_layers,
//             FdoAction::COUNT as i64,
//         );
//
//         let value_head = create_mlp(
//             &(vs / "value_head"),
//             final_shared_dim,
//             config.value_head_layers,
//             4,
//         );
//
//         return FullDokoPerfectInformationNetwork {
//             var_store,
//             config,
//
//             hands_layers,
//             trick_layers,
//
//             phase_embeddings,
//             player_input_embeddings,
//             reservations_embeddings,
//             announcement_embeddings,
//             game_type_embeddings,
//             card_embeddings,
//
//             shared,
//             policy_head,
//             value_head,
//         };
//     }
//
//     pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
//         let shape = x.size();
//         let (batch_size, total_dim) = match shape.as_slice() {
//             &[dim] => (1, dim),      
//             &[bs, dim] => (bs, dim), 
//             other => panic!("Unexpected input shape: {:?}", other),
//         };
//         debug_assert_eq!(total_dim, 147, "Input must have exactly 147 features.");
//
//         let x = if batch_size == 1 && shape.len() == 1 {
//             x.view([1, 147])
//         } else {
//             x.shallow_clone()
//         };
//
//         let phase = x.i((.., 0..1)); // shape: [batch, 1]
//         debug_assert_eq!(phase.size()[1], 1, "phase slice must be of length 1.");
//
//         let all_player_encodings = x.i((.., 1..24)); // shape: [batch, 23]
//         debug_assert_eq!(
//             all_player_encodings.size()[1],
//             23,
//             "all_player_encodings slice must be of length 23."
//         );
//
//         let announcement_position_encodings = x.i((.., 24..34)); // shape: [batch, 10]
//         debug_assert_eq!(
//             announcement_position_encodings.size()[1],
//             10,
//             "announcement_position_encodings slice must be of length 10."
//         );
//
//         let announcement_position_in_position_encodings = x.i((.., 34..44)); // [batch, 10]
//         debug_assert_eq!(
//             announcement_position_in_position_encodings.size()[1],
//             10,
//             "announcement_position_in_position_encodings slice must be length 10."
//         );
//
//         let lowest_announcements_encodings = x.i((.., 44..46)); // shape: [batch, 2]
//         debug_assert_eq!(
//             lowest_announcements_encodings.size()[1],
//             2,
//             "lowest_announcements_encodings slice must be of length 2."
//         );
//
//         let game_type_encoding = x.i((.., 46..47)); // shape: [batch, 1]
//         debug_assert_eq!(
//             game_type_encoding.size()[1],
//             1,
//             "game_type_encoding slice must be of length 1."
//         );
//
//         let reservation_encodings = x.i((.., 47..51)); // shape: [batch, 4]
//         debug_assert_eq!(
//             reservation_encodings.size()[1],
//             4,
//             "reservation_encodings slice must be of length 4."
//         );
//
//         let card_encodings = x.i((.., 51..147)); // shape: [batch, 96]
//         debug_assert_eq!(
//             card_encodings.size()[1],
//             96,
//             "card_encodings slice must be of length 96."
//         );
//
//         let phase_e = self.phase_embeddings.forward(&phase); // [batch, 1, phase_emb]
//         let players_e = self.player_input_embeddings.forward(&all_player_encodings);
//         let lowest_ann_e = self
//             .announcement_embeddings
//             .forward(&lowest_announcements_encodings);
//         let game_type_e = self.game_type_embeddings.forward(&game_type_encoding);
//         let reservations_e = self.reservations_embeddings.forward(&reservation_encodings);
//         let cards_e = self.card_embeddings.forward(&card_encodings);
//
//         let announcement_positions_f =
//             announcement_position_encodings.to_kind(Kind::Float) / (NUM_CARD_POSITIONS as f64);
//
//         let announcement_positions_in_pos_f = announcement_position_in_position_encodings
//             .to_kind(Kind::Float)
//             / (NUM_POSTIONS_IN_CARD_POSITION as f64);
//
//         let phase_e = phase_e.flatten(1, -1); // [batch, 1*phase_emb]
//         let players_e = players_e.flatten(1, -1); // [batch, 23*player_emb]
//         let lowest_ann_e = lowest_ann_e.flatten(1, -1);
//         let game_type_e = game_type_e.flatten(1, -1);
//         let reservations_e = reservations_e.flatten(1, -1);
//         let cards_e = cards_e.flatten(1, -1);
//
//         let concat_in = Tensor::cat(
//             &[
//                 phase_e,
//                 players_e,
//                 announcement_positions_f,        // 10 columns
//                 announcement_positions_in_pos_f, // 10 columns
//                 lowest_ann_e,
//                 game_type_e,
//                 reservations_e,
//                 cards_e,
//             ],
//             1,
//         );
//
//         let hidden = self.shared.forward(&concat_in);
//         let policy = self.policy_head.forward(&hidden);
//         let value = self.value_head.forward(&hidden);
//
//         (policy, value)
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use crate::full_doko::pi_network::{
//         FullDokoPerfectInformationNetwork, FullDokoPerfectInformationNetworkConfiguration,
//     };
//     use crate::full_doko::var1::encode_pi::encode_state_pi;
//     use rand::rngs::SmallRng;
//     use rand::SeedableRng;
//     use rs_full_doko::action::action::FdoAction;
//     use rs_full_doko::basic::phase::FdoPhase;
//     use rs_full_doko::state::state::FdoState;
//     use strum::EnumCount;
//     use tch::{nn, Device};
//
//     #[test]
//     pub fn play_full_game() {
//         let mut rng = SmallRng::from_os_rng();
//
//         let network = FullDokoPerfectInformationNetwork::new(
//             FullDokoPerfectInformationNetworkConfiguration {
//                 shared_layers: &[128, 64],
//                 policy_head_layers: &[64],
//                 value_head_layers: &[64],
//
//                 trick_layers: &[64],
//                 hands_layers: &[64],
//
//                 phase_embeddings_dim_size: 2,
//                 player_input_embeddings_dim_size: 2,
//                 reservations_embeddings_dim_size: 4,
//                 announcement_embeddings_dim_size: 4,
//                 game_type_embeddings_dim_size: 4,
//                 card_embeddings_dim_size: 6,
//                 device: Device::Cpu,
//             },
//         );
//
//         let mut state = FdoState::new_game(&mut rng);
//
//         loop {
//             let obs = state.observation_for_current_player();
//
//             if obs.phase == FdoPhase::Finished {
//                 break;
//             }
//
//             let x = encode_state_pi(&state, &obs);
//
//             let (policy, value) = network.forward(&tch::Tensor::from_slice(&x));
//
//             assert_eq!(policy.size(), &[1, FdoAction::COUNT as i64]);
//             assert_eq!(value.size(), &[1, 4]);
//
//             state.random_action_for_current_player(&mut rng);
//         }
//     }
// }
