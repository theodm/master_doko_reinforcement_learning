use strum::EnumCount;
use crate::full_doko::var1::ipi_output::ImperfectInformationOutput;

// use crate::full_doko::var1::announcement::{
//     ANNOUNCEMENT_OR_NONE_COUNT, NUM_CARD_POSITIONS, NUM_POSTIONS_IN_CARD_POSITION,
// };
// use crate::full_doko::var1::card::CARD_OR_NONE_COUNT;
// use crate::full_doko::var1::game_type::GAME_TYPE_OR_NONE_COUNT;
// use crate::full_doko::var1::phase::PHASE_COUNT;
// use crate::full_doko::var1::player::PLAYER_OR_NONE_COUNT;
// use strum::EnumCount;
// use tch::nn::{Module, ModuleT, VarStore};
// use tch::{nn, Device, IndexOp, Kind, Tensor};
// use crate::full_doko::var1::encode_ipi::NUM_MAX_CARDS_ON_HAND;
// use crate::full_doko::var1::ipi_output::ImperfectInformationOutput;
// use crate::full_doko::var1::visible_reservation::VISIBLE_RESERVATION_OR_NONE_COUNT;
//
pub const FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE: usize = 311;
pub const FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE: usize = ImperfectInformationOutput::COUNT as usize;
// unsafe impl Sync for FullDokoImperfectInformationNetwork {}
//
// macro_rules! debug_println {
//     ($($arg:tt)*) => {
//         if cfg!(debug_assertions) {
//             println!($($arg)*);
//         }
//     };
// }
//
// #[derive(Debug)]
// pub struct FullDokoImperfectInformationNetwork {
//     pub var_store: VarStore,
//
//     pub phase_embeddings: nn::Embedding,
//     pub player_input_embeddings: nn::Embedding,
//     pub visible_reservations_embeddings: nn::Embedding,
//     pub announcement_embeddings: nn::Embedding,
//     pub game_type_embeddings: nn::Embedding,
//     pub card_embeddings: nn::Embedding,
//
//     fc_layers: nn::SequentialT,
//
//     hands_layers: nn::SequentialT,
//     trick_layers: nn::SequentialT,
//
//     settings: FullDokoImperfectInformationNetworkConfiguration,
// }
//
// impl FullDokoImperfectInformationNetwork {
//     pub fn clone_network(&self) -> FullDokoImperfectInformationNetwork {
//         let mut new_network = FullDokoImperfectInformationNetwork::new(self
//             .settings
//             .clone()
//         );
//
//         new_network
//             .var_store
//             .copy(&self.var_store)
//             .unwrap();
//
//         new_network
//     }
// }
//
// #[derive(Debug, Clone)]
// pub struct FullDokoImperfectInformationNetworkConfiguration {
//     pub fc_layers: &'static [i64],
//     pub hands_layers: &'static [i64],
//     pub trick_layers: &'static [i64],
//
//     pub phase_embeddings_dim_size: i64,
//     pub player_input_embeddings_dim_size: i64,
//     pub visible_reservations_embeddings_dim_size: i64,
//     pub announcement_embeddings_dim_size: i64,
//     pub game_type_embeddings_dim_size: i64,
//     pub card_embeddings_dim_size: i64,
//
//     pub device: Device,
// }
//
// fn create_mlp(vs: &nn::Path,
//               in_dim: i64,
//               hidden_dims: &[i64],
//               out_dim: i64
// ) -> nn::Sequential {
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
//         net = net
//             .add(linear)
//             .add_fn(|x| x.relu());
//         current_in_dim = dim;
//     }
//
//     // Final layer to out_dim
//     let final_linear = nn::linear(
//         vs / "final_out",
//         current_in_dim,
//         out_dim,
//         Default::default(),
//     );
//     net.add(final_linear)
// }
//
//
// use tch::nn::{Linear, Path, Sequential};
// use crate::full_doko::res_block::{create_residual_backbone, ResidualBlock};
// use crate::full_doko::utils;
//
//
// impl FullDokoImperfectInformationNetwork {
//     pub fn new(
//         config: FullDokoImperfectInformationNetworkConfiguration,
//     ) -> FullDokoImperfectInformationNetwork {
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
//         let visible_reservations_embeddings = nn::embedding(
//             vs / "reservations_embeddings",
//             VISIBLE_RESERVATION_OR_NONE_COUNT as i64,
//             config.visible_reservations_embeddings_dim_size,
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
//         // println!("2");
//         let trick_layers = create_residual_backbone(
//             &(vs / "trick"),
//             1 * config.player_input_embeddings_dim_size + 8 * config.card_embeddings_dim_size,
//             config.trick_layers
//         );
//
//         let hands_layers = create_residual_backbone(
//             &(vs / "hands"),
//             48,
//             config.hands_layers
//         );
//
//         let output_size =
//             // 1 Phase geht durch die Embeddings
//             config.phase_embeddings_dim_size * 1
//             // 12 Spieler gehen durch die Embeddings
//             + config.player_input_embeddings_dim_size * 60
//             // 10 Positionen für Ansagen (nur skaliert)
//             + 10
//             // 10 Positionen in Position (nur skaliert)
//             + 10
//             // 2 Niedrigste Ansagen gehen durch die Embeddings
//             + config.announcement_embeddings_dim_size * 2
//             // 1 Spieltyp geht durch die Embeddings
//             + config.game_type_embeddings_dim_size * 1
//             // 7 Vorbehalte gehen durch die Embeddings
//             + config.visible_reservations_embeddings_dim_size * 7
//             // 6 Angaben über die Anzahl der Handkarten
//             + 6
//             // 4 Blätter gehen durch ihre eigenen layers
//             + config.hands_layers.last().unwrap() * 4
//             // 12 Spieler und 96 Karten gehen durch ihre eigenen layers
//             + config.trick_layers.last().unwrap() * 12
//             // 48 Karten
//             + 48;
//
//         // println!(
//         //     "Embedding concatenation size (input to first dense): {}",
//         //     output_size
//         // );
//
//         let (mut fc_layers, _) = create_residual_backbone(
//             &(vs / "shared"),
//
//             output_size,
//             config.fc_layers
//         );
//
//         // 2) Danach einen finalen Linear-Layer => z.B. Output = #ImperfectInformationOutput
//         fc_layers = fc_layers.add(nn::linear(
//             vs / "fc_final_out",
//             *config.fc_layers.last().unwrap(),  // letzte Hidden-Dim
//             ImperfectInformationOutput::COUNT as i64,
//             Default::default(),
//         ));
//
//         FullDokoImperfectInformationNetwork {
//             var_store,
//
//             phase_embeddings,
//             player_input_embeddings,
//             visible_reservations_embeddings,
//             announcement_embeddings,
//             game_type_embeddings,
//             card_embeddings,
//
//             fc_layers,
//             hands_layers: hands_layers.0,
//             trick_layers: trick_layers.0,
//
//             settings: config,
//         }
//     }
//
//     pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
//         let shape = x.size();
//
//         let (batch_size, total_dim) = match shape.as_slice() {
//             &[dim] => (1, dim),      // Single example
//             &[bs, dim] => (bs, dim), // Batched examples
//             other => panic!("Unexpected input shape: {:?}", other),
//         };
//
//         debug_assert_eq!(total_dim, 445, "Input must have exactly 445 features.");
//
//         let x = if batch_size == 1 && shape.len() == 1 {
//             x.view([1, 445])
//         } else {
//             x.shallow_clone()
//         };
//
//         let phase = x.i((.., 0..1));
//         debug_assert!(phase.lt(PHASE_COUNT).all().double_value(&[]) > 0.5, "phase must be in range.");
//         debug_assert_eq!(phase.size()[1], 1, "phase slice must be of length 1.");
//
//         let player_encodings = x.i((.., 1..73));
//         debug_assert!(player_encodings.lt(PLAYER_OR_NONE_COUNT).all().double_value(&[]) > 0.5, "player_encodings must be in range.");
//         debug_assert_eq!(
//             player_encodings.size()[1],
//             72,
//             "player_encodings slice must be of length 72."
//         );
//
//
//         let announcement_position_encodings = x.i((.., 73..83));
//         debug_assert!(announcement_position_encodings.lt(NUM_CARD_POSITIONS).all().double_value(&[]) > 0.5, "announcement_position_encodings must be in range.");
//         debug_assert_eq!(
//             announcement_position_encodings.size()[1],
//             10,
//             "announcement_position_encodings slice must be of length 10."
//         );
//
//         let announcement_position_in_position_encodings = x.i((.., 83..93));
//         debug_assert!(announcement_position_in_position_encodings.lt(NUM_POSTIONS_IN_CARD_POSITION).all().double_value(&[]) > 0.5, "announcement_position_in_position_encodings must be in range.");
//         debug_assert_eq!(
//             announcement_position_in_position_encodings.size()[1],
//             10,
//             "announcement_position_in_position_encodings slice must be length 10."
//         );
//
//         let lowest_announcements_encodings = x.i((.., 93..95));
//         debug_assert!(lowest_announcements_encodings.lt(ANNOUNCEMENT_OR_NONE_COUNT).all().double_value(&[]) > 0.5, "lowest_announcements_encodings must be in range.");
//         debug_assert_eq!(
//             lowest_announcements_encodings.size()[1],
//             2,
//             "lowest_announcements_encodings slice must be of length 2."
//         );
//
//         let game_type_encoding = x.i((.., 95..96));
//         debug_assert!(game_type_encoding.lt(GAME_TYPE_OR_NONE_COUNT).all().double_value(&[]) > 0.5, "game_type_encoding must be in range.");
//         debug_assert_eq!(
//             game_type_encoding.size()[1],
//             1,
//             "game_type_encoding slice must be of length 1."
//         );
//
//         let reservation_encodings = x.i((.., 96..103));
//         debug_assert!(reservation_encodings.lt(VISIBLE_RESERVATION_OR_NONE_COUNT).all().double_value(&[]) > 0.5, "reservation_encodings must be in range.");
//         debug_assert_eq!(
//             reservation_encodings.size()[1],
//             7,
//             "reservation_encodings slice must be of length 7."
//         );
//
//         let card_encodings = x.i((.., 103..199));
//         debug_assert!(card_encodings.lt(CARD_OR_NONE_COUNT).all().double_value(&[]) > 0.5, "card_encodings must be in range.");
//         debug_assert_eq!(
//             card_encodings.size()[1],
//             96,
//             "card_encodings slice must be of length 96."
//         );
//
//         let hand_counter = x.i((.., 199..205));
//         debug_assert!(hand_counter.lt(NUM_MAX_CARDS_ON_HAND + 1).all().double_value(&[]) > 0.5, "hand_counter must be in range.");
//         debug_assert_eq!(
//             hand_counter.size()[1],
//             6,
//             "hand_counter slice must be of length 6."
//         );
//
//         let card_on_hands = x.i((.., 205..397))
//             .to_kind(Kind::Float);
//         debug_assert_eq!(
//             card_on_hands.size()[1],
//             192,
//             "card_on_hands slice must be of length 192."
//         );
//
//         let card_was_played_at_position = x.i((.., 397..445));
//
//         // Phase-Embeddings
//         let phase_e = self.phase_embeddings.forward(&phase);
//         // Announcement-Embeddings
//         let lowest_ann_e = self
//             .announcement_embeddings
//             .forward(&lowest_announcements_encodings);
//         // Spieltyp-Embeddings
//         let game_type_e = self.game_type_embeddings.forward(&game_type_encoding);
//         // Vorbehalt-Embeddings
//         let visible_reservations_e = self.visible_reservations_embeddings.forward(&reservation_encodings);
//
//         let announcement_positions_f =
//             announcement_position_encodings.to_kind(Kind::Float) / (NUM_CARD_POSITIONS as f64);
//         let announcement_positions_in_pos_f = announcement_position_in_position_encodings
//             .to_kind(Kind::Float)
//             / (NUM_POSTIONS_IN_CARD_POSITION as f64);
//         let hand_counter_f = hand_counter.to_kind(Kind::Float) / (NUM_MAX_CARDS_ON_HAND as f64);
//         let card_was_played_at_position_f = card_was_played_at_position.to_kind(Kind::Float) / (NUM_CARD_POSITIONS as f64);
//
//
//         let players_e = self.player_input_embeddings.forward(&player_encodings);
//         let cards_e = self.card_embeddings.forward(&card_encodings);
//
//         let trick_players = players_e.narrow(1, 0, 12);
//         let remaining_players = players_e.narrow(1, 12, 60);
//
//         // players_e ist [batch_size, 12, player_input_embeddings_dim_size]
//         // cards_e ist [batch_size, 48, card_embeddings_dim_size]
//
//         let cards_e = cards_e.view([-1, 12, 8 * self.settings.card_embeddings_dim_size]);
//         let trick_players = trick_players.view([-1, 12, 1 * self.settings.player_input_embeddings_dim_size]);
//
//         // println!("cards_e: {:?}", cards_e.size());
//         // println!("trick_players: {:?}", trick_players.size());
//
//         let trick_input = Tensor::cat(&[trick_players, cards_e], 2);
//
//         debug_println!("trick_input: {}", trick_input);
//         let trick_input = self.trick_layers.forward_t(&trick_input, train);
//
//         let hands = card_on_hands.view([-1, 4, 48]);
//         debug_println!("hands: {}", hands);
//         let hands_l = self.hands_layers.forward_t(&hands, train);
//
//
//         let phase_e = phase_e.flatten(1, -1);
//         let remaining_players_e = remaining_players.flatten(1, -1);
//
//         let lowest_ann_e = lowest_ann_e.flatten(1, -1);
//         let game_type_e = game_type_e.flatten(1, -1);
//         let reservations_e = visible_reservations_e.flatten(1, -1);
//         let hands_l = hands_l.flatten(1, -1);
//         let trick_input = trick_input.flatten(1, -1);
//
//
//         debug_assert!(phase_e.size()[1] == self.settings.phase_embeddings_dim_size);
//         debug_assert_eq!(remaining_players_e.size()[1], 60 * self.settings.player_input_embeddings_dim_size);
//         debug_assert!(announcement_positions_f.size()[1] == 10);
//         debug_assert!(announcement_positions_in_pos_f.size()[1] == 10);
//         debug_assert!(lowest_ann_e.size()[1] == 2 * self.settings.announcement_embeddings_dim_size);
//         debug_assert!(game_type_e.size()[1] == self.settings.game_type_embeddings_dim_size);
//         debug_assert!(reservations_e.size()[1] == 7 * self.settings.visible_reservations_embeddings_dim_size);
//         debug_assert!(hand_counter_f.size()[1] == 6);
//         debug_assert!(hands_l.size()[1] == 4 * self.settings.hands_layers.last().unwrap());
//         debug_assert!(trick_input.size()[1] == 12 * self.settings.trick_layers.last().unwrap());
//         debug_assert!(card_was_played_at_position_f.size()[1] == 48);
//
//         let concat_in = Tensor::cat(
//             &[
//                 // Phase
//                 phase_e,
//                 // Spieler
//                 remaining_players_e,
//                 // 10 Positionen für Ansagen
//                 announcement_positions_f,        // 10 columns
//                 // 10 Positionen in Position
//                 announcement_positions_in_pos_f, // 10 columns
//                 // 2 Niedrigste Ansagen
//                 lowest_ann_e,
//                 // 1 Spieltyp
//                 game_type_e,
//                 // 7 Vorbehalte
//                 reservations_e,
//                 // 6 Angaben über die Anzahl der Handkarten
//                 hand_counter_f,
//                 // 4 Blätter (Ausgabe der hands_layers)
//                 hands_l,
//                 // 12 Spieler und 48 Karten (Ausgabe der trick_layers)
//                 trick_input,
//                 // 48 Karten wurden gespielt
//                 card_was_played_at_position
//             ],
//             1,
//         );
//
//         debug_println!("concat_in: {}", concat_in);
//
//         self
//             .fc_layers
//             .forward_t(&concat_in, train)
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use rand::rngs::SmallRng;
//     use rand::SeedableRng;
//     use rs_full_doko::basic::phase::FdoPhase;
//     use rs_full_doko::state::state::FdoState;
//     use strum::EnumCount;
//     use tch::Device;
//     use rs_full_doko::hand::hand::FdoHand;
//     use rs_full_doko::player::player::FdoPlayer;
//     use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
//     use crate::full_doko::ipi_network::{FullDokoImperfectInformationNetwork, FullDokoImperfectInformationNetworkConfiguration};
//     use crate::full_doko::var1::encode_ipi::encode_state_ipi;
//     use crate::full_doko::var1::ipi_output::ImperfectInformationOutput;
//
//     #[test]
//     pub fn play_full_game() {
//         let mut rng = SmallRng::from_os_rng();
//
//         let network = FullDokoImperfectInformationNetwork::new(
//             FullDokoImperfectInformationNetworkConfiguration {
//                 fc_layers: &[128, 64],
//
//                 hands_layers: &[128],
//                 trick_layers: &[128],
//
//                 phase_embeddings_dim_size: 2,
//                 player_input_embeddings_dim_size: 1,
//                 visible_reservations_embeddings_dim_size: 2,
//                 announcement_embeddings_dim_size: 2,
//                 game_type_embeddings_dim_size: 2,
//                 card_embeddings_dim_size: 2,
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
//             let x = encode_state_ipi(
//                 &state,
//                 &obs,
//                 PlayerZeroOrientedArr::from_full(
//                     [FdoHand::empty(); 4]
//                 ),
//                 PlayerZeroOrientedArr::from_full(
//                     [None; 4]
//                 ),
//                 FdoPlayer::RIGHT
//             );
//
//             let result = network.forward_t(&tch::Tensor::from_slice(&x), false);
//
//             assert_eq!(result.size(), &[1, ImperfectInformationOutput::COUNT as i64]);
//
//             state.random_action_for_current_player(&mut rng);
//         }
//
//         println!("card_embeddings: {}", network.card_embeddings.ws);
//         println!("player_input_embeddings: {}", network.player_input_embeddings.ws);
//     }
// }
