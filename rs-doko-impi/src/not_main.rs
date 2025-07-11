// use crate::impi_experience_replay_buffer::ImpiExperienceReplayBuffer;
// use std::os::linux::raw::stat;
// use std::sync::Arc;
// use std::sync::atomic::AtomicU32;
// use rand::prelude::{IndexedRandom, SmallRng};
// use rand::SeedableRng;
// use strum::EnumCount;
// use tch::Device;
// use tokio::sync::oneshot::Sender;
// use tokio::task;
// use rs_doko_mcts::env::envs::env_state_full_doko::McFullDokoEnvState;
// use rs_doko_mcts::mcts::mcts::CachedMCTS;
// use rs_doko_networks::full_doko::ipi_network::FullDokoImperfectInformationNetworkConfiguration;
// use rs_doko_networks::full_doko::var1::encode_ipi::encode_state_ipi;
// use rs_doko_networks::full_doko::var1::ipi_output::ImperfectInformationOutput;
// use rs_full_doko::action::action::FdoAction;
// use rs_full_doko::card::cards::FdoCard;
// use rs_full_doko::game_type::game_type::FdoGameType;
// use rs_full_doko::observation::observation::FdoObservation;
// use rs_full_doko::reservation::reservation::FdoReservation;
// use rs_full_doko::state::state::FdoState;
// use rs_full_doko::util::po_zero_arr::PlayerZeroOrientedArr;
// use crate::forward::{forward_pred_process, test_some};
// use crate::mcts_full_doko_policy::MCTSFullDokoPolicy;
// use crate::modified_random_full_doko_policy::ModifiedRandomFullDokoPolicy;
// use crate::train_impi::train_impi;
//
//
// #[tokio::main]
// async fn main() {
//
//
//     let random_policy = ModifiedRandomFullDokoPolicy::new();
//     // test_some(
//     //     PlayerZeroOrientedArr::from_full([
//     //         Arc::new(random_policy.clone()),
//     //         Arc::new(random_policy.clone()),
//     //         Arc::new(random_policy.clone()),
//     //         Arc::new(random_policy.clone()),
//     //     ]),
//     // ).await;
//
//
//     let mcts_policy = MCTSFullDokoPolicy::new();
//     train_impi(
//         4,
//
//         0.002,
//
//         512,
//
//         FullDokoImperfectInformationNetworkConfiguration {
//             fc_layers: &[
//                 2048, 2048, 2048, 2048, 2048, 2048
//             ],
//
//             hands_layers: &[512, 512],
//             trick_layers: &[128, 128],
//
//             phase_embeddings_dim_size: 2,
//             player_input_embeddings_dim_size: 2,
//             visible_reservations_embeddings_dim_size: 4,
//             announcement_embeddings_dim_size: 4,
//             game_type_embeddings_dim_size: 4,
//             card_embeddings_dim_size: 7,
//
//             device: Device::Cuda(0),
//         },
//
//         0.06,
//
//         PlayerZeroOrientedArr::from_full([
//             Arc::new(mcts_policy.clone()),
//             Arc::new(mcts_policy.clone()),
//             Arc::new(mcts_policy.clone()),
//             Arc::new(mcts_policy.clone()),
//         ]),
//         "new_network3".to_string(),
//
//         25000,
//         3
//     ).await
// }
