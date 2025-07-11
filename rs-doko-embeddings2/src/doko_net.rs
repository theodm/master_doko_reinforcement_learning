use rand::distr::Slice;
use tch::nn::Module;
use tch::{nn, Device};
use tch::{IndexOp, Tensor};

pub struct DokoNetwork {
    phase_embeddings: nn::Embedding,
    pub player_input_embeddings: nn::Embedding,
    pub reservations_embeddings: nn::Embedding,
    pub card_embeddings: nn::Embedding,

    hidden_layer_1: nn::Linear,
    hidden_layer_2: nn::Linear,
    value_head: nn::Linear,
    policy_head: nn::Linear,
}

const NUM_PHASES: i64 = 4;
const PHASE_EMBEDDINGS_DIM_SIZE: i64 = 2;

const NUM_PLAYERS_OR_NONE: i64 = 5;
const PLAYER_INPUT_EMBEDDINGS_DIM_SIZE: i64 = 4;
const NUM_RESERVATIONS_OR_NONE: i64 = 4;
const PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE: i64 = 2;
const NUM_CARDS_OR_NONE: i64 = 25;
const CARD_EMBEDDING_DIM_SIZE: i64 = 8;
const NUM_ACTIONS: i64 = 26;
const NUM_PLAYERS: i64 = 4;


impl DokoNetwork {
    pub(crate) fn new(vs: &nn::Path, device: Device) -> DokoNetwork {
        let mut phase_embeddings = nn::embedding(
            vs / "phase_embeddings",
            NUM_PHASES,
            PHASE_EMBEDDINGS_DIM_SIZE,
            Default::default(),
        );

        // phase_embeddings.ws = tch::Tensor::from_slice(
        //     &[
        //         0.0f32,
        //         1.0f32,
        //         2.0f32,
        //         3.0f32
        //     ]
        // )
        //     .view([4, 1])
        //     .to_dtype(tch::Kind::Float, false, false)
        //     .to_device(device);

        let mut player_input_embeddings = nn::embedding(
            vs / "player_input_embeddings",
            NUM_PLAYERS_OR_NONE,
            PLAYER_INPUT_EMBEDDINGS_DIM_SIZE,
            Default::default(),
        );

        // player_input_embeddings.ws = tch::Tensor::from_slice(
        //     &[
        //         9.5489e-1, -1.8551e0,
        //         -7.5117e-1, -5.6373e-3,
        //         2.4339e-4, -9.7017e-1,
        //         7.2844e-1, 2.6173e-2,
        //         5.6016e-4, 9.8593e-1
        //     ]
        // )
        //     .view([5, 2])
        //     .to_dtype(tch::Kind::Float, false, false)
        //     .to_device(device);
        //
        // player_input_embeddings.ws.set_requires_grad(false);

        let mut reservations_embeddings = nn::embedding(
            vs / "reservations_embeddings",
            NUM_RESERVATIONS_OR_NONE,
            PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE,
            Default::default(),
        );

        // reservations_embeddings.ws = tch::Tensor::from_slice(
        //     &[-0.7619,
        //         -1.2321,
        //         -0.1075,
        //         2.1247
        //     ]
        // )
        //     .view([4, 1])
        //     .to_dtype(tch::Kind::Float, false, false)
        //     .to_device(device);
        //
        // reservations_embeddings.ws.set_requires_grad(false);

        let mut card_embeddings = nn::embedding(
            vs / "card_embeddings",
            NUM_CARDS_OR_NONE,
            CARD_EMBEDDING_DIM_SIZE,
            Default::default(),
        );

        // card_embeddings.ws = tch::Tensor::from_slice(
        //     &[
        //         -1.9826e-2,  1.4208e-3,  6.0393e-3, -5.9130e-3,
        //          7.4546e-2,  2.2138e-2, -5.1060e-2,  9.5488e-2,
        //         -7.0131e-2,  5.0816e-2,  1.1120e-1,  1.5541e-3,
        //          6.2291e-2,  2.5137e-2, -8.0244e-3,  3.8102e-2,
        //          7.3335e-2,  1.1244e-2,  4.6585e-2, -1.7170e-2,
        //          1.7441e-2,  3.5621e-2, -1.2117e-3,  6.4266e-2,
        //         -8.6762e-2,  2.9845e-2,  1.2934e-1, -3.8638e-2,
        //          3.0440e-2, -9.5623e-3, -9.4150e-2,  1.8032e-1,
        //          8.0182e-2, -1.1631e-1,  2.4289e-1, -1.1529e-1,
        //          6.9634e-2,  4.1623e-2, -5.9780e-4,  1.0365e-2,
        //          9.6373e-2,  1.4285e-2,  5.7314e-2, -3.0833e-2,
        //         -1.1183e-2, -2.2159e-4, -3.3209e-2,  1.3751e-1,
        //         -1.2574e-1,  5.3448e-4,  1.0964e-1,  4.9583e-2,
        //          3.8688e-2, -1.0230e-2, -7.7949e-2,  1.7613e-1,
        //         -1.1909e-1, -7.1519e-4,  7.9893e-2,  9.6715e-2,
        //          8.7793e-2,  9.6449e-3,  7.2335e-3,  5.1698e-3,
        //         -1.3855e-1, -1.1787e-1, -6.2476e-1, -4.1579e-1,
        //         -2.5272e-2, -2.1389e-2, -3.2746e-2,  1.4634e-1,
        //         -1.2389e-1,  1.8615e-2,  1.0741e-1,  2.9644e-2,
        //          2.5758e-2, -2.0637e-3, -8.5322e-2,  1.8199e-1,
        //         -1.1516e-1,  2.6704e-3,  8.1790e-2,  8.2925e-2,
        //          8.2785e-2,  2.5051e-2,  2.1213e-2,  1.5915e-2,
        //          1.0573e-1,  1.6944e-2,  5.2025e-2, -4.7133e-2,
        //         -1.5638e-2,  7.4539e-3, -2.3748e-2,  1.2091e-1,
        //         -1.2356e-1,  9.4276e-3,  9.6218e-2,  5.8072e-2
        //     ]
        // )
        //     .view([25, 4])
        //     .to_dtype(tch::Kind::Float, false, false)
        //     .to_device(device);
        //
        // card_embeddings.ws.set_requires_grad(false);

        let output_size = (PHASE_EMBEDDINGS_DIM_SIZE
            + PLAYER_INPUT_EMBEDDINGS_DIM_SIZE * 13
            + CARD_EMBEDDING_DIM_SIZE * 48
            + CARD_EMBEDDING_DIM_SIZE * 12 * 4
            + PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE * 4);

        const LAYER_1_SIZE: i64 = 1024;
        const LAYER_2_SIZE: i64 = 512;

        println!("output_size: {}", output_size);

        let hidden_layer_1 = nn::linear(
            vs / "shared_hidden_1",
            output_size,
            LAYER_1_SIZE,
            Default::default(),
        );

        let hidden_layer_2 = nn::linear(
            vs / "shared_hidden_2",
            LAYER_1_SIZE,
            LAYER_2_SIZE,
            Default::default(),
        );

        let value_head = nn::linear(
            vs / "value_head",
            LAYER_2_SIZE,
            4,
            Default::default(),
        );

        let policy_head = nn::linear(
            vs / "policy_head",
            LAYER_2_SIZE,
            26,
            Default::default(),
        );

        return DokoNetwork {
            phase_embeddings,
            player_input_embeddings,
            reservations_embeddings,
            card_embeddings,
            hidden_layer_1,
            hidden_layer_2,
            value_head,
            policy_head,
        };
    }

    pub(crate) fn forward(&self, x: &tch::Tensor) -> (tch::Tensor, tch::Tensor) {
        // println!("x: {}", x);
        let old_x = x;
        let x = x.view([-1, 114]);

        let phase = x.i((.., 0..1));
        let player_inputs = x.i((.., 1..14));
        let card_inputs = x.i((.., 14..110));
        let reservation_inputs = x.i((.., 110..114));

        // println!("phase: {}", phase);
        // println!("player_inputs: {}", player_inputs);
        // println!("card_inputs: {}", card_inputs);

        let x_phase = self.phase_embeddings.forward(&phase);
        let x_player = self.player_input_embeddings.forward(&player_inputs);
        let x_card = self.card_embeddings.forward(&card_inputs);
        let x_reservation = self.reservations_embeddings.forward(&reservation_inputs);

        // println!("x_phase {}", x_phase);
        // println!("x_player {}", x_player);
        // println!("x_card {}", x_card);

        let x = tch::Tensor::cat(
            &[
                x_phase.flatten(1, -1),
                x_player.flatten(1, -1),
                x_card.flatten(1, -1),
                x_reservation.flatten(1, -1),
            ],
            1,
        );

        let x = self.hidden_layer_1.forward(&x);
        let x = x.relu();
        let x = self.hidden_layer_2.forward(&x);
        let x = x.relu();

        let x_value = self.value_head.forward(&x);
        let x_policy = self.policy_head.forward(&x);

        if old_x.size().len() == 2 {
            // println!("card_embeddings: {}", self.card_embeddings.ws);
            // println!("x: {}", x);
            return (x_value, x_policy);
        } else {
            let x_value = x_value.view([-1]);
            let x_policy = x_policy.view([-1]);

            return (x_value, x_policy);
        }
    }
}
