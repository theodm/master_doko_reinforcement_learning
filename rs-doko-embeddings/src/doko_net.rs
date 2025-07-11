use tch::{IndexOp, nn, Tensor};
use tch::nn::Module;

pub struct DokoNetwork {
    pub phase_embeddings: nn::Embedding,
    pub player_input_embeddings: nn::Embedding,
    pub reservations_embeddings: nn::Embedding,
    pub card_embeddings: nn::Embedding,

    head: nn::Sequential,
}

const NUM_PHASES: i64 = 4;
const PHASE_EMBEDDINGS_DIM_SIZE: i64 = 1;

const NUM_PLAYERS_OR_NONE: i64 = 5;
const PLAYER_INPUT_EMBEDDINGS_DIM_SIZE: i64 = 2;
const NUM_RESERVATIONS_OR_NONE: i64 = 4;
const PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE: i64 = 1;
const NUM_CARDS_OR_NONE: i64 = 25;
const CARD_EMBEDDING_DIM_SIZE: i64 = 4;
const NUM_ACTIONS: i64 = 26;
const NUM_PLAYERS: i64 = 4;

impl DokoNetwork {
    pub(crate) fn new(
        vs: &nn::Path
    ) -> DokoNetwork {

        let phase_embeddings = nn::embedding(
            vs / "phase_embeddings",
            NUM_PHASES,
            PHASE_EMBEDDINGS_DIM_SIZE,
            Default::default()
        );

        let player_input_embeddings = nn::embedding(
            vs / "player_input_embeddings",
            NUM_PLAYERS_OR_NONE,
            PLAYER_INPUT_EMBEDDINGS_DIM_SIZE,
            Default::default()
        );

        let reservations_embeddings = nn::embedding(
            vs / "reservations_embeddings",
            NUM_RESERVATIONS_OR_NONE,
            PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE,
            Default::default()
        );

        let card_embeddings = nn::embedding(
            vs / "card_embeddings",
            NUM_CARDS_OR_NONE,
            CARD_EMBEDDING_DIM_SIZE,
            Default::default()
        );

        let output_size = (
            PHASE_EMBEDDINGS_DIM_SIZE +
                PLAYER_INPUT_EMBEDDINGS_DIM_SIZE * 13 +
                CARD_EMBEDDING_DIM_SIZE * 48 + CARD_EMBEDDING_DIM_SIZE * 12 * 4 +
                PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE * 4
        );

        let head = nn::seq()
            .add(nn::linear(vs / "value_head" / "0", output_size, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "value_head" / "2", 512, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "value_head" / "4", 256, 4, Default::default()));

        return DokoNetwork {
            phase_embeddings,
            player_input_embeddings,
            reservations_embeddings,
            card_embeddings,
            head,
        };
    }

    pub(crate) fn forward(
        &self,
        x: &tch::Tensor
    ) -> Tensor {
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

        let x = tch::Tensor::cat(&[
            x_phase.flatten(1, -1),
            x_player.flatten(1, -1),
            x_card.flatten(1, -1),
            x_reservation.flatten(1, -1),
        ], 1);

        let x_value = self.head.forward(&x);

        if old_x.size().len() == 2 {
            return x_value;
        } else {
            let x_value = x_value.view([-1]);

            return x_value;
        }


    }
}