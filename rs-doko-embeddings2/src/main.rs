use rand::prelude::{SliceRandom, SmallRng};
use rand::{Rng, SeedableRng};
use tch::{Device, nn, Tensor};
use tch::nn::OptimizerConfig;
use tensorboard_rs::summary_writer::SummaryWriter;

mod doko_net;
mod encode_state;
mod experience_replay_ta;
mod RotArr;

use doko_net::DokoNetwork;
use rs_doko::action::action::DoAction;
use rs_doko::action::allowed_actions::allowed_actions_to_vec;
use rs_doko::basic::phase::DoPhase;
use rs_doko::observation::observation::DoObservation;
use rs_doko::state::state::DoState;
use crate::encode_state::{encode_state_with_reservations};
use crate::experience_replay_ta::ExperienceReplayWithThrowAway;

pub fn action2index(do_action: DoAction) -> usize {
    match do_action {
        DoAction::CardDiamondNine => 0,
        DoAction::CardDiamondTen => 1,
        DoAction::CardDiamondJack => 2,
        DoAction::CardDiamondQueen => 3,
        DoAction::CardDiamondKing => 4,
        DoAction::CardDiamondAce => 5,

        DoAction::CardHeartNine => 6,
        DoAction::CardHeartTen => 7,
        DoAction::CardHeartJack => 8,
        DoAction::CardHeartQueen => 9,
        DoAction::CardHeartKing => 10,
        DoAction::CardHeartAce => 11,

        DoAction::CardSpadeNine => 12,
        DoAction::CardSpadeTen => 13,
        DoAction::CardSpadeJack => 14,
        DoAction::CardSpadeQueen => 15,
        DoAction::CardSpadeKing => 16,
        DoAction::CardSpadeAce => 17,

        DoAction::CardClubNine => 18,
        DoAction::CardClubTen => 19,
        DoAction::CardClubJack => 20,
        DoAction::CardClubQueen => 21,
        DoAction::CardClubKing => 22,
        DoAction::CardClubAce => 23,

        DoAction::ReservationHealthy => 24,
        DoAction::ReservationWedding => 25,
    }
}

pub fn index2action(action_index: usize) -> DoAction {
    match action_index {
        0 => DoAction::CardDiamondNine,
        1 => DoAction::CardDiamondTen,
        2 => DoAction::CardDiamondJack,
        3 => DoAction::CardDiamondQueen,
        4 => DoAction::CardDiamondKing,
        5 => DoAction::CardDiamondAce,

        6 => DoAction::CardHeartNine,
        7 => DoAction::CardHeartTen,
        8 => DoAction::CardHeartJack,
        9 => DoAction::CardHeartQueen,
        10 => DoAction::CardHeartKing,
        11 => DoAction::CardHeartAce,

        12 => DoAction::CardSpadeNine,
        13 => DoAction::CardSpadeTen,
        14 => DoAction::CardSpadeJack,
        15 => DoAction::CardSpadeQueen,
        16 => DoAction::CardSpadeKing,
        17 => DoAction::CardSpadeAce,

        18 => DoAction::CardClubNine,
        19 => DoAction::CardClubTen,
        20 => DoAction::CardClubJack,
        21 => DoAction::CardClubQueen,
        22 => DoAction::CardClubKing,
        23 => DoAction::CardClubAce,

        24 => DoAction::ReservationHealthy,
        25 => DoAction::ReservationWedding,
        _ => panic!("Invalid action index"),
    }
}

pub fn allowed_actions_by_action_index(
    observation: &DoObservation
) -> heapless::Vec<usize, 26> {
    let allowed_actions = allowed_actions_to_vec(observation.allowed_actions_current_player);

    allowed_actions.iter().map(|x| action2index(*x)).collect()
}

fn generate_data(
    rng: &mut SmallRng,
    erb: &mut ExperienceReplayWithThrowAway
) {
    let mut doko = DoState::new_game(
        rng
    );

    let mut temp_states = Vec::with_capacity(55);
    loop {
        doko.random_action_for_current_player(
            rng
        );

        let obs = doko.observation_for_current_player();

        if obs.phase == DoPhase::Finished {
            break;
        }

        temp_states.push(
            doko.clone()
        );
    }

    let value_target = doko.observation_for_current_player()
        .finished_observation
        .unwrap()
        .player_points
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();

    let value_target_arr: [f32; 4] =
        value_target
            .as_slice()
            .try_into()
            .unwrap();

    let value_target_rot = RotArr::RotArr::new_from_0(
        0,
        value_target_arr
    );

    for state in temp_states {
        let encoded_state = encode_state_with_reservations(
            &state
        );

        let allowed_actions = allowed_actions_by_action_index(
            &state.observation_for_current_player()
        );

        let mut allowed_actions_mask: heapless::Vec<f32, 26> = heapless::Vec::from_slice(&[0.0f32; 26])
            .unwrap();

        for action in allowed_actions.iter() {
            allowed_actions_mask[*action] = 1.0;
        }

        let rot_arr_target = value_target_rot
            .rotated_for_i(
                state.observation_for_current_player().current_player.unwrap()
            );

        //if rng.gen_bool(0.4) {
            erb.append(
                &encoded_state,
                &rot_arr_target.extract(),
                &allowed_actions_mask.as_slice(),
            );
        //}

        if erb.memory_full {
            break;
        }
    }
}


fn generate_data_until_full(
    rng: &mut SmallRng,
    erb: &mut ExperienceReplayWithThrowAway
) {
    while !erb.memory_full {
        generate_data(rng, erb);
    }
}



fn cross_entropy_loss(
    input: &tch::Tensor,
    target: &tch::Tensor,
) -> tch::Tensor {
    input.cross_entropy_loss::<Tensor>(
        target,
        None,
        tch::Reduction::Mean,
        -100,
        0.0,
    )
}

fn main() {
    let mut summary_writer = SummaryWriter::new("logs/rs-embeddings2_1024x512_1024mb_32000_shared_lr_0.002");

    let mut rng = SmallRng::seed_from_u64(5123);

    let device = Device::Cuda(0);

    let mut vs = nn::VarStore::new(device);
    let net = DokoNetwork::new(&vs.root(), device);
    //
    // vs.load("doko-embeddings-500000.safetensors").unwrap();
    //
    // println!("{}", net.card_embeddings.ws);
    // println!("{}", net.player_input_embeddings.ws);
    // println!("{}", net.reservations_embeddings.ws);

    let mut opt = nn::Adam::default()
        .build(&vs, 0.002)
        .unwrap();

    let mut epoch = 0;

    let mut erb = ExperienceReplayWithThrowAway::new(
        114,
        4,
        26,
        128,
        32000
    );

    let start_time = std::time::Instant::now();

    let mut training_step = 0;
    let mut number_of_experiences = 0;
    loop {
        generate_data_until_full(
            &mut rng,
            &mut erb
        );

        println!("Data generated");

        let max_memory = erb
            .max_memory;

        let mut batch_indices: Vec<i64> = (0..max_memory as i64).collect();

        batch_indices.shuffle(&mut rng);

        let mut loss_sum = 0.0;
        let mut last_loss = 0.0;

        let state_memory_t = tch::Tensor::from_slice(&erb.states_memory)
            .view([erb.max_memory as i64, erb.state_dim as i64])
            .to_device(device);
        let value_memory_t = tch::Tensor::from_slice(&erb.value_targets_memory)
            .view([erb.max_memory as i64, erb.value_dim as i64])
            .to_device(device);
        let policy_memory_t = tch::Tensor::from_slice(&erb.policy_targets_memory)
            .view([erb.max_memory as i64, erb.policy_dim as i64])
            .to_device(device);

        for batch_index in 0..erb.n_minibatches {
            let start = batch_index * erb.minibatch_size;
            let end = start + erb.minibatch_size;
            let mb_indices = &batch_indices[start..end];

            let mb_indices_tensor = tch::Tensor::from_slice(mb_indices).to_device(device);
            let mb_states = state_memory_t.index_select(0, &mb_indices_tensor);
            let mb_values = value_memory_t.index_select(0, &mb_indices_tensor);
            let mb_policies = policy_memory_t.index_select(0, &mb_indices_tensor);

            let (value_preds, policy_preds) = net.forward(&mb_states);

            let policy_loss = cross_entropy_loss(
                &policy_preds,
                &mb_policies,
            );

            let value_loss = value_preds
                .view([-1])
                .mse_loss(&mb_values.view([-1]), tch::Reduction::Mean);

            if batch_index == 0 {
                println!("state: {}", mb_states.get(0));
                println!("mb_policies: {}", mb_policies.get(0));
                println!("value_preds: {}", value_preds.get(0));
                println!("mb_values: {}", mb_values.get(0));
                println!("policy_preds: {}", policy_preds.get(0));
            }

            let policy_loss_d = policy_loss.double_value(&[]);
            let value_loss_d = value_loss.double_value(&[]);

            let loss = policy_loss + value_loss;

            opt.zero_grad();
            loss.backward();
            opt.step();

            let loss = loss.double_value(&[]);
            summary_writer.add_scalar("loss", loss as f32, training_step);
            summary_writer.add_scalar("policy_loss", policy_loss_d as f32, training_step);
            summary_writer.add_scalar("value_loss", value_loss_d as f32, training_step);

            let elapsed_secs = start_time.elapsed().as_secs_f64();
            summary_writer.add_scalar("loss_t", loss as f32, elapsed_secs as usize);
            summary_writer.add_scalar("policy_loss_t", policy_loss_d as f32, elapsed_secs as usize);
            summary_writer.add_scalar("value_loss_t", value_loss_d as f32, elapsed_secs as usize);

            summary_writer.add_scalar("loss_noe", loss as f32, number_of_experiences);
            summary_writer.add_scalar("policy_loss_noe", policy_loss_d as f32, number_of_experiences);
            summary_writer.add_scalar("value_loss_noe", value_loss_d as f32, number_of_experiences);

            training_step += 1;
            number_of_experiences += erb.minibatch_size;
        }

        summary_writer.add_scalar("epoch_noe", epoch as f32, number_of_experiences);

        println!(
            "Trained {} minibatches",
            erb.n_minibatches
        );

        erb.memory_full = false;
        //
        // if epoch % 50 == 0 {
        //     summary_writer.add_scalar("loss", avg_loss as f32, epoch);
        // }

        if epoch % 50000 == 0 {
            let path = format!("doko-embeddings-{}-big.safetensors", epoch);
            vs.save(path.clone()).unwrap();

            println!("Saved model to {:?}", path);
        }

        epoch = epoch + 1;

    }
}
