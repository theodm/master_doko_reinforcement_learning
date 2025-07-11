use rand::prelude::SmallRng;
use rand::SeedableRng;
use tch::{Device, nn, Tensor};
use tch::nn::OptimizerConfig;
use tensorboard_rs::summary_writer::SummaryWriter;

mod doko_net;
mod encode_state;

use doko_net::DokoNetwork;
use rs_doko::state::state::DoState;
use crate::encode_state::{encode_state, encode_state_with_reservations};

fn generate_data(
    rng: &mut SmallRng,
) -> ([i64; 114], [f32; 4]) {
    let mut doko = DoState::new_game(
        rng
    );

    loop {
        let is_finished = doko.random_action_for_current_player(
            rng
        );

        if is_finished {
            break;
        }
    }

    let result_points =
        doko.observation_for_current_player()
            .finished_observation
            .unwrap()
            .player_points;

    let x = encode_state_with_reservations(
        &doko
    );

    (
        x.iter()
            .map(|i| *i as i64)
            .collect::<Vec<i64>>()
            .as_slice()
            .try_into()
            .unwrap(),
        result_points
            .iter()
            .map(|i| *i as f32)
            .collect::<Vec<f32>>()
            .as_slice()
            .try_into()
            .unwrap()
    )

}

fn generate_data2(
    rng: &mut SmallRng
) -> (Tensor, Tensor) {
    let mut data = [
        0i64; 128 * 114
    ];
    let mut results = [
        0f32; 128 * 4
    ];

    for i in 0..128 {
        let (x, y) = generate_data(rng);

        data[i * 114..(i + 1) * 114].copy_from_slice(&x);
        results[i * 4..(i + 1) * 4].copy_from_slice(&y);
    }

    return (
        Tensor::from_slice(&data).to_device(Device::Cuda(0)).view([128, 114]),
        Tensor::from_slice(&results).to_device(Device::Cuda(0)).view([128, 4])
    )
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
    let mut summary_writer = SummaryWriter::new("logdir/rs-embeddings2");

    let mut rng = SmallRng::seed_from_u64(5123);

    let mut vs = nn::VarStore::new(Device::Cuda(0));
    let net = DokoNetwork::new(&vs.root());
    //
    // vs.load("doko-embeddings-500000.safetensors").unwrap();
    //
    // println!("{}", net.card_embeddings.ws);
    // println!("{}", net.player_input_embeddings.ws);
    // println!("{}", net.reservations_embeddings.ws);

    let mut opt = nn::Adam::default()
        .build(&vs, 1e-3)
        .unwrap();

    let mut epoch = 0;
    loop {
        let (data, results) = generate_data2(&mut rng);

        let x = data;
        let y = results;

        let predicted = net
            .forward(&x);

        let loss = predicted
            .view([-1])
            .mse_loss(&y.view([-1]), tch::Reduction::Mean);

        if (epoch % 5000 == 0) {
            println!("x 1: {}", x.get(0));
            println!("predicted 1: {}", predicted.get(0));
            println!("real 1: {}", y.get(0));
        }

        opt.zero_grad();
        opt.backward_step(&loss);
        opt.step();

        if epoch % 50 == 0 {
            summary_writer.add_scalar("loss", loss.double_value(&[]) as f32, epoch);
        }

        if epoch % 1000 == 0 {
            println!("epoch: {:?} loss: {:?}", epoch, loss);
        }

        if epoch % 50000 == 0 {
            let path = format!("doko-embeddings-{}-big.safetensors", epoch);
            vs.save(path.clone()).unwrap();

            println!("Saved model to {:?}", path);
        }

        epoch = epoch + 1;

    }
}
