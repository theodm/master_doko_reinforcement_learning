use std::cmp::min;
use std::path::Path;
use graphviz_rust::print;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use tensorboard_rs::summary_writer::SummaryWriter;
use rs_doko_networks::full_doko::ipi_network::{ FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_INPUT_SIZE, FULL_DOKO_IMPERFECT_INFORMATION_NETWORK_OUTPUT_SIZE};
use crate::impi_experience_replay_buffer::ImpiExperienceReplayBuffer;
use crate::network::ImpiNetwork;

pub struct NetworkTrainer {
    pub number_of_trained_examples: usize,
}

// fn cross_entropy_loss(
//     input: &tch::Tensor,
//     target: &tch::Tensor,
// ) -> tch::Tensor {
//     input.cross_entropy_loss::<Tensor>(
//         target,
//         None,
//         tch::Reduction::Mean,
//         -100,
//         0.0,
//     )
// }

impl NetworkTrainer {
    pub fn new() -> Self {
        NetworkTrainer {
            number_of_trained_examples: 0,
        }
    }

    pub fn train_network(
        &mut self,

        buffer: &Vec<([i64; 311], [f32; 33])>,

        epoch: usize,

        network: &mut Box<dyn ImpiNetwork>,
        multi_progress: MultiProgress
    ) {
        let mut rng = &mut SmallRng::from_os_rng();

        let (state_memory, target_memory): (Vec<[i64; 311]>, Vec<[f32; 33]>) = buffer
            .iter()
            .cloned()
            .unzip();

        let mut pb = multi_progress.add(
            ProgressBar::new(1)
        );

        pb
            .set_style(
                ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg} [{wide_bar:.cyan/blue}] {human_pos} / {human_len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-")
            );
        pb.set_message("Training ".to_string());

        let state_memory = state_memory
            .iter()
            .flatten()
            .copied()
            .collect();

        let target_memory = target_memory
            .iter()
            .flatten()
            .copied()
            .collect();

        network.fit(state_memory, target_memory);

        pb.finish_and_clear();
    }
}