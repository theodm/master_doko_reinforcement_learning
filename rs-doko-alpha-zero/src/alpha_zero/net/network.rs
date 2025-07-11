use std::cell::RefCell;
use std::cmp::min;
use std::fmt::Debug;
use std::intrinsics::transmute;
use std::ops::DerefMut;
use std::path::Path;
use rand::prelude::{SliceRandom, SmallRng};
use rand_distr::num_traits::Zero;
use std::rc::Rc;
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use tensorboard_rs::summary_writer::SummaryWriter;
use tokio::sync::Mutex;
use crate::alpha_zero::net::experience_replay_buffer3::ExperienceReplayBuffer3;

pub trait Network: Debug + Send + Sync {
    fn predict(&self, input: Vec<i64>) -> (Vec<f32>, Vec<f32>);

    fn fit(
        &mut self,
        input: Vec<i64>,
        target: (Vec<f32>, Vec<f32>)
    );

    fn number_of_trained_examples(&self) -> i64;

    fn clone_network(&self, device_index: usize) -> Box<dyn Network>;
}
