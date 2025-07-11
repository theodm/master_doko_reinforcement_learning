use std::cmp::min;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SmallRng;
use tensorboard_rs::summary_writer::SummaryWriter;
use crate::alpha_zero::alpha_zero_train_options::AlphaZeroTrainOptions;
use crate::alpha_zero::net::network::Network;
