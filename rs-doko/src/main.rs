
use crate::test_generator::test_generator::test_generator;

mod state;
mod action;
mod observation;
mod card;
mod hand;
mod player;
mod trick;
mod reservation;
mod basic;
mod util;
mod stats;
mod teams;
mod display;

mod test_generator;

fn main() {
    test_generator();
}

