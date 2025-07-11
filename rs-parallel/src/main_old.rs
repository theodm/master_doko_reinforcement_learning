mod Buffer;
mod Buffer2;

use std::sync::{Arc, mpsc, Mutex};
use futures::channel::oneshot;
use futures::channel::oneshot::Receiver;
use tokio::join;
use crate::Buffer::BatchProcessor;

#[tokio::main]
async fn main() {
    let buffer = Buffer::BatchProcessor::new(256);

    let num_of_games_per_epoch = 256;

    let x: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    loop {
        let mut handles = Vec::new();

        for i in 0..num_of_games_per_epoch {
            let mut buffer = buffer.clone();

            let result = tokio::spawn(async move {
                let state = i as f32;

                let res = buffer
                    .process(state)
                    .await;

                println!("Result: {state} -> {:?} ({:?})", res,  std::thread::current().id());

                let memory = vec![state; 256];

                return memory;
            });

            handles.push(result);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Train

        break;




    }

}
