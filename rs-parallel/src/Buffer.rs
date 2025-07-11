use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, fence, Ordering};
use futures::future::select;
use tokio::sync::{Mutex, MutexGuard, Notify, RwLock};
use tokio::{select, task, time};
use tokio::time::error::Elapsed;
use tokio::time::{sleep, timeout};

struct Results {
    values: Vec<f32>,
    policy: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct BatchProcessor {
    pub(crate) buffer: Arc<Mutex<Vec<f32>>>,
    pub(crate) notify_result_was_calculated: Arc<Notify>,
    // pub(crate) notify_buffer_is_full: Arc<Notify>,

    pub(crate) results: Arc<RwLock<Results>>,

    pub(crate) batch_size: usize,
}

fn x(state: f32) -> (f32, f32) {
    let value = state * 2.0;
    let policy = state / 2.0;

    return (value, policy);
}

fn batch_x(
    states: &Vec<f32>,
    results: &mut Results,
) {
    for i in 0..states.len() {
        let (value, policy) = x(states[i]);

        results.values[i] = value;
        results.policy[i] = policy;
    }

    println!("----------------")
}

impl BatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::with_capacity(batch_size))),
            notify_result_was_calculated: Arc::new(Notify::new()),
            // notify_buffer_is_full: Arc::new(Notify::new()),
            batch_size,
            results: Arc::new(
                RwLock::new(
                    Results {
                        values: vec![0.0; batch_size],
                        policy: vec![0.0; batch_size],
                    }
                )
            ),
        }
    }

    // pub async fn force_calculation_periodically(
    //     &self,
    // ) {
    //     loop {
    //         let await_handle = std::time::Duration::from_secs(2);
    //         let notify_full_handle = self.notify_buffer_is_full.notified();
    //
    //         timeout(
    //             await_handle,
    //             notify_full_handle
    //         ).await;
    //
    //         self.buffer
    //     }
    // }



    async fn read_results(
        &self,
        index: usize,
    ) -> (f32, f32) {
        let wait_for_result = self
            .notify_result_was_calculated
            .notified();

        if index == 0 {
            let timeout = sleep(std::time::Duration::from_secs(5));

            select! {
                _ = wait_for_result => {

                }
                _ = timeout => {
                    self.calc_and_notify(
                        self
                            .buffer
                            .lock()
                            .await
                    ).await;
                }
            }
        } else {
            wait_for_result.await;
        }

        let results = self
            .results
            .read()
            .await;

        let (value, policy) = (
            results.values[index],
            results.policy[index],
        );

        drop(results);

        return (value, policy);
    }

    async fn calc_and_notify(
        &self,
        mut buffer: MutexGuard<'_, Vec<f32>>,
    ) {
        let mut results = self
            .results
            .write()
            .await;

        batch_x(
            &buffer,
            &mut results,
        );

        buffer.clear();

        self
            .notify_result_was_calculated
            .notify_waiters();

        drop(results);
        drop(buffer);
    }

    pub async fn process(&mut self, state: f32) -> (f32, f32) {
        let mut buffer = self
            .buffer
            .lock()
            .await;

        let current_index = buffer.len();
        buffer.push(state);

        if buffer.len() == self.batch_size {
            // Buffer ist voll, jetzt verarbeiten.
            self
                .calc_and_notify(buffer)
                .await;

            return self.read_results(current_index).await;
        } else {
            // Ansonsten: Anderen die Möglichkeit geben, neue Dinge hinzuzufügen und selbst
            // auf das Ergebnis warten.
            drop(buffer);

            return self.read_results(current_index).await;
        }
    }
}


// let timeout_or_notified = timeout(
//     std::time::Duration::from_secs(5),
//     self
//         .notify_result_was_calculated
//         .notified()
// ).await;
//
// match timeout_or_notified {
//     Ok(_) => {
//         return self.read_results(current_index).await;
//     }
//     Err(_) => {
//         self.calc_and_notify(
//             self
//                 .buffer
//                 .lock()
//                 .await
//         ).await;
//
//         return self.read_results(current_index).await;
//     }
// }