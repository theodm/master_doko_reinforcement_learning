use pyo3::{PyObject, Python};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use rs_doko_impi::tensorboard::TensorboardSender;

pub struct TensorboardController {
    pub(crate) py_controller: PyObject
}

impl TensorboardController {
    pub(crate) fn new(py_controller: PyObject) -> TensorboardController {
        TensorboardController {
            py_controller: py_controller
        }
    }

    fn scalar(&self, tag: &str, value: f32, step: i64) {
        Python::with_gil(|py| {
            self
                .py_controller
                .call_method1(py, "scalar", (tag, value, step))
                .unwrap();
        });
    }
}

impl TensorboardSender for TensorboardController {
    fn scalar(&self, tag: &str, value: f32, step: i64) {
        self.scalar(tag, value, step);
    }
}

impl rs_doko_alpha_zero::alpha_zero::tensorboard::tensorboard_sender::TensorboardSender for TensorboardController {
    fn scalar(&self, tag: &str, value: f32, step: i64) {
        self.scalar(tag, value, step);
    }
}


//
// async fn create_tensorboard_channel(
//     tensorboard_controller: TensorboardController
// ) -> (UnboundedSender<(str, f32, i64)>, JoinHandle<()>) {
//     let channel = tokio::sync::mpsc::unbounded_channel();
//
//     let (sender, mut receiver) = channel;
//
//     let receiver_handle = tokio::spawn(async move {
//         while let Some((tag, value, step)) = receiver
//             .recv()
//             .await {
//             tensorboard_controller.scalar(&tag, value, step);
//         }
//     });
//
//     return (sender, receiver_handle);
// }