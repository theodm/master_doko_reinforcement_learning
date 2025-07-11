use pyo3::{PyObject, Python};
use tokio::task;
use rs_doko_alpha_zero::alpha_zero::net::network::Network;

#[derive(Debug)]
pub struct PyAlphaZeroNetwork {
    pub(crate) network: PyObject
}

impl PyAlphaZeroNetwork {
    pub(crate) fn new(network: PyObject) -> PyAlphaZeroNetwork {
        PyAlphaZeroNetwork {
            network
        }
    }
}

impl Network for PyAlphaZeroNetwork {
    fn predict(&self, input: Vec<i64>) -> (Vec<f32>, Vec<f32>) {
        Python::with_gil(|py| {
            let input_array = input;

            let res = self
                .network
                .call_method1(py, "predict", (input_array,))
                .map(|x| x.extract::<(Vec<f32>, Vec<f32>)>(py))
                .unwrap()
                .unwrap();

            res
        })
    }

    fn clone_network(&self, device_index: usize) -> Box<dyn Network> {
        Python::with_gil(|py| {
            let cloned_network = self
                .network
                .call_method1(py, "clone", (device_index,))
                .unwrap();

            Box::new(PyAlphaZeroNetwork {
                network: cloned_network
            })
        })
    }

    fn fit(&mut self, input: Vec<i64>, target: (Vec<f32>, Vec<f32>)) {
        Python::with_gil(|py| {
            self
                .network
                .call_method1(py, "fit", (input, target))
                .unwrap();
        });
    }

    fn number_of_trained_examples(&self) -> i64 {
        Python::with_gil(|py| {
            self
                .network
                .call_method0(py, "number_of_trained_examples")
                .unwrap()
                .extract::<i64>(py)
                .unwrap()
        })
    }
}