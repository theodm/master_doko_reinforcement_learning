use pyo3::{PyErr, PyObject, Python};
use pyo3::prelude::PyAnyMethods;
use rs_doko_alpha_zero::alpha_zero::net::network::Network;
use rs_doko_impi::network::ImpiNetwork;

pub struct PyImpiNetwork {
    pub(crate) network: PyObject
}
// Hilfsfunktion zur Formatierung des Python-Fehlers mit Stacktrace
fn format_python_error(py: Python, err: PyErr) -> String {
    let traceback_str = err.traceback(py)
        .and_then(|tb| {
            let tb_module = py.import("traceback").ok()?;
            let format_tb = tb_module.getattr("format_exception") .ok()?;
            let formatted = format_tb.call1((err.get_type(py), err.value(py), tb)).ok()?;
            let joined = formatted.str().ok()?;
            Some(joined.to_string())
        });

    let value_str = err.value(py).str().map(|s| s.to_string()).unwrap_or_else(|_| "<Fehler beim Extrahieren der Fehlernachricht>".to_string());

    if let Some(tb) = traceback_str {
        format!("{}\n{}", value_str, tb)
    } else {
        value_str
    }
}

impl ImpiNetwork for crate::impi::network::PyImpiNetwork {
    fn predict(&self, input: Vec<i64>) -> Vec<f32> {
        Python::with_gil(|py| {
            let input_array = input;

            match self.network.call_method1(py, "predict", (input_array,)) {
                Ok(result) => {
                    match result.extract::<Vec<f32>>(py) {
                        Ok(output) => output,
                        Err(e) => {
                            let err_msg = format_python_error(py, e);
                            panic!("Fehler beim Extrahieren des Ergebnisses: {}", err_msg);
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format_python_error(py, e);
                    panic!("Fehler beim Aufruf der predict-Methode: {}", err_msg);
                }
            }
        })
    }

    fn clone_network(&self) -> Box<dyn ImpiNetwork> {
        Python::with_gil(|py| {
            let cloned_network = self
                .network
                .call_method0(py, "clone")
                .unwrap();

            Box::new(crate::impi::network::PyImpiNetwork {
                network: cloned_network
            })
        })
    }

    fn fit(&mut self, input: Vec<i64>, target: Vec<f32>) {
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