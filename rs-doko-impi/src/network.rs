
pub trait ImpiNetwork : Send {
    fn predict(&self, input: Vec<i64>) -> Vec<f32>;

    fn fit(
        &mut self,
           input: Vec<i64>,
           target: Vec<f32>
    );

    fn number_of_trained_examples(&self) -> i64;

    fn clone_network(&self) -> Box<dyn ImpiNetwork>;
}