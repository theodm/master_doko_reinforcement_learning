
pub trait TensorboardSender: Send + Sync {
    fn scalar(&self, tag: &str, value: f32, step: i64);
}



