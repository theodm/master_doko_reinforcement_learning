// use tch::nn::{self, BatchNormConfig, BatchNorm, Linear, ModuleT, Path, SequentialT};
// use tch::{Kind, Tensor};
//
// #[derive(Debug)]
// pub struct ResidualBlock {
//     layers: SequentialT,
//     shortcut: Option<Linear>,
// }
//
// impl ResidualBlock {
//     pub fn new(vs: &Path, in_dim: i64, out_dim: i64) -> Self {
//         let bn_config = BatchNormConfig {
//             cudnn_enabled: true,
//             ..Default::default()
//         };
//
//         let fc1 = nn::linear(vs / "fc1", in_dim, out_dim, Default::default());
//         let bn1 = nn::batch_norm1d(vs / "bn1", out_dim, bn_config);
//         let fc2 = nn::linear(vs / "fc2", out_dim, out_dim, Default::default());
//         let bn2 = nn::batch_norm1d(vs / "bn2", out_dim, bn_config);
//
//         let shortcut = if in_dim != out_dim {
//             Some(nn::linear(vs / "shortcut", in_dim, out_dim, Default::default()))
//         } else {
//             None
//         };
//
//         let layers = nn::seq_t()
//             .add(fc1)
//             .add_fn_t(move |x, train| bn1.forward_t(x, train))
//             .add_fn_t(|x, _train| x.relu())
//             .add(fc2)
//             .add_fn_t(move |x, train| bn2.forward_t(x, train));
//
//         Self { layers, shortcut }
//     }
// }
//
// impl ModuleT for ResidualBlock {
//     fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
//         let residual = match &self.shortcut {
//             Some(proj) => proj.forward_t(xs, train),
//             None => xs.shallow_clone(),
//         };
//
//         let out = self.layers.forward_t(xs, train);
//         (out + residual).relu()
//     }
// }
//
// pub fn create_residual_backbone(
//     vs: &Path,
//     in_dim: i64,
//     hidden_dims: &[i64],
// ) -> (SequentialT, i64) {
//     let mut net = nn::seq_t();
//     let mut current_dim = in_dim;
//
//     for (i, &dim) in hidden_dims.iter().enumerate() {
//         net = net.add(ResidualBlock::new(
//             &(vs / format!("res_block_{i}").as_str()),
//             current_dim,
//             dim,
//         ));
//         current_dim = dim;
//     }
//
//     (net, current_dim)
// }
//
//
// pub fn create_residual_network(
//     vs: &Path,
//     in_dim: i64,
//     hidden_dims: &[i64],
//     out_dim: i64,
// ) -> SequentialT {
//     let (mut net, final_dim) = create_residual_backbone(vs, in_dim, hidden_dims);
//
//     net = net.add(nn::linear(
//         vs / "final_out",
//         final_dim,
//         out_dim,
//         Default::default(),
//     ));
//
//     // net = net.add_fn_t(|x, _train| x.relu());
//
//     net
// }