// use tch::nn;
//
// pub(crate) fn create_mlp_without_out(vs: &nn::Path,
//                                      in_dim: i64,
//                                      hidden_dims: &[i64]
// ) -> nn::Sequential {
//     let mut net = nn::seq();
//     let mut current_in_dim = in_dim;
//
//     // Hidden layers
//     for (i, &dim) in hidden_dims.iter().enumerate() {
//         let linear = nn::linear(
//             vs / format!("layer_{i}"),
//             current_in_dim,
//             dim,
//             Default::default(),
//         );
//         net = net
//             .add(linear)
//             .add_fn(|x| x.relu());
//         current_in_dim = dim;
//     }
//
//     net
// }