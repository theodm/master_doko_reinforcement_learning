
from rs_doko_py_bridge import p_gather_games
from az_doko import AlphaZeroTicTacToeNetworkConfig, AlphaZeroTicTacToeNetwork, TrainConfig, NetworkWithTrainer
from TensorboardController import TensorboardController

#
# tb = TensorboardController(f"runs/whats")
#
# config = AlphaZeroTicTacToeNetworkConfig(
#     device="cuda",
# )
#
# train_config = TrainConfig(
#     batch_size=128,
#     learning_rate=0.0003,
#     compile=False
# )
#
# module = AlphaZeroTicTacToeNetwork(
#     config=config
# )
#
# # Anzahl Parameter
# n_params = sum(p.numel() for p in module.parameters())
# print("number of parameters: %.2fM" % (n_params/1e6,))
#
# network = NetworkWithTrainer(
#     module=module,
#     config=config,
#     train_config=train_config,
#     tb=tb
# )
#
# network.load("models7/model_17318500.pth")

network = {}
p_gather_games(
    100000,
    network
)