
from az_doko import AlphaZeroTicTacToeNetworkConfig, AlphaZeroTicTacToeNetwork
from az_doko import TrainConfig
from az_doko import NetworkWithTrainer
import sys
import __main__  
import az_doko  

def create_az(tb, path):
    sys.modules['__main__'].AlphaZeroTicTacToeNetworkConfig = AlphaZeroTicTacToeNetworkConfig
    sys.modules['__main__'].TrainConfig = TrainConfig

    config = AlphaZeroTicTacToeNetworkConfig(
        device="cuda",
    )

    train_config = TrainConfig(
        batch_size=128,
        learning_rate=0.0003,
        compile=False
    )

    module = AlphaZeroTicTacToeNetwork(
        config=config
    )

    # Anzahl Parameter
    n_params = sum(p.numel() for p in module.parameters())
    print("number of parameters: %.2fM" % (n_params/1e6,))

    network = NetworkWithTrainer(
        module=module,
        config=config,
        train_config=train_config,
        tb=tb
    )

    network.load(path)

    return network