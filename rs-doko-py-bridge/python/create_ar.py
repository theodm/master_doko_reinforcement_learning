
import sys
import __main__ 
import az_doko  
from main import ImperfectInformationNetworkConfig, ImperfectInformationNetwork, TrainConfig, NetworkWithTrainer

def create_ar(tb, path):
    sys.modules['__main__'].ImperfectInformationNetworkConfig = ImperfectInformationNetworkConfig
    sys.modules['__main__'].TrainConfig = TrainConfig

    config = ImperfectInformationNetworkConfig(
        device="cuda",

        n_embd=256,
        n_layer=4,
        n_head=4,

        attn_pdrop = 0.1,
        resid_pdrop = 0.1,
        embd_pdrop = 0.1
    )

    train_config = TrainConfig(
        batch_size=512,
        learning_rate=0.0003,
        compile=False
    )

    module = ImperfectInformationNetwork(
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