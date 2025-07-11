import torch
from torch import multiprocessing

from rs_doko_py_bridge import p_train_impi2

from TensorboardController import TensorboardController
from main import ImperfectInformationNetworkConfig, ImperfectInformationNetwork, TrainConfig, NetworkWithTrainer

from create_az import create_az


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    run_name = "hopefully_final_mcts"

    tb = TensorboardController(f"runs/{run_name}")

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

    p_train_impi2(
        network,
        {},
        0.05,
        1000000,
        tb
    )
