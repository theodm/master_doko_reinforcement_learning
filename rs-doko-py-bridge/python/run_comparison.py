import torch
from TensorboardController import TensorboardController

from rs_doko_py_bridge import p_run_comparison
from create_ar import create_ar
from create_az import create_az

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    run_name = "runrunrun"

    tb = TensorboardController(f"runs/{run_name}")

    mcts_network_ar = create_ar(
        tb,
        "/workspace/models/model_142000000.pt",
    )

    az_network_ar = create_ar(
        tb,
        "/workspace/models/model_az_guess_overfit.pt",
    )

    az_network = create_az(
        tb,
        "/workspace/models/model_az_latest.pth",
    )

    p_run_comparison(
        az_network_ar,
        mcts_network_ar,

        az_network,

        1500
    )





