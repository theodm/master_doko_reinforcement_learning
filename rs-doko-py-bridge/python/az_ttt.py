import copy
import dataclasses
import multiprocessing
import sys
import time
from typing import Self, Tuple

import numpy
import torch
from jaxtyping import Float, Int
from numpy import array
from torch import nn, Tensor
from torch.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from TensorboardController import TensorboardController

@dataclasses.dataclass
class AlphaZeroTicTacToeNetworkConfig:
    device: str


class AlphaZeroTicTacToeNetwork(nn.Module):

    def __init__(self, config: AlphaZeroTicTacToeNetworkConfig):
        super(AlphaZeroTicTacToeNetwork, self).__init__()

        self.config = config

        self.net = nn.Sequential(
            nn.Linear(9, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, 32, bias=True),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, 9, bias=True)
        )
        self.value_head = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, 2, bias=True)
        )

        self.to(config.device)

    def forward(self, x: Int[Tensor, "batch 9"]) -> (Float[Tensor, "batch 2"], Float[Tensor, "batch 9"]):
        x = self.net(x.to(dtype=torch.float))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return value, policy

    def log_embeddings(self, writer: SummaryWriter, step: int):
        pass


@dataclasses.dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    compile: bool


class NetworkWithTrainer:
    module: AlphaZeroTicTacToeNetwork
    config: AlphaZeroTicTacToeNetworkConfig
    train_config: TrainConfig

    optimizer: torch.optim.Optimizer
    policy_criterion: nn.CrossEntropyLoss
    value_criterion: nn.MSELoss

    tb: TensorboardController

    number_of_experiences: int

    def __init__(
            self,
            module: AlphaZeroTicTacToeNetwork,
            config: AlphaZeroTicTacToeNetworkConfig,
            train_config: TrainConfig,
            tb: TensorboardController,
            number_of_experiences: int = 0
    ):
        self.module = module
        self.config = config
        self.train_config = train_config

        self.scaler = torch.amp.GradScaler(self.config.device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=train_config.learning_rate)

        self.value_criterion = nn.MSELoss().to(config.device)
        self.policy_criterion = nn.CrossEntropyLoss().to(config.device)

        self.number_of_experiences = number_of_experiences
        self.tb = tb

        if self.train_config.compile:
            _time = time.time()
            self.module.compile()
            print(f"Model compiled in {_time - time.time()} seconds")

    def predict(self, x: list[int]) -> (list[float], list[float]):
        with (torch.no_grad()):
            # with autocast(self.config.device):
            x_as_tensor = torch.tensor(x, device=config.device).view([-1, 9])

            value, policy = self.module.forward(x_as_tensor)

            res = (value.view([-1]).cpu().numpy().tolist(), policy.view([-1]).cpu().numpy().tolist())

            return res

    def fit(self, x: list[int], target: Tuple[list[float], list[float]]):
        states_as_tensor: Int[Tensor, "batch 9"] = torch.tensor(x, device=config.device).view([-1, 9])
        values_as_tensor: Float[Tensor, "batch 2"] = torch.tensor(target[0], device=config.device).view([-1, 2])
        policies_as_tensor: Float[Tensor, "batch 9"] = torch.tensor(target[1], device=config.device).view([-1, 9])

        dataset = TensorDataset(states_as_tensor, values_as_tensor, policies_as_tensor)
        dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size) # Nach aktueller Logik bereits geshuffled!, shuffle=True)

        optimizer = self.optimizer

        start_time = time.time()
        total_experiences = 0

        for i, (states_batch, value_batch, policy_batch) in enumerate(dataloader):
            states_batch = states_batch.to(self.config.device)
            value_batch = value_batch.to(self.config.device)
            policy_batch = policy_batch.to(self.config.device)

            # Forward pass
            (y_value, y_policy) = self.module(states_batch)
            loss_value = self.value_criterion(y_value, value_batch)
            loss_policy = self.policy_criterion(y_policy, policy_batch)

            loss = loss_value + loss_policy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            total_experiences += len(states_batch)
            self.number_of_experiences += len(states_batch)
            self.tb.scalar("loss", loss.item(), self.number_of_experiences)

            previous_experiences = self.number_of_experiences - len(states_batch)
            if (self.number_of_experiences // 50000) > (previous_experiences // 50000):
                self.module.log_embeddings(self.tb.writer, self.number_of_experiences)

        # Gesamtzeit für das Training berechnen
        total_time = time.time() - start_time

        # Durchschnittliche Zeit pro Experience berechnen
        if total_experiences > 0:
            avg_time_per_experience = total_time / total_experiences
            self.tb.scalar("time_per_experience", avg_time_per_experience, self.number_of_experiences)


    def save(self, path: str):
        """ Speichert das Modell und den Optimizer in einer Datei """
        checkpoint = {
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_config": self.train_config,
            "config": self.config
        }
        torch.save(checkpoint, path)
        print(f"Modell gespeichert unter {path}")

    def load(self, path: str):
        """ Lädt das Modell und den Optimizer aus einer Datei """
        checkpoint = torch.load(path)

        self.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Falls train_config oder config geändert wurden, hier updaten
        self.train_config = checkpoint["train_config"]
        self.config = checkpoint["config"]

        print(f"Modell geladen von {path}")

    def number_of_trained_examples(self) -> int:
        return self.number_of_experiences

    def clone(self, device_index: int) -> Self:
        config = copy.deepcopy(self.config)

        if config.device.startswith("cuda"):
            config.device = f"cuda:{device_index % torch.cuda.device_count()}"

        return NetworkWithTrainer(
            module=copy.deepcopy(self.module).to(config.device),
            config=config,
            train_config=self.train_config,
            tb=self.tb,
            number_of_experiences=self.number_of_experiences
        )

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

run_name = "tic_tac_toe"

tb = TensorboardController(f"runs/{run_name}")

config = AlphaZeroTicTacToeNetworkConfig(
    device="cuda",
)

train_config = TrainConfig(
    batch_size=128,
    learning_rate=0.001,
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

from alpha_zero_training import call_alpha_zero_training

call_alpha_zero_training(
    game="tic-tac-toe",

    number_of_batch_receivers=2,
    batch_nn_max_delay_in_ms=10,
    batch_nn_buffer_size=250,

    checkpoint_every_n_epochs=1,
    probability_of_keeping_experience=1.0,

    games_per_epoch=500,
    max_concurrent_games=768,

    mcts_iterations=300,
    dirichlet_alpha=1.00,
    dirichlet_epsilon=0.50,

    puct_exploration_constant=4.0,

    min_or_max_value=1.0,

    value_target="default",

    node_arena_capacity=2500,
    state_arena_capacity=2500,
    cache_size_batch_processor=5000,

    mcts_workers=4,
    max_concurrent_games_in_evaluation=768,
    evaluation_every_n_epochs=1,
    skip_evaluation=False,

    network=network,
    tensorboard_controller=tb,

    epoch_start=None,
    temperature=1.25
)