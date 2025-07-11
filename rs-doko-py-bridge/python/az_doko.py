import copy
import dataclasses
import faulthandler
import multiprocessing
import os
import sys
import time
from typing import Self, Tuple

import numpy
import numpy as np
import torch
from jaxtyping import Float, Int
from numpy import array
from torch import nn, Tensor
from torch.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from TensorboardController import TensorboardController


import time
from SelfAttention import SelfAttention
from TransformerBlock import TransformerBlock

faulthandler.enable(all_threads=True)

class PerClassAttentionMLPHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 33):
        super().__init__()
        self.card_queries = nn.Parameter(torch.randn(num_classes, d_model))
        self.scale = d_model ** -0.5

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x: Tensor) -> Tensor:
        Q = self.card_queries  # [classes, d_model]
        K = V = x  # [batch, seq_len, d_model]

        scores = torch.einsum("cd,bsd->bcs", Q, K) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bcs,bsd->bcd", weights, V)  # [batch, classes, d_model]

        # Jetzt über alle Karten iterieren und eigene MLP anwenden
        logits = []
        for c, mlp in enumerate(self.mlps):
            logits_c = mlp(attended[:, c, :])  # [batch, 1]
            logits.append(logits_c)

        logits = torch.cat(logits, dim=1)  # [batch, classes]
        return logits


@dataclasses.dataclass
class AlphaZeroTicTacToeNetworkConfig:
    device: str

    n_embd: int = 160

    n_head: int = 4

    n_layer: int = 4
    n_layer_policy: int = 1
    n_layer_value: int = 1

    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    expected_batch_size: int = 2048

GAME_OBJECT_VOCAB_SIZE = 43
GAME_POSITION_VOCAB_SIZE = 57
GAME_PLAYER_WHO_PLAYED_CARD_VOCAB_SIZE = 5
GAME_SUBPOSITION_VOCAB_SIZE = 13
GAME_TEAM_VOCAB_SIZE = 3
PHASE_VOCAB_SIZE = 3


class AlphaZeroTicTacToeNetwork(nn.Module):

    def __init__(self, config: AlphaZeroTicTacToeNetworkConfig):
        super(AlphaZeroTicTacToeNetwork, self).__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            game_object_e = nn.Embedding(GAME_OBJECT_VOCAB_SIZE, config.n_embd),
            position_e = nn.Embedding(GAME_POSITION_VOCAB_SIZE, config.n_embd),
            player_played_e = nn.Embedding(GAME_PLAYER_WHO_PLAYED_CARD_VOCAB_SIZE, config.n_embd),
            subposition_e = nn.Embedding(GAME_SUBPOSITION_VOCAB_SIZE, config.n_embd),
            team_e = nn.Embedding(GAME_TEAM_VOCAB_SIZE, config.n_embd),
            phase_e = nn.Embedding(PHASE_VOCAB_SIZE, config.n_embd),

            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            p_h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer_policy)]),
            v_h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer_value)]),

            p_ln_f = nn.LayerNorm(config.n_embd),
            v_ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.value_head = PerClassAttentionMLPHead(config.n_embd, num_classes=4)
        self.policy_head = PerClassAttentionMLPHead(config.n_embd, num_classes=39)

        self.to(self.config.device)

    def forward(self, x: Int[Tensor, "batch 311"]) -> (Float[Tensor, "batch 4"], Float[Tensor, "batch 39"]):
        game_objects = self.transformer.game_object_e(x[:, 0:62])
        positions = self.transformer.position_e(x[:, 62:124])
        player_played = self.transformer.player_played_e(x[:, 124:186])
        subposition = self.transformer.subposition_e(x[:, 186:248])
        team = self.transformer.team_e(x[:, 248:310])
        phase = (self.transformer.phase_e(x[:, 310:311])
                 .expand(-1, 62, -1))

        x = self.transformer.drop(game_objects + positions + player_played + subposition + team + phase)

        for block in self.transformer.h:
            x = block(x)

        x_value = x
        x_policy = x

        for block in self.transformer.p_h:
            x_policy = block(x_policy)

        for block in self.transformer.v_h:
            x_value = block(x_value)

        x_policy = self.transformer.p_ln_f(x_policy)
        x_value = self.transformer.v_ln_f(x_value)

        value = self.value_head(x_value)
        policy = self.policy_head(x_policy)

        return value, policy

    def log_embeddings(self, writer: SummaryWriter, step: int):
        writer.add_embedding(
            self.transformer.game_object_e.weight.detach().cpu(),
            metadata=[
                # 0
                "None",
                # 1
                "♥10",
                "♣Q",
                "♠Q",
                "♥Q",
                "♦Q",
                "♣J",
                "♠J",
                "♥J",
                "♦J",
                "♦A",
                "♦10",
                "♦K",
                "♦9",
                "♣A",
                "♣10",
                "♣K",
                "♣9",
                "♠A",
                "♠10",
                "♠K",
                "♠9",
                "♥A",
                "♥K",
                "♥9",
                # 25
                "Gesund",
                "Hochzeit",
                "♦-Solo",
                "♥-Solo",
                "♠-Solo",
                "♣-Solo",
                "Q-Solo",
                "J-Solo",
                "T-Solo",
                "frei",
                "frei",
                # 36
                "frei",
                "Keine Ankündigung",
                "Re/Kontra / CounterReContra",
                "Keine 90",
                "Keine 60",
                "Keine 30",
                "Schwarz",
            ],
            tag="game_object_embeddings",
            global_step=step
        )

        writer.add_embedding(
            self.transformer.position_e.weight.detach().cpu(),
            metadata=[
                "None",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
                "32",
                "33",
                "34",
                "35",
                "36",
                "37",
                "38",
                "39",
                "40",
                "41",
                "42",
                "43",
                "44",
                "45",
                "46",
                "47",
                "48",
                "49",
                "50",
                "51",
                "52",
                "Bottom",
                "Left",
                "Top",
                "Right"
            ],
            tag="position_embeddings",
            global_step=step
        )

        writer.add_embedding(
            self.transformer.player_played_e.weight.detach().cpu(),
            metadata=[
                "None",
                "Bottom",
                "Left",
                "Top",
                "Right"
            ],
        )

        writer.add_embedding(
            self.transformer.subposition_e.weight.detach().cpu(),
            metadata=[
                "None",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "1. Karte",
                "2. Karte",
            ],
            tag="subposition_embeddings",
            global_step=step
        )

        writer.add_embedding(
            self.transformer.team_e.weight.detach().cpu(),
            metadata=[
                "None",
                "Re",
                "Kontra"
            ],
            tag="team_embeddings",
            global_step=step
        )

        writer.add_embedding(
            self.transformer.phase_e.weight.detach().cpu(),
            metadata=[
                "Reservation",
                "Announcement",
                "PlayCard"
            ],
            tag="phase_embeddings",
            global_step=step
        )

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

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=train_config.learning_rate)

        self.value_criterion = nn.MSELoss().to(self.config.device)
        self.policy_criterion = nn.CrossEntropyLoss().to(self.config.device)

        self.number_of_experiences = number_of_experiences
        self.tb = tb

        self.input_shape = (config.expected_batch_size, 311)
        self.input_tensor = torch.empty(self.input_shape, dtype=torch.int32, device=self.config.device)

        if self.train_config.compile:
            import torch_tensorrt
            _time = time.time()
            print(torch._dynamo.list_backends())
            print("Compiling model...")
            self.module.compile(backend="tensorrt")
            print(f"Model compiled in {_time - time.time()} seconds")

    def predict(self, x: list[int]) -> (list[float], list[float]):
        self.module.eval()
        with torch.inference_mode():
            # with autocast(self.config.device):
            # 1. Liste → NumPy-Array, int32 (für Embeddings)
            x_np = np.asarray(x, dtype=np.int32)  # Shape: [batch_size * 311]

            # 2. In Tensor umwandeln (CPU), dann in GPU-Tensor kopieren
            torch_x = torch.from_numpy(x_np)

            # 3. Copy in vorinitialisierten Tensor (nur so viel wie nötig)
            batch_size = len(x) // 311
            assert batch_size <= self.input_tensor.shape[0], \
                f"Batch size {batch_size} exceeds preallocated batch size {self.input_tensor.shape[0]}"

            self.input_tensor[:batch_size].copy_(torch_x.view(-1, 311))

            value, policy = self.module.forward(self.input_tensor[:batch_size])

            res = (value.view([-1]).cpu().numpy().tolist(), policy.view([-1]).cpu().numpy().tolist())

            return res

    def fit(self, x: list[int], target: Tuple[list[float], list[float]]):
        self.module.train()

        states_as_tensor: Int[Tensor, "batch 311"] = torch.tensor(x, device=self.config.device).view([-1, 311])
        values_as_tensor: Float[Tensor, "batch 4"] = torch.tensor(target[0], device=self.config.device).view([-1, 4])
        policies_as_tensor: Float[Tensor, "batch 39"] = torch.tensor(target[1], device=self.config.device).view([-1, 39])

        dataset = TensorDataset(states_as_tensor, values_as_tensor, policies_as_tensor)

        optimizer = self.optimizer

        total_start_time = time.time()

        epochs = 1
        for epoch in range(epochs):  # Mehrfaches Training mit denselben Daten
            dataloader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)

            epoch_start_time = time.time()
            total_experiences = 0

            for i, (states_batch, value_batch, policy_batch) in enumerate(dataloader):
                states_batch = states_batch.to(self.config.device)
                value_batch = value_batch.to(self.config.device)
                policy_batch = policy_batch.to(self.config.device)

                # Forward pass
                y_value, y_policy = self.module(states_batch)
                loss_value = self.value_criterion(y_value, value_batch)
                loss_policy = self.policy_criterion(y_policy, policy_batch)

                loss = loss_value + loss_policy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_experiences += len(states_batch)
                self.number_of_experiences += len(states_batch)

                # TensorBoard Logging
                self.tb.scalar("loss", loss.item(), self.number_of_experiences)
                self.tb.scalar("loss_value", loss_value.item(), self.number_of_experiences)
                self.tb.scalar("loss_policy", loss_policy.item(), self.number_of_experiences)

                if i % 250 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataloader)}: Loss={loss.item():.4f}, Loss Value={loss_value.item():.4f}, Loss Policy={loss_policy.item():.4f}")

                previous_experiences = self.number_of_experiences - len(states_batch)
                if (self.number_of_experiences // 50000) > (previous_experiences // 50000):
                    self.module.log_embeddings(self.tb.writer, self.number_of_experiences)

            # Zeitmessung für diese Epoche
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} abgeschlossen in {epoch_time:.2f} Sekunden.")

        # Gesamtzeit fürs Training
        total_time = time.time() - total_start_time
        avg_time_per_experience = total_time / total_experiences if total_experiences > 0 else 0
        self.tb.scalar("time_per_experience", avg_time_per_experience, self.number_of_experiences)

        # Modell speichern
        self.save_with_limit("models7", f"model_{self.number_of_experiences}.pth")
        print(f"Training abgeschlossen. Gesamtzeit: {total_time:.2f} Sekunden.")

    def save_with_limit(self, folder_path: str, checkpoint_name: str):
        """Speichert das Modell und den Optimizer, hält nur die letzten 50 Checkpoints"""

        # Sicherstellen, dass der Ordner existiert
        os.makedirs(folder_path, exist_ok=True)

        # Erstelle den vollständigen Pfad für das neue Modell
        checkpoint_path = os.path.join(folder_path, checkpoint_name)

        # Speichern des aktuellen Modells
        checkpoint = {
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_config": self.train_config,
            "config": self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Modell gespeichert unter {checkpoint_path}")

        # Liste der gespeicherten Checkpoints abrufen und nach Zeit sortieren (älteste zuerst)
        checkpoints = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pth")],
            key=os.path.getctime
        )

        # Wenn mehr als 50 Checkpoints existieren, die ältesten löschen
        while len(checkpoints) > 50:
            oldest_checkpoint = checkpoints.pop(0)  # Ältestes Element entfernen
            os.remove(oldest_checkpoint)
            print(f"Alter Checkpoint gelöscht: {oldest_checkpoint}")

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
        checkpoint = torch.load(path, weights_only=False)

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
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    run_name = "hopefully_final_az2_fort"

    tb = TensorboardController(f"runs/{run_name}")

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
    #
    network.load("models6/model_29107239.pth")
    # print("Model loaded")

    from alpha_zero_training import call_alpha_zero_training

    call_alpha_zero_training(
        game="doko",

        number_of_batch_receivers=3,
        batch_nn_max_delay_in_ms=1,
        batch_nn_buffer_size=config.expected_batch_size,

        checkpoint_every_n_epochs=1,
        probability_of_keeping_experience=1.0,

        games_per_epoch=8192,
        max_concurrent_games=8192,

        mcts_iterations=200,
        dirichlet_alpha=0.8,
        dirichlet_epsilon=0.3,

        puct_exploration_constant=4.0,

        min_or_max_value=1.0,

        value_target="avg",

        node_arena_capacity=200*15,
        state_arena_capacity=200*15,
        cache_size_batch_processor=400,

        mcts_workers=30,
        max_concurrent_games_in_evaluation=2048,
        evaluation_every_n_epochs=1,
        skip_evaluation=False,

        network=network,
        tensorboard_controller=tb,

        epoch_start=79,
        temperature=0.5
    )