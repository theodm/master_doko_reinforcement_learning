import copy
import csv
import dataclasses
import multiprocessing
import os
import sys
import time
from typing import Self

from rs_doko_py_bridge import execute_impi
import numpy
import torch
from jaxtyping import Float, Int
from numpy import array
from torch import nn, Tensor
from torch.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from TransformerBlock import TransformerBlock
from TensorboardController import TensorboardController

from light_dataloader import TensorDataLoader

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

class PerClassAttentionHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 33):
        super().__init__()
        self.card_embeddings = nn.Parameter(torch.randn(num_classes, d_model))  # Query für jede Karte
        self.scale = d_model ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        Q = self.card_embeddings  # [classes, d_model]
        K = x  # [batch, seq_len, d_model]
        V = x

        scores = torch.einsum("cd,bsd->bcs", Q, K) * self.scale  # [batch, classes, seq_len]

        weights = torch.softmax(scores, dim=-1)  # über seq_len

        attended = torch.einsum("bcs,bsd->bcd", weights, V)  # [batch, classes, d_model]

        logits = torch.einsum("bcd,cd->bc", attended, Q)  # [batch, classes]

        return logits

@dataclasses.dataclass
class ImperfectInformationNetworkConfig:
    device: str

    n_embd: int
    n_head: int

    attn_pdrop: float
    resid_pdrop: float

    embd_pdrop: float

    n_layer: int

GAME_OBJECT_VOCAB_SIZE = 43
GAME_POSITION_VOCAB_SIZE = 57
GAME_PLAYER_WHO_PLAYED_CARD_VOCAB_SIZE = 5
GAME_SUBPOSITION_VOCAB_SIZE = 13
GAME_TEAM_VOCAB_SIZE = 3
GAME_PLAYER_WHO_TO_GUESS_CARD_FOR_VOCA_SIZE = 5

class ImperfectInformationNetwork(nn.Module):

    def __init__(self, config: ImperfectInformationNetworkConfig):
        super(ImperfectInformationNetwork, self).__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            game_object_e = nn.Embedding(GAME_OBJECT_VOCAB_SIZE, config.n_embd),
            position_e = nn.Embedding(GAME_POSITION_VOCAB_SIZE, config.n_embd),
            player_played_e = nn.Embedding(GAME_PLAYER_WHO_PLAYED_CARD_VOCAB_SIZE, config.n_embd),
            subposition_e = nn.Embedding(GAME_SUBPOSITION_VOCAB_SIZE, config.n_embd),
            team_e = nn.Embedding(GAME_TEAM_VOCAB_SIZE, config.n_embd),
            player_guess_e = nn.Embedding(GAME_PLAYER_WHO_TO_GUESS_CARD_FOR_VOCA_SIZE, config.n_embd),

            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = PerClassAttentionMLPHead(config.n_embd, num_classes=33)

        self.to(self.config.device)

    def forward(self, x: Int[Tensor, "batch 311"]) -> Float[Tensor, "batch 33"]:
        game_objects = self.transformer.game_object_e(x[:, 0:62])
        positions = self.transformer.position_e(x[:, 62:124])
        player_played = self.transformer.player_played_e(x[:, 124:186])
        subposition = self.transformer.subposition_e(x[:, 186:248])
        team = self.transformer.team_e(x[:, 248:310])

        player_guess = (self
                        .transformer
                        .player_guess_e(x[:, 310:311])
                        .expand(-1, 62, -1))

        x = self.transformer.drop(game_objects + positions + player_played + subposition + team + player_guess)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Zum Schluss die lineare Schicht
        x = self.lm_head(x)

        return x

    def log_embeddings(self, writer: SummaryWriter, step: int):
        """
        Speichert die Embeddings in TensorBoard.
        """
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
                "NotRevealed",
                "NoneYet",
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
            tag="player_played_embeddings",
            global_step=step
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
            self.transformer.player_guess_e.weight.detach().cpu(),
            metadata=[
                "None",
                "Bottom",
                "Left",
                "Top",
                "Right"
            ],
            tag="phase_embeddings",
            global_step=step
        )

@dataclasses.dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    compile: bool

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model)) 
        self.scale = d_model ** 0.5

    def forward(self, x): 
        q = self.query.expand(x.size(0), -1).unsqueeze(1)

        attn_scores = torch.matmul(q, x.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        pooled = torch.matmul(attn_weights, x).squeeze(1)
        return pooled

class CardDotProductHead(nn.Module):
    def __init__(self, d_model, num_cards=33):
        super().__init__()
        self.card_embeddings = nn.Parameter(torch.randn(num_cards, d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x): 
        context = x.mean(dim=1)         
        context = self.proj(context)     
        logits = torch.matmul(context, self.card_embeddings.T)  
        return logits

class NetworkWithTrainer:
    module: ImperfectInformationNetwork
    config: ImperfectInformationNetworkConfig
    train_config: TrainConfig

    optimizer: torch.optim.Optimizer
    criterion: nn.CrossEntropyLoss

    tb: TensorboardController

    number_of_experiences: int

    def __init__(
            self,
            module: ImperfectInformationNetwork,
            config: ImperfectInformationNetworkConfig,
            train_config: TrainConfig,
            tb: TensorboardController,
            number_of_experiences: int = 0
    ):
        self.module = module
        self.config = config
        self.train_config = train_config

        self.scaler = torch.amp.GradScaler(self.config.device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=train_config.learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.config.device)

        self.number_of_experiences = number_of_experiences
        self.tb = tb

        if self.train_config.compile:
            _time = time.time()
            print("Compiling model...")
            self.module.compile()
            print(f"Model compiled in {_time - time.time()} seconds")

    def predict(self, x: list[int]) -> list[float]:
        self.module.eval()
        with (((torch.no_grad()))):
            # with autocast(self.config.device):
            x_as_tensor = torch.tensor(x, device=self.config.device).view([-1, 311])

            res = self.module.forward(x_as_tensor)

            res = res.reshape(-1).cpu().numpy().tolist()

            self.module.train()
            return res

    def fit(self, x: list[int], y: list[float]):
        x_as_tensor: Int[Tensor, "batch 311"] = torch.tensor(x, device='cpu').view([-1, 311])
        y_as_tensor: Float[Tensor, "batch"] = torch.tensor(y, device='cpu').view([-1, 33])

        optimizer = self.optimizer
        criterion = self.criterion

        start_time = time.time()
        total_experiences = 0

        batch_size = self.train_config.batch_size
        num_samples = x_as_tensor.shape[0]

        print(f"Training on {num_samples} samples")
        print(f"Training on {num_samples} samples")
        print(f"Training on {num_samples} samples")
        print(f"Training on {num_samples} samples")
        print(f"Training on {num_samples} samples")

        for i in range(0, num_samples, batch_size):
            x_batch = x_as_tensor[i:i+batch_size].to(self.config.device)
            y_batch = y_as_tensor[i:i+batch_size].to(self.config.device)

            # Forward pass
            y_pred = self.module(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            batch_len = len(x_batch)
            total_experiences += batch_len
            self.number_of_experiences += batch_len
            self.tb.scalar("loss", loss.item(), self.number_of_experiences)

            if i // batch_size % 250 == 0:
                print(f"Batch {i // batch_size}/{(num_samples + batch_size - 1) // batch_size}: Loss={loss.item():.4f}")

            previous_experiences = self.number_of_experiences - batch_len
            if (self.number_of_experiences // 50000) > (previous_experiences // 50000):
                self.module.log_embeddings(self.tb.writer, self.number_of_experiences)

        # Gesamtzeit für das Training berechnen
        total_time = time.time() - start_time

        # Durchschnittliche Zeit pro Experience berechnen
        if total_experiences > 0:
            avg_time_per_experience = total_time / total_experiences
            self.tb.scalar("time_per_experience", avg_time_per_experience, self.number_of_experiences)

        # Modell speichern
        self.save(f"models5/model_{self.number_of_experiences}.pt")

    def save(self, path: str):
        """ Speichert das Modell und den Optimizer in einer Datei """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

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

        self.train_config = checkpoint["train_config"]
        self.config = checkpoint["config"]

        print(f"Modell geladen von {path}")

    def number_of_trained_examples(self) -> int:
        return self.number_of_experiences

    def clone(self) -> Self:
        return NetworkWithTrainer(
            module=copy.deepcopy(self.module),
            config=self.config,
            train_config=self.train_config,
            tb=self.tb,
            number_of_experiences=self.number_of_experiences
        )

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    run_name = "hopefully_final2"

    tb = TensorboardController(f"runs/{run_name}")

    config = ImperfectInformationNetworkConfig(
        device="cuda",

        n_embd=196,
        n_layer=6,
        n_head=4,

        attn_pdrop = 0.1,
        resid_pdrop = 0.1,
        embd_pdrop = 0.1
    )

    train_config = TrainConfig(
        batch_size=128,
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

    print(f"Number of simultaneous games: {multiprocessing.cpu_count() - 1}")

    execute_impi(
        multiprocessing.cpu_count() - 1,
        network,
        0.1,
        5000,
        1,
        tb
    )

