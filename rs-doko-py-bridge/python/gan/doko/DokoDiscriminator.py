import torch
import torch.nn as nn
from torch import Tensor
from typing import List

NUM_PHASES: int = 4
PHASE_EMBEDDINGS_DIM_SIZE: int = 4

NUM_PLAYERS_OR_NONE: int = 5
PLAYER_INPUT_EMBEDDINGS_DIM_SIZE: int = 4

NUM_RESERVATIONS_OR_NONE: int = 4
PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE: int = 4

NUM_CARDS_OR_NONE: int = 25
CARD_EMBEDDING_DIM_SIZE: int = 4

GUESSED_STATE_DIM: int = 48 * 3

class DokoDiscriminatorNetwork(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()

        self.phase_embeddings = nn.Embedding(
            num_embeddings=NUM_PHASES,
            embedding_dim=PHASE_EMBEDDINGS_DIM_SIZE
        )
        self.player_input_embeddings = nn.Embedding(
            num_embeddings=NUM_PLAYERS_OR_NONE,
            embedding_dim=PLAYER_INPUT_EMBEDDINGS_DIM_SIZE
        )
        self.reservations_embeddings = nn.Embedding(
            num_embeddings=NUM_RESERVATIONS_OR_NONE,
            embedding_dim=PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE
        )
        self.card_embeddings = nn.Embedding(
            num_embeddings=NUM_CARDS_OR_NONE,
            embedding_dim=CARD_EMBEDDING_DIM_SIZE
        )

        input_from_state: int = (
            PHASE_EMBEDDINGS_DIM_SIZE
            + PLAYER_INPUT_EMBEDDINGS_DIM_SIZE * 13
            + CARD_EMBEDDING_DIM_SIZE * 48
            + CARD_EMBEDDING_DIM_SIZE * 12 
            + PLAYER_RESERVATION_EMBEDDINGS_DIM_SIZE * 4
        )

        input_from_guessed: int = GUESSED_STATE_DIM  # 144

        input_to_first_hidden: int = input_from_state + input_from_guessed

        layers = []
        in_dim = input_to_first_hidden
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)

        self.last_layer = nn.Linear(in_dim, 1)

    def forward(self, state: Tensor, guessed_state: Tensor) -> Tensor:

        phase = state[:, 0:1]
        player_inputs = state[:, 1:14]
        card_inputs = state[:, 14:74]
        reservation_inputs = state[:, 74:78]

        x_phase = self.phase_embeddings(phase.long())
        x_player = self.player_input_embeddings(player_inputs.long())
        x_card = self.card_embeddings(card_inputs.long())
        x_reservation = self.reservations_embeddings(reservation_inputs.long())

        x_phase = x_phase.view(x_phase.size(0), -1)
        x_player = x_player.view(x_player.size(0), -1)
        x_card = x_card.view(x_card.size(0), -1)
        x_reservation = x_reservation.view(x_reservation.size(0), -1)

        guessed_state = guessed_state.view(guessed_state.size(0), -1)

        x = torch.cat([x_phase, x_player, x_card, x_reservation, guessed_state], dim=1)

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        x = self.last_layer(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    hidden_dims = [64, 64]
    net = DokoDiscriminatorNetwork(hidden_dims)

    state_example = torch.randint(0, 1, (2, 78))

    guessed_state_example = torch.randn(2, GUESSED_STATE_DIM)
    output = net.forward(state_example, guessed_state_example)

    print("Output-Shape:", output.shape)  
    print("Output-Werte:", output)
