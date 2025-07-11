import torch
from torch import nn


class SelfAttentionMask(nn.Module):
    n_embd: int
    n_head: int

    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.resid_dropout = nn.Dropout(
            config.resid_pdrop
        )

    def forward(self, x):
        B, T, C = x.size()

        mask = (torch.tril(torch.ones(T, T, device=x.device))
                .bool())

        y, _ = self.attn(x, x, x, attn_mask=~mask)  

        y = self.resid_dropout(self.c_proj(y))
        return y