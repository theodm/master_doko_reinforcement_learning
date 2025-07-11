from torch import nn
from SelfAttention import SelfAttention
from SelfAttentionMask import SelfAttentionMask

class TransformerBlock(nn.Module):
    def __init__(self, config, masked=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if masked:
            self.attn = SelfAttentionMask(config)
        else:
            self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))

        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x