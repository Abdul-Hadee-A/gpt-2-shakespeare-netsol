import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 1000
    block_size: int = 256     # how many tokens the model can look at
    n_layers: int = 6
    n_heads: int = 6          # query heads
    n_kv_heads: int = 2       # shared key/value heads (the GQA part)
    d: int = 384              # size of the model
    dropout: float = 0.1

class GQA(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.nh = n_heads
        self.nkv = n_kv_heads
        self.hd = d // n_heads            # size of each head
        self.q = nn.Linear(d, n_heads * self.hd, bias=False)
        self.k = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        # copy each shared key/value so every query head has a partner
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # is_causal=True stops a word from peeking at future words
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o(y)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = GQA(cfg.d, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d, 4 * cfg.d),
            nn.GELU(),
            nn.Linear(4 * cfg.d, cfg.d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention, with a shortcut
        x = x + self.mlp(self.ln2(x))    # MLP, with a shortcut
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d)    # what the token is
        self.pos = nn.Embedding(cfg.block_size, cfg.d)    # where it is
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight                # weight tying (GPT-2 does this)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    cfg = Config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))   # fake tokens
    logits, loss = model(idx, idx)
    print(logits.shape)
    print(loss.item())
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")