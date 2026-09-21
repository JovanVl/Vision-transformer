import torch

from .attention import MultiHeadAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, heads: int, embed_dim: int, dropout=0.15, mlp_ratio=4):
        super().__init__()
        self.norm_1 = torch.nn.LayerNorm(embed_dim)
        self.norm_2 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(heads, embed_dim, dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.dropout(self.attn(self.norm_1(x)))
        return x + self.dropout(self.mlp(self.norm_2(x)))

class Encoder(torch.nn.Module):
    def __init__(self, heads: int, embed_dim: int, N: int, dropout=0.15, mlp_ratio=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(heads, embed_dim, dropout, mlp_ratio) for _ in range(N)]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)