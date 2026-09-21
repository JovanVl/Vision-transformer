import torch
import math

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    attn_p = scores.softmax(dim=-1)
    
    if dropout is not None:
        attn_p = dropout(attn_p)

    return torch.matmul(attn_p, value), attn_p

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads: int, embed_dim: int, dropout=0.15):
        super().__init__()
        assert embed_dim % heads == 0
        
        self.d_k = embed_dim // heads
        self.heads = heads
        self.attn = None
        
        self.q_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.output_linear = torch.nn.Linear(embed_dim, embed_dim)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        batch, seq_len, dim = x.shape
        
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        query, key, value = [t.view(batch, seq_len, self.heads, self.d_k).transpose(1, 2) for t in [query, key, value]]

        output, self.attn = attention(query, key, value, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)

        return self.output_linear(output)