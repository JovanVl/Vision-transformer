import torch

from .transformer_encoder import Encoder

from .patch_embedding import PatchEmbedding

class ViT(torch.nn.Module):
    def __init__(self, img_size: int, in_channels: int, num_classes: int, embed_dim: int, 
                 heads: int, N, P=16, mlp_ratio=4, dropout=0.15):
        super().__init__()
        assert img_size % P == 0
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, P)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = torch.nn.Parameter(torch.randn(1, (img_size // P)**2 + 1, embed_dim))
        self.encoder = Encoder(heads, embed_dim, N, dropout, mlp_ratio)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding
        x = self.encoder(x)
        cls_output = x[:,0]
        
        return self.head(cls_output)