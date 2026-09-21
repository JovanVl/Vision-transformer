import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, P=16):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=P, stride=P)

    def forward(self, x: torch.Tensor):
        return self.patch_conv(x).flatten(2).transpose(1, 2)
