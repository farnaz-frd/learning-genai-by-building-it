"""A small LayerNorm implementation for comparison with PyTorch."""

import torch
from torch import Tensor, nn


class MyLayerNorm(nn.Module):
    """Layer normalization over the last dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
