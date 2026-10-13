"""Attention implementations used in the learning experiments."""

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
):
    """Compute scaled dot-product attention.

    Q, K, and V may have any matching leading dimensions, e.g.
    (sequence, d_k), (batch, heads, sequence, d_k).

    A supplied mask uses True/non-zero for positions that are allowed
    and False/zero for positions that should be masked.
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with combined per-layer projections."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.h = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, causal: bool = True):
        """Args:
        x: (batch, sequence_length, d_model)
        causal: if True, prevent tokens from attending to future tokens.
        """
        if x.ndim != 3:
            raise ValueError("Expected x with shape (batch, sequence, d_model).")

        batch_size, seq_len, channels = x.shape
        if channels != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, received {channels}."
            )

        Q = self.Wq(x).view(
            batch_size, seq_len, self.h, self.d_k
        ).transpose(1, 2)
        K = self.Wk(x).view(
            batch_size, seq_len, self.h, self.d_k
        ).transpose(1, 2)
        V = self.Wv(x).view(
            batch_size, seq_len, self.h, self.d_k
        ).transpose(1, 2)

        mask = None
        if causal:
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )

        out, _weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        return self.Wo(out)
