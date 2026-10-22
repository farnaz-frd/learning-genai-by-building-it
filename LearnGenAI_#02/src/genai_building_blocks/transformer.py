"""Minimal decoder-only Transformer components for educational experiments."""

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside a Transformer block."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with residual connections."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal=True)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Small character-level GPT-style language model.

    This is an educational implementation, not a production language model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        if vocab_size <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, num_layers, and max_seq_len must be positive.")

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        if idx.ndim != 2:
            raise ValueError("Expected token IDs with shape (batch, sequence).")
        _, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Autoregressively sample tokens from the model."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
