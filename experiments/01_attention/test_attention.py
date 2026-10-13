"""Reproducible tests for the custom attention implementation."""

import torch
import torch.nn.functional as F

from genai_building_blocks.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_non_causal_attention_matches_pytorch():
    torch.manual_seed(42)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)

    actual, weights = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (2, 3, 5, 8)
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_causal_attention_matches_pytorch():
    torch.manual_seed(42)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)
    causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))

    actual, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights.triu(diagonal=1), torch.zeros_like(weights))
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_attention_output_shapes_and_head_dimension():
    torch.manual_seed(42)
    model = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 10, 64)
    out = model(x)

    assert model.d_k == 16
    assert out.shape == x.shape


def test_attention_rejects_non_divisible_dimensions():
    try:
        MultiHeadAttention(d_model=10, num_heads=3)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for incompatible dimensions.")
