# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h   = num_heads
        self.d_k = d_model // num_heads
        self.Wq  = nn.Linear(d_model, d_model, bias=False)
        self.Wk  = nn.Linear(d_model, d_model, bias=False)
        self.Wv  = nn.Linear(d_model, d_model, bias=False)
        self.Wo  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        Q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        mask = torch.tril(torch.ones(T, T, device=x.device)) if causal else None
        out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.blocks    = nn.Sequential(*[TransformerBlock(d_model, num_heads)
                                         for _ in range(num_layers)])
        self.ln_final  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok  = self.token_emb(idx)
        pos  = self.pos_emb(torch.arange(T, device=idx.device))
        x    = tok + pos
        x    = self.blocks(x)
        x    = self.ln_final(x)
        return self.head(x)