# lora.py
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        d_out, d_in = original_linear.weight.shape

        # freeze the original weights
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        self.bias = original_linear.bias

        # LoRA matrices: W + ΔW where ΔW = B·A
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)  # (r, d_in)
        self.B = nn.Parameter(torch.zeros(d_out, rank))         # (d_out, r)

        self.rank  = rank
        self.scale = alpha / rank   # scaling factor

    def forward(self, x):
        # original output + low-rank update
        original = x @ self.weight.T
        if self.bias is not None:
            original += self.bias
        lora_update = x @ self.A.T @ self.B.T   # x·Aᵀ·Bᵀ
        return original + self.scale * lora_update

def apply_lora(model, rank=4, alpha=16, target_modules=('Wq', 'Wv')):
    """Replace target Linear layers with LoRALinear."""
    for block in model.blocks:
        for name in target_modules:
            original = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear(original, rank, alpha))
    return model

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ── test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from model import TinyGPT
    import torch

    # load your trained Persian model
    checkpoint = torch.load('custom_data.pt')
    vocab_size = checkpoint['vocab_size']

    # create model and load weights
    model = TinyGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                    num_layers=4, max_seq_len=64)
    model.load_state_dict(torch.load('persian_model.pt'))

    print("── before LoRA ──")
    total, trainable = count_parameters(model)
    print(f"total:     {total:,}")
    print(f"trainable: {trainable:,}")
    print(f"ratio:     {trainable/total*100:.1f}%")

    # apply LoRA — only Wq and Wv get low-rank adapters
    model = apply_lora(model, rank=4, alpha=16, target_modules=('Wq', 'Wv'))

    print("\n── after LoRA ──")
    total, trainable = count_parameters(model)
    print(f"total:     {total:,}")
    print(f"trainable: {trainable:,}")
    print(f"ratio:     {trainable/total*100:.1f}%")

    # verify shapes
    print("\n── LoRA layer inspection ──")
    lora_layer = model.blocks[0].attn.Wq
    print(f"A shape: {lora_layer.A.shape}  (rank × d_in)")
    print(f"B shape: {lora_layer.B.shape}  (d_out × rank)")
    print(f"scale:   {lora_layer.scale}")

    # verify forward pass still works
    x = torch.randint(0, vocab_size, (1, 10))
    out = model(x)
    print(f"\nforward pass output: {out.shape}  ✓")

    # count LoRA params only
    lora_params = sum(p.numel() for name, p in model.named_parameters()
                      if 'lora' in name.lower() or
                      any(n in name for n in ['.A', '.B']))
    print(f"\nLoRA-only parameters: {lora_params:,}")