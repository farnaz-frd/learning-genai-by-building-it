# lora_finetune.py
import torch
import torch.nn.functional as F
from model import TinyGPT
from lora import apply_lora, count_parameters

# ── load data ──
checkpoint = torch.load('custom_data.pt')
train_data = checkpoint['train']
val_data   = checkpoint['val']
stoi       = checkpoint['stoi']
itos       = checkpoint['itos']
vocab_size = checkpoint['vocab_size']

decode = lambda l: ''.join(itos[i] for i in l)

def get_batch(split, block_size=64, batch_size=8):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# ── load pre-trained model ──
model = TinyGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                num_layers=4, max_seq_len=64)
model.load_state_dict(torch.load('LearnGenAI_#05/persian_model.pt'))

# ── apply LoRA — freeze Wq and Wv, add low-rank adapters ──
model = apply_lora(model, rank=4, alpha=16, target_modules=('Wq', 'Wv'))

# ── freeze EVERYTHING except LoRA matrices ──
for name, param in model.named_parameters():
    if '.A' not in name and '.B' not in name:
        param.requires_grad = False

total, trainable = count_parameters(model)
print(f"total params:     {total:,}")
print(f"trainable params: {trainable:,}  ({trainable/total*100:.2f}%)")
print(f"frozen params:    {total-trainable:,}\n")

# only pass LoRA params to optimizer
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer   = torch.optim.AdamW(lora_params, lr=1e-3)

# ── training loop ──
print("── LoRA fine-tuning ──")
best_val  = float('inf')
train_losses, val_losses = [], []

for step in range(3000):
    x, y   = get_batch('train')
    logits = model(x)
    loss   = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        with torch.no_grad():
            xv, yv   = get_batch('val')
            val_loss = F.cross_entropy(
                model(xv).view(-1, vocab_size), yv.view(-1))
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        marker = ' ← best' if val_loss.item() < best_val else ''
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), 'lora_best.pt')
        print(f"step {step:4d}  train {loss.item():.4f}  val {val_loss.item():.4f}{marker}")

# ── compare curves ──
print("\n── train vs val gap comparison ──")
print(f"{'step':<8} {'train':>8} {'val':>8} {'gap':>8}")
for i, step in enumerate(range(0, 3000, 300)):
    gap = val_losses[i] - train_losses[i]
    print(f"{step:<8} {train_losses[i]:>8.4f} {val_losses[i]:>8.4f} {gap:>8.4f}")

# ── generate ──
model.eval()
context = torch.zeros((1,1), dtype=torch.long)
with torch.no_grad():
    for _ in range(200):
        logits   = model(context[:, -64:])
        probs    = F.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, 1)
        context  = torch.cat([context, next_tok], dim=1)

print("\n── generated text (LoRA fine-tuned) ──")
print(decode(context[0].tolist()))

# add to end of lora_finetune.py
# compare: how much did the weights actually change?
print("\n── weight change analysis ──")
original = TinyGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                   num_layers=4, max_seq_len=64)
original.load_state_dict(torch.load('LearnGenAI_#05/persian_model.pt'))

# check Wq in block 0 — should be identical (frozen)
orig_wq = original.blocks[0].attn.Wq.weight
lora_wq = model.blocks[0].attn.Wq.weight

diff = (orig_wq - lora_wq).abs().max().item()
print(f"Wq weight change (frozen):     {diff:.8f}  ← should be 0.0")

# check FFN — should have changed (not frozen)
orig_ff = original.blocks[0].ffn.net[0].weight
lora_ff = model.blocks[0].ffn.net[0].weight

diff2 = (orig_ff - lora_ff).abs().max().item()
print(f"FFN weight change (trainable): {diff2:.6f}  ← should be > 0")

# show the actual LoRA delta being applied
A = model.blocks[0].attn.Wq.A
B = model.blocks[0].attn.Wq.B
delta = (B @ A)
print(f"\nLoRA delta (B·A) for Wq:")
print(f"  shape:     {delta.shape}")
print(f"  max value: {delta.abs().max().item():.6f}")
print(f"  mean abs:  {delta.abs().mean().item():.6f}")
print(f"  → this is the ONLY change to Wq's effective weight")