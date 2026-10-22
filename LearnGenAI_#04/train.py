# # prepare_custom.py  — run this once
# import torch, os

# text = open('custom.txt', encoding='utf-8').read()
# print(f"dataset size: {len(text):,} characters")

# chars = sorted(set(text))
# print(f"vocab size:   {len(chars)} unique characters")
# print(f"sample:       {text[:200]}")

# stoi = {c:i for i,c in enumerate(chars)}
# itos = {i:c for i,c in enumerate(chars)}

# data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
# n    = int(0.9 * len(data))

# torch.save({'train': data[:n], 'val': data[n:],
#             'stoi': stoi, 'itos': itos,
#             'vocab_size': len(chars)}, 'custom_data.pt')
# print("saved → custom_data.pt")

#___________________________ train ___________________________________________________________________________
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from model import TinyGPT

# ── paste your full model code here (all classes from attention.py) ──
# scaled_dot_product_attention, MultiHeadAttention, FeedForward,
# TransformerBlock, TinyGPT — all of them

# ── load data ──
checkpoint = torch.load('custom_data.pt')
train_data  = checkpoint['train']
val_data    = checkpoint['val']
stoi        = checkpoint['stoi']
itos        = checkpoint['itos']
vocab_size  = checkpoint['vocab_size']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

def get_batch(split, block_size=64, batch_size=8):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# ── model — smaller because dataset is tiny ──
torch.manual_seed(42)
model = TinyGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                num_layers=4, max_seq_len=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

total = sum(p.numel() for p in model.parameters())
print(f"parameters: {total:,}")
print(f"vocab size: {vocab_size}")
print(f"train tokens: {len(train_data):,}\n")

# ── train ──
for step in range(3000):
    x, y   = get_batch('train')
    logits = model(x)
    loss   = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        # also compute val loss
        with torch.no_grad():
            xv, yv = get_batch('val')
            val_loss = F.cross_entropy(
                model(xv).view(-1, vocab_size), yv.view(-1))
        print(f"step {step:4d}  train {loss.item():.4f}  val {val_loss.item():.4f}")

# ── save model ──
torch.save(model.state_dict(), 'persian_model.pt')
print("\nmodel saved → persian_model.pt")

# ── generate ──
model.eval()
context = torch.zeros((1,1), dtype=torch.long)
with torch.no_grad():
    for _ in range(200):
        logits   = model(context[:, -64:])
        probs    = F.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, 1)
        context  = torch.cat([context, next_tok], dim=1)

print("\n── generated Persian text ──")
print(decode(context[0].tolist()))