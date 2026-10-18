# train_shakespeare.py
import torch
import torch.nn.functional as F
from model import TinyGPT

text  = open('shakespeare.txt').read()
chars = sorted(set(text))
assert len(chars) == 65, len(chars)
stoi  = {c: i for i, c in enumerate(chars)}
data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n     = int(0.9 * len(data))
train, val = data[:n], data[n:]
print(f"characters: {len(data):,}  train: {len(train):,}  val: {len(val):,}")

def get_batch(split, block_size=64, batch_size=16):
    d  = train if split == 'train' else val
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

torch.manual_seed(42)
model = TinyGPT(vocab_size=65, d_model=128, num_heads=4,
                num_layers=4, max_seq_len=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  (original run: 816,128)\n")

# losses you logged in the first run, for comparison
logged = {0: 4.3242, 200: 2.3835, 400: 2.1484, 600: 1.8644, 800: 1.9330,
          1000: 1.7529, 1200: 1.6737, 1400: 1.6152, 1600: 1.5575, 1800: 1.5310}

for step in range(2000):
    x, y   = get_batch('train')
    logits = model(x)
    loss   = F.cross_entropy(logits.view(-1, 65), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"step {step:4d}  new {loss.item():.4f}  original {logged[step]:.4f}")

torch.save({'state_dict': model.state_dict(), 'chars': chars}, 'shakespeare_model.pt')
print("\nsaved -> shakespeare_model.pt")
