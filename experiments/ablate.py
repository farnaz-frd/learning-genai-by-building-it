# ablate.py
import torch, torch.nn.functional as F
from model import TinyGPT

ckpt = torch.load('shakespeare_model.pt')
chars = ckpt['chars']
stoi = {c: i for i, c in enumerate(chars)}
model = TinyGPT(vocab_size=65, d_model=128, num_heads=4, num_layers=4, max_seq_len=64)
model.load_state_dict(ckpt['state_dict'])
model.eval()

text = open('shakespeare.txt').read()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
val = data[int(0.9 * len(data)):]

T, N, BATCH = 64, 512, 64
step = (len(val) - T - 1) // N
starts = [i * step for i in range(N)]
X = torch.stack([val[s:s + T] for s in starts])
Y = torch.stack([val[s + 1:s + T + 1] for s in starts])

def zero_heads(heads, d_k):
    def hook(module, args):
        x = args[0].clone()
        for h in heads:
            x[..., h * d_k:(h + 1) * d_k] = 0
        return (x,)
    return hook

def per_window_loss(layer=None, heads=()):
    handle = None
    if layer is not None:
        a = model.blocks[layer].attn
        handle = a.Wo.register_forward_pre_hook(zero_heads(heads, a.d_k))
    out = []
    with torch.no_grad():
        for b in range(0, N, BATCH):
            logits = model(X[b:b + BATCH])
            l = F.cross_entropy(logits.reshape(-1, 65), Y[b:b + BATCH].reshape(-1),
                                reduction='none').view(-1, T).mean(1)
            out.append(l)
    if handle:
        handle.remove()
    return torch.cat(out)

base = per_window_loss()
print(f"{N} validation windows of {T} chars")
print(f"baseline val loss: {base.mean().item():.4f}\n")
print("ablation                 val loss   delta    SE    delta/SE")

def report(name, layer, heads):
    l = per_window_loss(layer, heads)
    d = l - base
    se = d.std().item() / (N ** 0.5)
    print(f"{name:24s} {l.mean().item():.4f}  {d.mean().item():+.4f}  {se:.4f}  {d.mean().item() / se:6.1f}")

for L in range(4):
    for hd in range(4):
        report(f"layer {L} head {hd}", L, [hd])
    print()
report("layer 0 heads 1+3", 0, [1, 3])
report("layer 0 heads 0+1+3", 0, [0, 1, 3])
report("layer 0 all four heads", 0, [0, 1, 2, 3])