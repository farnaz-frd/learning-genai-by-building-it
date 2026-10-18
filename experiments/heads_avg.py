# heads_avg.py
import math, torch
from model import TinyGPT

# ---- load model and rebuild the same validation split ----
ckpt = torch.load('shakespeare_model.pt')
chars = ckpt['chars']
stoi = {c: i for i, c in enumerate(chars)}
model = TinyGPT(vocab_size=65, d_model=128, num_heads=4, num_layers=4, max_seq_len=64)
model.load_state_dict(ckpt['state_dict'])
model.eval()

text = open('shakespeare.txt').read()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
val = data[int(0.9 * len(data)):]

def attn_weights(block, x):
    """Attention weights (B, h, T, T) of one block, given the block's input x."""
    a = block.attn
    h = block.ln1(x)
    B, T, C = h.shape
    Q = a.Wq(h).view(B, T, a.h, a.d_k).transpose(1, 2)
    K = a.Wk(h).view(B, T, a.h, a.d_k).transpose(1, 2)
    s = Q @ K.transpose(-2, -1) / math.sqrt(a.d_k)
    mask = torch.tril(torch.ones(T, T))
    s = s.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(s, dim=-1)

# ---- Part A: the sentence from your snapshot, block 0, last token ----
with torch.no_grad():
    sample = "To be or not to be"
    idx = torch.tensor([[stoi[c] for c in sample]])
    x = model.token_emb(idx) + model.pos_emb(torch.arange(idx.shape[1]))
    w = attn_weights(model.blocks[0], x)
    print("Part A: block 0, last token 'e', weight on the previous 'b' (position 16)")
    for hd in range(4):
        row = w[0, hd, -1]
        print(f"  head {hd}: on 'b' = {row[16].item():.3f}   min {row.min().item():.3f}   max {row.max().item():.3f}")
    print("  snapshot values: head0 0.587, head1 0.925, head3 0.967, head2 flat (0.032 to 0.091)\n")

# ---- Part B: average over many validation windows ----
T, N, BATCH, T0 = 64, 256, 64, 16     # window length, windows, batch size, first query position used
step = (len(val) - T - 1) // N
starts = [i * step for i in range(N)]
offsets = list(range(0, 17))          # 0 = the token itself, 1 = previous char, ...
prof = torch.zeros(4, 4, 17)          # layer, head, offset
tv = torch.zeros(4, 4, 4)             # layer, head a, head b: total variation distance
count = 0

with torch.no_grad():
    for b in range(0, N, BATCH):
        ids = torch.stack([val[s:s + T] for s in starts[b:b + BATCH]])
        n = ids.shape[0]
        x = model.token_emb(ids) + model.pos_emb(torch.arange(T))
        for L, block in enumerate(model.blocks):
            w = attn_weights(block, x)                      # (n, 4, T, T)
            for k in offsets:
                d = w.diagonal(offset=-k, dim1=-2, dim2=-1)  # attention of position t to t-k
                d = d[..., max(T0 - k, 0):]                  # keep only queries t >= T0
                prof[L, :, k] += d.mean(dim=(0, 2)) * n
            for a in range(4):
                for c in range(4):
                    diff = 0.5 * (w[:, a, T0:] - w[:, c, T0:]).abs().sum(-1)
                    tv[L, a, c] += diff.mean() * n
            x = block(x)
        count += n

prof /= count
tv /= count
print(f"Part B: {count} validation windows of {T} chars, queries at positions >= {T0}")
print("Average attention mass by distance back (each row sums to 1)")
print("layer head |  self   -1    -2    -3    -4  | -5..-8 -9..-16  >16")
for L in range(4):
    for hd in range(4):
        p = prof[L, hd]
        mid1 = p[5:9].sum().item()
        mid2 = p[9:17].sum().item()
        rest = 1 - p.sum().item()
        print(f"  {L}    {hd}   | {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f} |  {mid1:.3f}  {mid2:.3f}  {rest:.3f}")
    print()

print("Total variation distance between heads (0 = identical attention, 1 = completely different)")
for L in range(4):
    pairs = [(a, c) for a in range(4) for c in range(a + 1, 4)]
    print(f"  layer {L}: " + "  ".join(f"{a}-{c}: {tv[L, a, c]:.3f}" for a, c in pairs))