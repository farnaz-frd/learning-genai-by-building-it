from pathlib import Path

STORY_PATH = "custom.txt"     
GEN_PATH = "generated.txt"    
SPLIT = 0.9                   

story = Path(STORY_PATH).read_text(encoding="utf-8")
gen = Path(GEN_PATH).read_text(encoding="utf-8")

n_train = int(len(story) * SPLIT)
train_text, val_text = story[:n_train], story[n_train:]

print(f"story chars: {len(story):,}  train: {len(train_text):,}  val: {len(val_text):,}  generated: {len(gen):,}")

def overlap(text, reference, k):
    windows = [text[i:i + k] for i in range(len(text) - k + 1)]
    hits = sum(1 for w in windows if w in reference)
    return hits, len(windows)

for k in (10, 20):
    print(f"\n--- substring length k = {k} ---")
    for name, text, ref_name, ref in [
        ("generated", gen, "train", train_text),
        ("generated", gen, "val", val_text),
        ("val (baseline)", val_text, "train", train_text),
    ]:
        h, n = overlap(text, ref, k)
        print(f"{name:15s} in {ref_name:5s}: {h:5d} / {n:5d} = {100 * h / max(n, 1):5.1f}%")