"""Train a small character-level GPT on Tiny Shakespeare.

Run from the repository root:
    python experiments/03_tiny_gpt/train_tiny_gpt.py

The script downloads the Tiny Shakespeare text on first run and saves a local
checkpoint. Review the dataset source and terms before redistributing data.
"""

from pathlib import Path
import urllib.request

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_building_blocks.transformer import TinyGPT  # noqa: E402


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = ROOT / "data" / "shakespeare.txt"
CHECKPOINT_PATH = ROOT / "outputs" / "tiny_gpt.pt"


def load_text() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading dataset to {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Dataset appears empty or unexpectedly small.")
    return text


def main():
    torch.manual_seed(42)
    text = load_text()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda ids: "".join(itos[int(i)] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]

    # Keep this <= max_seq_len below.
    block_size = 64
    batch_size = 16
    d_model = 128
    num_heads = 4
    num_layers = 4
    max_seq_len = block_size
    train_steps = 2000
    eval_every = 200
    learning_rate = 1e-3

    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError("Dataset is too short for the configured block_size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(
        vocab_size=len(chars),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch(source):
        starts = torch.randint(len(source) - block_size, (batch_size,))
        x = torch.stack([source[i : i + block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    def estimate_loss():
        model.eval()
        results = {}
        with torch.no_grad():
            for name, source in (("train", train_data), ("val", val_data)):
                losses = []
                for _ in range(10):
                    x, y = get_batch(source)
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, len(chars)), y.reshape(-1)
                    )
                    losses.append(loss.item())
                results[name] = sum(losses) / len(losses)
        model.train()
        return results

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters")

    model.train()
    for step in range(train_steps):
        x, y = get_batch(train_data)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == train_steps - 1:
            losses = estimate_loss()
            print(
                f"step {step:4d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f}"
            )

    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=500, temperature=0.8)
    print("\n--- Generated sample ---\n")
    print(decode(generated[0].cpu().tolist()))

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": chars,
            "config": {
                "vocab_size": len(chars),
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "max_seq_len": max_seq_len,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"\nCheckpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
