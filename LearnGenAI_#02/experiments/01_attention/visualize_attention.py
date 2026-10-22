"""Visualize last-token attention weights for a trained TinyGPT checkpoint.

Run after training:
    python experiments/03_tiny_gpt/visualize_attention.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genai_building_blocks.transformer import TinyGPT  # noqa: E402


CHECKPOINT_PATH = ROOT / "outputs" / "tiny_gpt.pt"
OUTPUT_PATH = ROOT / "outputs" / "attention_heads.png"
SAMPLE = "To be or not to be"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Run train_tiny_gpt.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False is needed here because the checkpoint includes a list of
    # vocabulary characters as well as tensors/config. Only load checkpoints you trust.
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}

    missing = [ch for ch in SAMPLE if ch not in stoi]
    if missing:
        raise ValueError(f"Sample contains characters absent from the vocabulary: {missing}")

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    token_ids = torch.tensor([[stoi[ch] for ch in SAMPLE]], device=device)
    with torch.no_grad():
        x = model.token_emb(token_ids)
        positions = torch.arange(token_ids.shape[1], device=device)
        x = x + model.pos_emb(positions)

        first_block = model.blocks[0]
        x = first_block.ln1(x)
        batch_size, seq_len, channels = x.shape
        attn = first_block.attn
        q = attn.Wq(x).view(batch_size, seq_len, attn.h, attn.d_k).transpose(1, 2)
        k = attn.Wk(x).view(batch_size, seq_len, attn.h, attn.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (attn.d_k ** 0.5)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)[0].cpu()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, attn.h, figsize=(4 * attn.h, 4), squeeze=False)
    for head in range(attn.h):
        ax = axes[0, head]
        ax.bar(range(seq_len), weights[head, -1].numpy())
        ax.set_title(f"Head {head}: last-token attention")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(list(SAMPLE), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Token")
        ax.set_ylabel("Attention weight")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"Saved attention visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
