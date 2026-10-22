# Learning GenAI by Building It

A hands-on exploration of generative AI through implementation, experiments,
debugging, and comparisons with established implementations.

The goal is to understand how these systems work by building their core
components, testing assumptions, and documenting what experiments demonstrate.

## Repository structure

- `src/genai_building_blocks/`: reusable attention, Transformer, and normalization code.
- `experiments/01_attention/`: tests comparing custom attention with PyTorch.
- `experiments/03_tiny_gpt/`: character-level training, generation, and attention visualization.

Experiments will be added as they are implemented and verified.

## Setup

Use Python 3.10+ (Python 3.12 is a reasonable starting point), then create a
virtual environment from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the attention tests

The tests use fixed seeds and check both causal and non-causal attention.

```bash
python -m pytest -q
```

On Windows, if pytest cannot find the `src` package, set `PYTHONPATH` for the
current PowerShell session before running tests:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## Train the tiny GPT

From the repository root:

```bash
python experiments/03_tiny_gpt/train_tiny_gpt.py
```

The script downloads the Tiny Shakespeare text on first run, trains a small
character-level GPT, prints a generated sample, and saves a checkpoint under
`outputs/`. Dataset source:
https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare

The dataset and checkpoint are not committed to Git.

## Visualize attention

After training:

```bash
python experiments/03_tiny_gpt/visualize_attention.py
```

The plot is saved to `outputs/attention_heads.png`. It visualizes the first
Transformer block's attention weights for the last character in the sample
`To be or not to be`. Attention weights are descriptive signals, not proof that
a head has a single fixed semantic role.

## Principles

- Build to understand, not just to reproduce.
- Compare implementations when a meaningful reference exists.
- Test assumptions instead of relying on plausible outputs.
- Document mistakes and limitations alongside successful results.
- Report measured results without claiming more than the evidence supports.

## Follow the journey

This repository accompanies the **Learning GenAI by Building It** LinkedIn series.
New experiments and findings will be added as they are implemented and tested.
