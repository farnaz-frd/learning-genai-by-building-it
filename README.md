# Learning GenAI by Building It

A hands-on exploration of generative AI through implementation, experiments, debugging, and comparisons with established implementations.

The goal is to understand how these systems work by building their core components, testing assumptions, and documenting what the experiments actually demonstrate.

## Experiments

| # | Topic | Focus |
|---|---|---|
| 01 | Attention from scratch | Scaled dot-product attention and comparison with PyTorch |
| 02 | Multi-head attention | Inspecting attention weights and testing head ablations |
| 03 | A tiny GPT | Character-level language modeling and text generation |
| 04 | LoRA | Understanding parameter-efficient fine-tuning |
| 05 | Retrieval-Augmented Generation | Comparing retrieval methods and testing failure cases |

Experiments will be added as they are implemented and verified.

## Getting started

### Requirements

- Python 3.12 (or another version verified to work with the installed dependencies)
- PyTorch
- Matplotlib

### Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run an experiment

Follow the instructions in the relevant experiment directory. Each experiment should document its inputs, execution steps, observed results, and limitations.

## Principles

- Build to understand, not just to reproduce.
- Compare implementations where a meaningful reference exists.
- Test assumptions instead of relying on plausible outputs.
- Document mistakes and limitations alongside successful results.
- Report measured results without claiming more than the evidence supports.

## Follow the journey

This repository accompanies the **Learning GenAI by Building It** LinkedIn series.

New experiments, results, and lessons will be added as the work progresses.