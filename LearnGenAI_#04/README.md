# Persian TinyGPT: fluent output, and how much of it was copied

Companion code for **Learning GenAI by Building It, #04**.

A character-level GPT (208,512 parameters) trained from scratch on a single Persian children's story, then checked in two ways:

1. train vs validation loss during training
2. verbatim overlap between generated text and the training text

## What I found

The generated text read like the story, with some broken words. The loss curves and the overlap check say more than the text does.

**Loss (logged every 300 steps, single batch, so noisy):**

| step | train | val |
|-----:|------:|-----:|
| 0 | 3.9335 | 3.8017 |
| 300 | 2.1800 | 2.8420 |
| 600 | 1.9079 | 2.9841 |
| 900 | 1.0774 | 3.5473 |
| 1200 | 0.5869 | 4.4530 |
| 1500 | 0.2918 | 5.4208 |
| 1800 | 0.2291 | 5.4597 |
| 2100 | 0.1931 | 5.7158 |
| 2400 | 0.1763 | 6.3061 |
| 2700 | 0.1768 | 5.7433 |

For reference, random guessing over 42 characters is ln(42) = 3.74. The untrained model scored 3.93.

The lowest logged validation loss is 2.8420 at step 300. After that, train loss keeps falling while validation loss rises.

**Verbatim overlap** (a sliding window over the 200-character generated sample, counting windows that appear exactly in the reference text):

| window | generated in train | generated in val | val in train (baseline) |
|-------:|-------------------:|-----------------:|------------------------:|
| 10 chars | 122 / 191 (63.9%) | 3 / 191 (1.6%) | 15 / 447 (3.4%) |
| 20 chars | 65 / 181 (35.9%) | 0 / 181 (0.0%) | 0 / 437 (0.0%) |

The baseline column is the held-out text checked against the training text. It shows how much overlap the story produces by repeating itself, which is little.

**Reading:** about 4,100 training tokens for about 208,000 parameters is roughly 50 parameters per training token. The train/val gap and the overlap numbers both point to substantial copying from the training story. They do not prove that most of the output is copied.

## Caveats

- The overlap check uses one 200-character sample, so the percentages are rough.
- The sliding windows overlap, so one copied stretch counts several times.
- Losses are single-batch values, and the validation set is only 456 characters.
- One training run, no repeated seeds.

## Files

| file | what it is |
|---|---|
| `model.py` | TinyGPT: scaled dot-product attention, multi-head attention, feed-forward, transformer block |
| `train.py` | trains the model, prints loss every 300 steps, saves `persian_model.pt`, prints a generated sample |
| `overlap_check.py` | verbatim-overlap check between the generated text and the train/val split |
| `custom.txt` | training text |
| `generated.txt` | the 200-character sample used in the overlap check |
| `persian_model.pt` | weights after training |

## Setup

| item | value |
|---|---|
| training text | 4,554 characters, 42 unique characters |
| split | `n = int(0.9 * len(data))`: 4,098 train, 456 val |
| model | d_model 64, 4 heads, 4 layers, max_seq_len 64 |
| parameters | 208,512 |
| training | batch 8, block size 64, 3000 steps, lr 1e-3 |

Python 3.12 and PyTorch.

## Run

```
pip install -r requirements.txt
python train.py
```

Save the generated text printed at the end of training into `generated.txt` (UTF-8), then:

```
python overlap_check.py
```

`overlap_check.py` prints the file lengths first. They should read 4,554 total, 4,098 train and 456 val. If they do not, the script is not reading the same text and split as `train.py`.

---

*Learning GenAI by Building It, #04*
