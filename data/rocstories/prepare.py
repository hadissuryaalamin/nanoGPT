"""Prepare ROCStories into contiguous token streams for nanoGPT.

Preprocessing summary:
- Source data: Hugging Face dataset ``mintujupally/ROCStories``.

Split strategy:
    ROCStories original splits
    ├── train (98k stories)
    │   ├── train.bin  ← 90%  fed to the model during training (88k stories)
    │   └── val.bin    ← 10%  monitors loss / detects overfitting (10k stories)
    └── test  (3.7k stories)
        └── test.bin   ← final held-out benchmark; touch only after training

- Story text construction: prefers ``story``; otherwise joins
    ``sentence1``..``sentence5`` with spaces; falls back to ``text``.
- Tokenization: GPT-2 BPE via ``tiktoken`` using ``encode_ordinary``.
    This intentionally avoids adding any extra special tokens during encoding.
- Custom special tokens:
    * ``<|startoftext|>`` (token ID = 50257) prepended before every story.
    * ``<|endoftext|>``   (token ID = 50256) appended  after  every story.
  These two tokens bracket each story in the flattened token stream so the
  model can learn both where a story begins and where it ends.
- Concatenation strategy: all stories in a split are flattened into one long
    1D token sequence, then saved as ``uint16``.

Custom token note:
    GPT-2's tiktoken vocabulary ends at ID 50256 (``<|endoftext|>``).
    ``<|startoftext|>`` is assigned the next available ID **50257**.
    vocab_size in meta.pkl is therefore bumped to 50258 so that nanoGPT
    allocates an embedding row for both special tokens.
"""

import os
import tiktoken
import numpy as np
from datasets import load_dataset
import pickle


# ---------------------------------------------------------------------------
# Custom special-token IDs
# ---------------------------------------------------------------------------

# GPT-2 native EOT token (ID 50256) — already part of tiktoken's gpt2 vocab
EOT_TOKEN_ID = 50256   # <|endoftext|>

# We assign the very next ID as our custom start-of-story marker.
# tiktoken's gpt2 encoding has no token at 50257, so we claim it ourselves.
SOT_TOKEN_ID = 50257   # <|startoftext|>  (custom)

# nanoGPT needs to know the true vocab size so its embedding table covers
# both special tokens (50256 and 50257).
CUSTOM_VOCAB_SIZE = 50258


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def story_text(example: dict) -> str:
    """Build a single story string from common ROCStories field layouts.

    Order of precedence:
    1. ``story``
    2. ``sentence1``..``sentence5`` joined with spaces
    3. ``text``
    """
    if "story" in example and example["story"]:
        return str(example["story"]).strip()

    sentence_keys = [
        key
        for key in ("sentence1", "sentence2", "sentence3", "sentence4", "sentence5")
        if key in example and example[key]
    ]
    if sentence_keys:
        return " ".join(str(example[key]).strip() for key in sentence_keys).strip()

    if "text" in example and example["text"]:
        return str(example["text"]).strip()

    return ""


def tokenize_split(split, encoder) -> np.ndarray:
    """Tokenize a dataset split into one contiguous uint16 token array.

    Each non-empty story is wrapped with custom special tokens:
        [SOT] <story tokens…> [EOT]

    ``encode_ordinary`` is used so tiktoken does NOT inject any implicit
    special tokens during encoding — only our explicit IDs are inserted.
    """
    ids = []
    for example in split:
        text = story_text(example)
        if not text:
            continue
        ids.append(SOT_TOKEN_ID)                   # <|startoftext|>
        ids.extend(encoder.encode_ordinary(text))  # story body
        ids.append(EOT_TOKEN_ID)                   # <|endoftext|>
    return np.array(ids, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Download dataset
# ---------------------------------------------------------------------------

print("Downloading ROCStories from Hugging Face …")
dataset = load_dataset("mintujupally/ROCStories")

# ---------------------------------------------------------------------------
# Build the three splits
# ---------------------------------------------------------------------------

# 1. Final held-out test set — never touch during training
test_split = dataset["test"]
print(f"  original test  : {len(test_split):>7,} stories")

# 2. Split the original train set → train (90 %) + val (10 %)
train_val   = dataset["train"].train_test_split(test_size=0.1, seed=1337, shuffle=True)
train_split = train_val["train"]
val_split   = train_val["test"]   # "test" key here is just HF naming; this IS our val set
print(f"  training split : {len(train_split):>7,} stories  (90 % of original train)")
print(f"  validation split:{len(val_split):>7,} stories  (10 % of original train)")

# ---------------------------------------------------------------------------
# Tokenise & save .bin files
# ---------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")
base_dir = os.path.dirname(os.path.abspath(__file__))

for name, split in {"train": train_split, "val": val_split}.items():
    print(f"\nTokenising {name} split …")
    ids = tokenize_split(split, enc)
    out_path = os.path.join(base_dir, f"{name}.bin")
    ids.tofile(out_path)
    print(f"  → {out_path}  ({len(ids):,} tokens, {ids.nbytes / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# Save meta.pkl with tokenizer info
# ---------------------------------------------------------------------------

meta = {
    # Bumped to 50258 to cover both <|endoftext|> (50256) and
    # the custom <|startoftext|> (50257).
    "vocab_size": CUSTOM_VOCAB_SIZE,
    "tokenizer": "gpt2",
    "special_tokens": {
        "<|endoftext|>":   EOT_TOKEN_ID,
        "<|startoftext|>": SOT_TOKEN_ID,
    },
}
meta_path = os.path.join(base_dir, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"\nMeta saved → {meta_path}  (vocab_size={CUSTOM_VOCAB_SIZE})")
print(f"  Special tokens: <|startoftext|>={SOT_TOKEN_ID}, <|endoftext|>={EOT_TOKEN_ID}")

# ---------------------------------------------------------------------------
# Save test split as plain text (one story per line)
# ---------------------------------------------------------------------------

print("\nSaving test split as plain text …")
test_path = os.path.join(base_dir, "test.txt")
with open(test_path, "w", encoding="utf-8") as f:
    for example in test_split:
        text = story_text(example)
        if text:
            f.write(text + "\n\n")
print(f"  → {test_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\nDone!")
print("Files created:")
for name in ("train", "val"):
    print(f"  {os.path.join(base_dir, f'{name}.bin')}")
print(f"  {os.path.join(base_dir, 'test.txt')}")
print(f"  {os.path.join(base_dir, 'meta.pkl')}")
print()
print("Token stream layout per story:")
print("  [50257: <|startoftext|>]  <story tokens…>  [50256: <|endoftext|>]")
print()
print("Usage reminder:")
print("  train.bin + val.bin  →  use during model training")
print("  test.txt             →  plain text stories for final evaluation after training")
print("  meta.pkl             →  vocab_size (50258) + special token IDs, read by nanoGPT at train time")