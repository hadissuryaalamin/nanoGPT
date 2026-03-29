"""Prepare ROCStories into contiguous token streams for nanoGPT.

Preprocessing summary:
- Source data: Hugging Face dataset ``mintujupally/ROCStories``.

Split strategy:
    ROCStories original splits
    ├── train (98k stories) → train.bin  fed to the model during training
    └── test  (3.7k stories) → val.bin   monitors loss / detects overfitting

- Story text construction: prefers ``story``; otherwise joins
    ``sentence1``..``sentence5`` with spaces; falls back to ``text``.
- Tokenization: GPT-2 BPE via ``tiktoken`` using ``encode_ordinary``.
    This intentionally avoids adding any extra special tokens during encoding.
- Story separator: appends one GPT-2 end-of-text token (``<|endoftext|>``)
    after every non-empty story.
- Concatenation strategy: all stories in a split are flattened into one long
    1D token sequence, then saved as ``uint16``.
"""

import os
import tiktoken
import numpy as np
from datasets import load_dataset
import pickle


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

    Each non-empty story is tokenized with ``encode_ordinary`` (no implicit
    special-token injection), followed by one ``encoder.eot_token`` to delimit
    story boundaries in the flattened stream.
    """
    ids = []
    for example in split:
        text = story_text(example)
        if not text:
            continue
        ids.extend(encoder.encode_ordinary(text))
        ids.append(encoder.eot_token)
    return np.array(ids, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Download dataset
# ---------------------------------------------------------------------------

print("Downloading ROCStories from Hugging Face …")
dataset = load_dataset("mintujupally/ROCStories")

# ---------------------------------------------------------------------------
# Build the three splits
# ---------------------------------------------------------------------------

train_split = dataset["train"]
val_split   = dataset["test"]
print(f"  training split   : {len(train_split):>7,} stories")
print(f"  validation split : {len(val_split):>7,} stories")

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

# Save meta.pkl with tokenizer info
meta = {
    "vocab_size": enc.n_vocab,
    "tokenizer": "gpt2",
}
meta_path = os.path.join(base_dir, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"\nMeta saved → {meta_path}  (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\nDone!")
print("Files created:")
for name in ("train", "val"):
    print(f"  {os.path.join(base_dir, f'{name}.bin')}")
print(f"  {os.path.join(base_dir, 'meta.pkl')}")
print()
print("Usage reminder:")
print("  train.bin  →  full original train split, used during model training")
print("  val.bin    →  full original test split, monitors loss")
print("  meta.pkl   →  vocab_size + tokenizer info, read by nanoGPT at train time")