"""Prepare ROCStories with discourse tags for 5-sentence story SFT (Stage 2).

Each story is formatted with explicit narrative role tags:

    <|s1|> {setup}    <|s2|> {conflict}  <|s3|> {reaction}
    <|s4|> {attempt}  <|s5|> {resolution} <|endoftext|>

The model learns:
  - <|s1|>  orientation / setup
  - <|s2|>  inciting event / conflict
  - <|s3|>  character reaction / feeling
  - <|s4|>  action / attempt
  - <|s5|>  resolution / ending

At inference, prompt with "<|s1|>" and the model generates a complete
5-sentence story, stopping at <|endoftext|>.

Output:
    data/rocstories_discourse/train.bin
    data/rocstories_discourse/val.bin
    data/rocstories_discourse/meta.pkl
"""

import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256

S1 = "<|s1|> "
S2 = " <|s2|> "
S3 = " <|s3|> "
S4 = " <|s4|> "
S5 = " <|s5|> "


def build_sample(example):
    """Build one discourse-tagged sample from a ROCStories example."""
    # Prefer individual sentence columns
    if example.get("sentence1"):
        s1 = str(example.get("sentence1", "")).strip()
        s2 = str(example.get("sentence2", "")).strip()
        s3 = str(example.get("sentence3", "")).strip()
        s4 = str(example.get("sentence4", "")).strip()
        s5 = str(example.get("sentence5", "")).strip()
    elif example.get("story"):
        # Try splitting the story field by sentence
        import re
        parts = re.split(r'(?<=[.!?])\s+', str(example["story"]).strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 5:
            return None
        s1, s2, s3, s4 = parts[0], parts[1], parts[2], parts[3]
        s5 = " ".join(parts[4:])
    elif example.get("text"):
        import re
        parts = re.split(r'(?<=[.!?])\s+', str(example["text"]).strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 5:
            return None
        s1, s2, s3, s4 = parts[0], parts[1], parts[2], parts[3]
        s5 = " ".join(parts[4:])
    else:
        return None

    if not all([s1, s2, s3, s4, s5]):
        return None

    text = f"{S1}{s1}{S2}{s2}{S3}{s3}{S4}{s4}{S5}{s5}"
    return enc.encode_ordinary(text) + [EOT]


def tokenize_split(split):
    ids, skipped = [], 0
    for example in split:
        sample = build_sample(example)
        if sample is None:
            skipped += 1
            continue
        ids.extend(sample)
    if skipped:
        print(f"  (skipped {skipped} incomplete examples)")
    return np.array(ids, dtype=np.uint16)


print("Downloading mintujupally/ROCStories ...")
dataset = load_dataset("mintujupally/ROCStories")

sample = dataset["train"][0]
print(f"Columns : {list(sample.keys())}")
print(f"Sample  : {sample}")

train_val   = dataset["train"].train_test_split(test_size=0.1, seed=1337, shuffle=True)
train_split = train_val["train"]
val_split   = train_val["test"]

print(f"\n  train : {len(train_split):,} stories")
print(f"  val   : {len(val_split):,} stories")

# Print an example formatted sample
ex_sample = build_sample(train_split[0])
ex_text = enc.decode(ex_sample[:-1])  # exclude EOT for display
print(f"\nExample formatted sample:\n  {ex_text}\n")

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

for name, split in {"train": train_split, "val": val_split}.items():
    print(f"Tokenising {name} ...")
    ids = tokenize_split(split)
    path = os.path.join(out_dir, f"{name}.bin")
    ids.tofile(path)
    print(f"  -> {path}  ({len(ids):,} tokens, {ids.nbytes / 1e6:.1f} MB)")

meta = {
    "vocab_size": enc.n_vocab,
    "tokenizer": "gpt2",
    "discourse_tags": {
        "s1": S1.strip(), "s2": S2.strip(),
        "s3": S3.strip(), "s4": S4.strip(), "s5": S5.strip(),
    },
}
meta_path = os.path.join(out_dir, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print(f"\nMeta saved -> {meta_path}")
print("Done! Next: python train.py config/finetune_rocstories_discourse.py")
