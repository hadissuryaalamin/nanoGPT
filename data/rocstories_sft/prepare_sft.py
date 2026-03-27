"""Prepare ROCStories for SFT ending-prediction fine-tuning.

Format setiap story:
    <|story|> sentence1 sentence2 sentence3 sentence4 <|ending|> sentence5 <|endoftext|>

Output:
    data/rocstories_sft/train.bin
    data/rocstories_sft/val.bin
    data/rocstories_sft/meta.pkl
"""

import os
import re
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

STORY_SEP  = "<|story|>"
ENDING_SEP = "<|ending|>"

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256


def encode(text):
    return enc.encode_ordinary(text)


def split_sentences(text):
    """Pecah story string menjadi list kalimat berdasarkan tanda baca."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_sample(example):
    """Bangun satu sample SFT: [4 kalimat konteks] -> [kalimat 5 / ending].

    Support dua format:
    1. Kolom sentence1..sentence5
    2. Kolom text berisi satu string 5 kalimat
    """
    # Format 1: kolom terpisah sentence1..sentence5
    if example.get("sentence1"):
        s1 = str(example.get("sentence1", "")).strip()
        s2 = str(example.get("sentence2", "")).strip()
        s3 = str(example.get("sentence3", "")).strip()
        s4 = str(example.get("sentence4", "")).strip()
        s5 = str(example.get("sentence5", "")).strip()

    # Format 2: satu kolom 'text' (mintujupally/ROCStories)
    elif example.get("text"):
        sentences = split_sentences(str(example["text"]))
        if len(sentences) < 5:
            return None
        s1, s2, s3, s4 = sentences[0], sentences[1], sentences[2], sentences[3]
        s5 = " ".join(sentences[4:])

    else:
        return None

    if not all([s1, s2, s3, s4, s5]):
        return None

    context  = f"{STORY_SEP} {s1} {s2} {s3} {s4} {ENDING_SEP} "
    response = s5
    return encode(context) + encode(response) + [EOT]


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Downloading ROCStories ...")
dataset = load_dataset("mintujupally/ROCStories")

sample = dataset["train"][0]
print(f"Kolom dataset : {list(sample.keys())}")
print(f"Contoh data   : {sample}")

train_val   = dataset["train"].train_test_split(test_size=0.2, seed=1337, shuffle=True)
train_split = train_val["train"]
val_split   = train_val["test"]

print(f"\n  train : {len(train_split):,} stories")
print(f"  val   : {len(val_split):,} stories")

out_dir = os.path.join("data", "rocstories_sft")
os.makedirs(out_dir, exist_ok=True)

for name, split in {"train": train_split, "val": val_split}.items():
    print(f"\nTokenising {name} ...")
    ids = tokenize_split(split)
    path = os.path.join(out_dir, f"{name}.bin")
    ids.tofile(path)
    print(f"  -> {path}  ({len(ids):,} tokens, {ids.nbytes/1e6:.1f} MB)")

meta = {
    "vocab_size": enc.n_vocab,
    "tokenizer": "gpt2",
    "story_sep": STORY_SEP,
    "ending_sep": ENDING_SEP,
}
meta_path = os.path.join(out_dir, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print(f"\nMeta saved -> {meta_path}")
print("\nDone! Sekarang jalankan: python train.py config/train_rocstories_sft.py")