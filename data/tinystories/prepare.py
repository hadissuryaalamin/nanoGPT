"""Prepare TinyStories for nanoGPT fine-tuning (Stage 1).

Source  : roneneldan/TinyStories  (~2.1M GPT-4-generated stories, ~475M tokens)
Format  : plain story text + <|endoftext|> separator  (no special tags)
Purpose : improve fluency and coherence before discourse SFT on ROCStories

Output:
    data/tinystories/train.bin
    data/tinystories/val.bin
    data/tinystories/meta.pkl
"""

import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256


def tokenize_split(split):
    ids = []
    skipped = 0
    for example in split:
        text = example.get("text", "").strip()
        if not text:
            skipped += 1
            continue
        ids.extend(enc.encode_ordinary(text))
        ids.append(EOT)
    if skipped:
        print(f"  (skipped {skipped} empty examples)")
    return np.array(ids, dtype=np.uint16)


print("Downloading roneneldan/TinyStories ...")
dataset = load_dataset("roneneldan/TinyStories")

print(f"  train      : {len(dataset['train']):,} stories")
print(f"  validation : {len(dataset['validation']):,} stories")

out_dir = os.path.dirname(os.path.abspath(__file__))

for name, split in {"train": dataset["train"], "val": dataset["validation"]}.items():
    print(f"\nTokenising {name} ...")
    ids = tokenize_split(split)
    path = os.path.join(out_dir, f"{name}.bin")
    ids.tofile(path)
    print(f"  -> {path}  ({len(ids):,} tokens, {ids.nbytes / 1e6:.1f} MB)")

meta = {"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}
meta_path = os.path.join(out_dir, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print(f"\nMeta saved -> {meta_path}")
print("Done! Next: python train.py config/finetune_tinystories.py")
