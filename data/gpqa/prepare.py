"""Prepare GPQA dataset for QA fine-tuning with nanoGPT LLaMA model.

Downloads GPQA from Hugging Face (idavidrein/gpqa), formats each example as:

    Question: <question>
    (A) <choice_a>
    (B) <choice_b>
    (C) <choice_c>
    (D) <choice_d>
    Answer: (<correct_letter>)<|endoftext|>

Tokenized with GPT-2 BPE via tiktoken. Saved as train.bin / val.bin.
"""

import os
import tiktoken
import numpy as np
from datasets import load_dataset


def format_qa(example):
    """Format a GPQA example into a QA prompt string."""
    q = example.get('Question', example.get('question', '')).strip()
    # GPQA has fields for the 4 choices and the correct answer
    choices = []
    for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
        val = example.get(key, '')
        if val:
            choices.append(val.strip())

    correct = example.get('Correct Answer', '').strip()

    # Shuffle choices deterministically based on question hash but keep track of correct
    import hashlib
    h = int(hashlib.sha256(q.encode()).hexdigest(), 16) % 24
    # Generate a permutation index from h
    indices = list(range(len(choices)))
    for i in range(len(indices) - 1, 0, -1):
        j = h % (i + 1)
        h //= (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    shuffled = [choices[i] for i in indices]

    # Find which letter the correct answer ended up at
    correct_idx = shuffled.index(correct) if correct in shuffled else 0
    letters = ['A', 'B', 'C', 'D']
    correct_letter = letters[correct_idx]

    text = f"Question: {q}\n"
    for i, choice in enumerate(shuffled):
        text += f"({letters[i]}) {choice}\n"
    text += f"Answer: ({correct_letter})"

    return text


enc = tiktoken.get_encoding("gpt2")

# GPQA dataset — try the "diamond" subset first (hardest), fall back to main
try:
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True)
    print("Loaded GPQA Diamond subset")
except Exception:
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", trust_remote_code=True)
        print("Loaded GPQA Main subset")
    except Exception:
        dataset = load_dataset("Idavidrein/gpqa", trust_remote_code=True)
        print("Loaded GPQA default")

# Figure out splits
if "train" in dataset:
    full_data = dataset["train"]
else:
    first_key = next(iter(dataset.keys()))
    full_data = dataset[first_key]

print(f"Total examples: {len(full_data)}")

# Train/val split (90/10)
split = full_data.train_test_split(test_size=0.1, seed=1337, shuffle=True)
train_split = split["train"]
val_split = split["test"]

print(f"Train: {len(train_split)}, Val: {len(val_split)}")

# Tokenize
train_ids = []
val_ids = []

for example in train_split:
    text = format_qa(example)
    if text:
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        train_ids.extend(ids)

for example in val_split:
    text = format_qa(example)
    if text:
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        val_ids.extend(ids)

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

base_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

print(f"train tokens: {len(train_ids)}")
print(f"val tokens: {len(val_ids)}")
print("GPQA data prepared successfully!")
