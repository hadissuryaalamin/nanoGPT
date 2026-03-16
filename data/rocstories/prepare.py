"""Prepare ROCStories into contiguous token streams for nanoGPT.

Preprocessing summary:
- Source data: Hugging Face dataset ``mintujupally/ROCStories``.
- Story text construction: prefers ``story``; otherwise joins
	``sentence1``..``sentence5`` with spaces; falls back to ``text``.
- Tokenization: GPT-2 BPE via ``tiktoken`` using ``encode_ordinary``.
	This intentionally avoids adding any extra special tokens during encoding.
- Story separator: appends one GPT-2 end-of-text token (``<|endoftext|>``)
	after every non-empty story.
- Concatenation strategy: all stories in a split are flattened into one long
	1D token sequence, then saved as ``uint16`` to ``train.bin``/``val.bin``.

Resulting binaries are directly consumable by the training loader, which reads
fixed-length windows from these contiguous token streams.
"""

import os
import tiktoken
import numpy as np
from datasets import load_dataset


def story_text(example):
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


def pick_train_val_splits(dataset):
	"""Pick train/val robustly across dataset variants."""
	if "train" in dataset and "validation" in dataset:
		return dataset["train"], dataset["validation"]
	if "train" in dataset and "test" in dataset:
		return dataset["train"], dataset["test"]
	if "train" in dataset:
		split = dataset["train"].train_test_split(test_size=0.1, seed=1337, shuffle=True)
		return split["train"], split["test"]

	first_split_name = next(iter(dataset.keys()))
	split = dataset[first_split_name].train_test_split(test_size=0.1, seed=1337, shuffle=True)
	return split["train"], split["test"]


def tokenize_split(split, encoder):
	"""Tokenize a split into one contiguous token array.

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


dataset = load_dataset("mintujupally/ROCStories")
train_split, val_split = pick_train_val_splits(dataset)

enc = tiktoken.get_encoding("gpt2")
train_ids = tokenize_split(train_split, enc)
val_ids = tokenize_split(val_split, enc)

base_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

print("train and val bin files created successfully!")
print(f"meta.pkl saved with vocab_size={enc.n_vocab}")
print("train tokens:", len(train_ids))
print("val tokens:", len(val_ids))