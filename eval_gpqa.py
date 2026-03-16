"""
Evaluate LLaMA QA model on GPQA-style multiple choice questions.

Loads a fine-tuned checkpoint and evaluates accuracy on QA prompts
by comparing model predictions against correct answers.

Usage:
    py eval_gpqa.py
    py eval_gpqa.py --out_dir=out-llama-gpqa --max_examples=50
"""

import os
import re
import hashlib
from contextlib import nullcontext

import torch
import tiktoken

from model_llama import LLaMAConfig, LLaMA

# -----------------------------------------------------------------------------
# config
out_dir = 'out-llama-gpqa'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
seed = 1337
max_examples = -1  # -1 means all
gen_max_new_tokens = 10
temperature = 0.1  # low temp for deterministic answer selection
top_k = 5
print_first_n = 5

exec(open('configurator.py').read())
# -----------------------------------------------------------------------------


def format_qa_prompt(example):
    """Format a GPQA example into prompt (without answer) and correct letter."""
    q = example.get('Question', example.get('question', '')).strip()
    choices = []
    for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
        val = example.get(key, '')
        if val:
            choices.append(val.strip())

    correct = example.get('Correct Answer', '').strip()

    # Same deterministic shuffle as in prepare.py
    h = int(hashlib.sha256(q.encode()).hexdigest(), 16) % 24
    indices = list(range(len(choices)))
    for i in range(len(indices) - 1, 0, -1):
        j = h % (i + 1)
        h //= (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    shuffled = [choices[i] for i in indices]

    correct_idx = shuffled.index(correct) if correct in shuffled else 0
    letters = ['A', 'B', 'C', 'D']
    correct_letter = letters[correct_idx]

    prompt = f"Question: {q}\n"
    for i, choice in enumerate(shuffled):
        prompt += f"({letters[i]}) {choice}\n"
    prompt += "Answer: ("

    return prompt, correct_letter


def extract_answer(text):
    """Extract the answer letter from model output."""
    match = re.search(r'([A-D])', text)
    return match.group(1) if match else None


# Setup
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
llamaconf = LLaMAConfig(**checkpoint['model_args'])
model = LLaMA(llamaconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

enc = tiktoken.get_encoding('gpt2')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda t: enc.decode(t)

# Load GPQA dataset for evaluation
from datasets import load_dataset
try:
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True)
except Exception:
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", trust_remote_code=True)
    except Exception:
        dataset = load_dataset("Idavidrein/gpqa", trust_remote_code=True)

if "train" in dataset:
    eval_data = dataset["train"]
else:
    first_key = next(iter(dataset.keys()))
    eval_data = dataset[first_key]

# Use a held-out portion for eval (last 10%)
split = eval_data.train_test_split(test_size=0.1, seed=1337, shuffle=True)
eval_examples = split["test"]

if max_examples > 0:
    eval_examples = eval_examples.select(range(min(max_examples, len(eval_examples))))

print(f"Evaluating on {len(eval_examples)} examples...")

correct = 0
total = 0
results = []

with torch.no_grad():
    with ctx:
        for i, example in enumerate(eval_examples):
            prompt, correct_letter = format_qa_prompt(example)
            x_ids = encode(prompt)

            # Truncate if too long
            if len(x_ids) > llamaconf.block_size - gen_max_new_tokens:
                x_ids = x_ids[-(llamaconf.block_size - gen_max_new_tokens):]

            x = torch.tensor(x_ids, dtype=torch.long, device=device)[None, ...]
            y = model.generate(x, max_new_tokens=gen_max_new_tokens, temperature=temperature, top_k=top_k)

            new_ids = y[0].tolist()[len(x_ids):]
            pred_text = decode(new_ids).strip()
            pred_letter = extract_answer(pred_text)

            is_correct = pred_letter == correct_letter
            if is_correct:
                correct += 1
            total += 1

            results.append({
                'pred': pred_letter,
                'correct': correct_letter,
                'is_correct': is_correct,
            })

            if i < print_first_n:
                print(f"[{i}] Predicted: ({pred_letter}) | Correct: ({correct_letter}) | {'✓' if is_correct else '✗'}")
                print(f"     Raw output: {pred_text[:80]}")
                print('-----')

accuracy = correct / total if total > 0 else 0
random_baseline = 0.25  # 4-choice QA

print()
print('===== GPQA Evaluation Results =====')
print(f'model            : LLaMA from {out_dir}')
print(f'examples         : {total}')
print(f'correct          : {correct}')
print(f'accuracy         : {accuracy:.4f} ({accuracy*100:.1f}%)')
print(f'random baseline  : {random_baseline:.4f} ({random_baseline*100:.1f}%)')
print(f'above random     : {accuracy - random_baseline:+.4f}')
