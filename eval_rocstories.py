"""
Evaluate ROCStories generations from prompts against reference stories.

This script loads a checkpoint (or GPT-2), generates continuations from
`eval_prompts.txt`, and reports simple lexical overlap metrics against
`eval_stories.txt`:
- unigram precision
- unigram recall
- unigram F1
- exact match (normalized text)

Defaults are intentionally lightweight for quick iteration.
"""

import os
import re
from collections import Counter
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# model/load config
init_from = 'resume'  # 'resume' or a GPT-2 variant like 'gpt2'
out_dir = 'out-rocstories'  # used when init_from == 'resume'
seed = 1337
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# generation config
prompts_file = 'data/rocstories/eval_prompts.txt'
stories_file = 'data/rocstories/eval_stories.txt'
max_examples = -1  # -1 means use all
gen_max_new_tokens = 80
temperature = 0.8
top_k = 200
print_first_n = 3

exec(open('configurator.py').read())  # overrides from CLI / config file
# -----------------------------------------------------------------------------


def read_prompts(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def read_stories(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # Stories are expected as paragraphs separated by blank lines.
    chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if c.strip()]
    # Collapse internal newlines/spaces for stable comparison.
    return [re.sub(r'\s+', ' ', c).strip() for c in chunks]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def word_tokens(text):
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def overlap_scores(pred_text, ref_text):
    pred = word_tokens(pred_text)
    ref = word_tokens(ref_text)

    if len(pred) == 0 and len(ref) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0

    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum(min(pred_counts[w], ref_counts[w]) for w in pred_counts)

    precision = overlap / max(1, len(pred))
    recall = overlap / max(1, len(ref))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    exact = 1.0 if normalize_text(pred_text) == normalize_text(ref_text) else 0.0
    return precision, recall, f1, exact


def get_reference_continuation(prompt, full_story):
    p = normalize_text(prompt)
    s = normalize_text(full_story)
    if s.startswith(p):
        cont = full_story[len(prompt):].strip()
        return cont if cont else full_story
    return full_story


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unsupported init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

enc = tiktoken.get_encoding('gpt2')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda t: enc.decode(t)

prompts = read_prompts(prompts_file)
stories = read_stories(stories_file)

n = min(len(prompts), len(stories))
if n == 0:
    raise ValueError('No prompts/stories found.')

if len(prompts) != len(stories):
    print(f"Warning: prompts={len(prompts)} stories={len(stories)}; using first {n} pairs")

pairs = list(zip(prompts[:n], stories[:n]))
if max_examples is not None and max_examples >= 0:
    pairs = pairs[:max_examples]

sum_p = 0.0
sum_r = 0.0
sum_f1 = 0.0
sum_em = 0.0

with torch.no_grad():
    with ctx:
        for i, (prompt, story) in enumerate(pairs):
            x_ids = encode(prompt)
            x = torch.tensor(x_ids, dtype=torch.long, device=device)[None, ...]

            y = model.generate(
                x,
                max_new_tokens=gen_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            new_ids = y[0].tolist()[len(x_ids):]
            pred_cont = decode(new_ids).strip()
            ref_cont = get_reference_continuation(prompt, story)

            p, r, f1, em = overlap_scores(pred_cont, ref_cont)
            sum_p += p
            sum_r += r
            sum_f1 += f1
            sum_em += em

            if i < max(0, int(print_first_n)):
                print(f"[{i}] prompt: {prompt}")
                print(f"[{i}] pred  : {pred_cont[:240]}")
                print(f"[{i}] ref   : {ref_cont[:240]}")
                print(f"[{i}] P/R/F1/EM: {p:.3f}/{r:.3f}/{f1:.3f}/{em:.3f}")
                print('-----')

count = len(pairs)
if count == 0:
    raise ValueError('No evaluation pairs after filtering.')

print('===== ROCStories Generation Eval =====')
print(f'model              : {init_from}')
print(f'out_dir            : {out_dir}')
print(f'pairs              : {count}')
print(f'gen_max_new_tokens : {gen_max_new_tokens}')
print(f'temperature         : {temperature}')
print(f'top_k              : {top_k}')
print(f'avg_precision      : {sum_p / count:.4f}')
print(f'avg_recall         : {sum_r / count:.4f}')
print(f'avg_f1             : {sum_f1 / count:.4f}')
print(f'exact_match_rate   : {sum_em / count:.4f}')
