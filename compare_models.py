"""
Run sample_batch.py for every .pt checkpoint in a directory and save
each model's outputs to a separate JSONL file for comparison.

Usage:
    python compare_models.py
    python compare_models.py --out_dir=out --prompts_file=data/rocstories/eval_prompts.txt
    python compare_models.py --num_samples=3 --max_new_tokens=256

How it works:
    For each *.pt file found in `out_dir`, this script:
      1. Backs up out_dir/ckpt.pt if it exists
      2. Copies the target checkpoint to out_dir/ckpt.pt
      3. Runs sample_batch.py --init_from=resume --out_dir=<out_dir>
      4. Saves generated samples to compare_outputs/<model_stem>.jsonl
      5. Restores the original ckpt.pt

Output layout:
    compare_outputs/
        PT_ROCStories.jsonl
        FT_Tinystories.jsonl
        FT_Discourse.jsonl
        Final.jsonl
        summary.txt          <- all prompts + all model responses side by side
"""

import os
import glob
import json
import shutil
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Config — all overridable via --key=value CLI args
# ---------------------------------------------------------------------------
out_dir        = 'out'
prompts_file   = 'data/rocstories/eval_prompts.txt'
num_samples    = 1
max_new_tokens = 200
device         = 'cuda'
compare_dir    = 'compare_outputs'

# Parse CLI overrides (same pattern as configurator.py)
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key, _, val = arg[2:].partition('=')
        if key in ('num_samples', 'max_new_tokens'):
            globals()[key] = int(val)
        else:
            globals()[key] = val

# ---------------------------------------------------------------------------
# Discover checkpoints — skip ckpt.pt (that's the swap slot sample_batch.py loads)
# ---------------------------------------------------------------------------
SWAP_FILENAME = 'ckpt.pt'

pt_files = sorted([
    f for f in glob.glob(os.path.join(out_dir, '*.pt'))
    if os.path.basename(f) != SWAP_FILENAME
])

if not pt_files:
    print(f"No .pt checkpoints found in '{out_dir}' (excluding {SWAP_FILENAME}). Exiting.")
    sys.exit(1)

print(f"Found {len(pt_files)} checkpoint(s) to compare:")
for p in pt_files:
    print(f"  {p}")
print()

os.makedirs(compare_dir, exist_ok=True)

ckpt_slot   = os.path.join(out_dir, SWAP_FILENAME)
ckpt_backup = os.path.join(out_dir, '_swap_backup.pt')

# Back up existing ckpt.pt once
backed_up = False
if os.path.exists(ckpt_slot):
    shutil.copy2(ckpt_slot, ckpt_backup)
    backed_up = True
    print(f"Backed up existing {SWAP_FILENAME} -> {ckpt_backup}")

# ---------------------------------------------------------------------------
# Run sample_batch.py for each checkpoint
# ---------------------------------------------------------------------------
results = {}   # model_name -> list of {prompt, generated_text}

for pt_path in pt_files:
    model_name   = os.path.splitext(os.path.basename(pt_path))[0]
    output_jsonl = os.path.join(compare_dir, f"{model_name}.jsonl")

    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  Ckpt  : {pt_path}")
    print(f"  Out   : {output_jsonl}")
    print(f"{'='*60}")

    # Swap checkpoint into slot
    shutil.copy2(pt_path, ckpt_slot)

    cmd = [
        sys.executable, 'sample_batch.py',
        '--init_from=resume',
        f'--out_dir={out_dir}',
        f'--start=FILE:{prompts_file}',
        '--batch_prompts=True',
        f'--num_samples={num_samples}',
        f'--max_new_tokens={max_new_tokens}',
        f'--output_file={output_jsonl}',
        f'--device={device}',
    ]

    proc = subprocess.run(cmd, text=True)

    if proc.returncode != 0:
        print(f"  [ERROR] sample_batch.py exited with code {proc.returncode}")
        results[model_name] = []
        continue

    # Load the JSONL output for the summary
    records = []
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    results[model_name] = records
    print(f"  Saved {len(records)} record(s) to {output_jsonl}")

# Restore original ckpt.pt
if backed_up:
    shutil.copy2(ckpt_backup, ckpt_slot)
    os.remove(ckpt_backup)
    print(f"\nRestored original {SWAP_FILENAME}")
elif os.path.exists(ckpt_slot):
    os.remove(ckpt_slot)

# ---------------------------------------------------------------------------
# Write side-by-side summary.txt
# ---------------------------------------------------------------------------
summary_path = os.path.join(compare_dir, 'summary.txt')
model_names  = [os.path.splitext(os.path.basename(p))[0] for p in pt_files]

# Collect all unique prompts in order
all_prompts = []
seen = set()
for name in model_names:
    for rec in results.get(name, []):
        p = rec.get('prompt', '').strip()
        if p and p not in seen:
            all_prompts.append(p)
            seen.add(p)

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("MODEL COMPARISON SUMMARY\n")
    f.write(f"Models  : {', '.join(model_names)}\n")
    f.write(f"Prompts : {prompts_file}\n")
    f.write(f"Samples : {num_samples} per prompt per model\n")
    f.write(f"MaxToks : {max_new_tokens}\n")
    f.write("=" * 80 + "\n\n")

    for prompt in all_prompts:
        f.write(f"PROMPT: {prompt}\n")
        f.write("-" * 80 + "\n")

        for name in model_names:
            # Find all records for this prompt from this model
            recs = [r for r in results.get(name, []) if r.get('prompt', '').strip() == prompt]
            f.write(f"[{name}]\n")
            if not recs:
                f.write("  (no output)\n")
            else:
                for i, rec in enumerate(recs, 1):
                    text = rec.get('generated_text', '').strip()
                    # Word-wrap at 76 chars for readability
                    wrapped = textwrap.fill(text, width=76, initial_indent='  ', subsequent_indent='  ')
                    if len(recs) > 1:
                        f.write(f"  -- sample {i} --\n")
                    f.write(wrapped + "\n")
            f.write("\n")

        f.write("=" * 80 + "\n\n")

print(f"\nSide-by-side summary written to: {summary_path}")
print("\nDone.")
