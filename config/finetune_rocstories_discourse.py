# =============================================================================
# Stage 2 Fine-Tuning: TinyStories FT model -> ROCStories Discourse SFT
# =============================================================================
# Goal    : teach the model to generate exactly 5-sentence stories with
#           explicit narrative structure using discourse tags
#
# Format  : <|s1|> setup <|s2|> conflict <|s3|> reaction <|s4|> attempt <|s5|> resolution
#
# Base    : out-tinystories-ft/ckpt_best.pt  (output of Stage 1)
#
# BEFORE RUNNING:
#   mkdir -p out-rocstories-discourse
#   cp out-tinystories-ft/ckpt_best.pt out-rocstories-discourse/ckpt.pt
#
# Then run:
#   python train.py config/finetune_rocstories_discourse.py
#
# INFERENCE after training:
#   Prompt the model with "<|s1|>" and it will generate a full 5-sentence story.
#   Example prompt: "<|s1|> John loved hiking."
# =============================================================================

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
out_dir   = 'out-rocstories-discourse'
init_from = 'resume'                # loads out-rocstories-discourse/ckpt.pt
always_save_checkpoint = True      # only save when val loss improves

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
wandb_log      = True
wandb_project  = 'rocstories-discourse'
wandb_run_name = 'stage2-discourse-sft'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset = 'rocstories_discourse'    # -> data/rocstories_discourse/train.bin & val.bin

# -----------------------------------------------------------------------------
# Architecture — loaded from checkpoint, listed here for reference only
# n_layer=6, n_head=6, n_embd=384, block_size=256, bias=False, vocab_size=50257
block_size = 256
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
batch_size                  = 32
gradient_accumulation_steps = 2     # effective batch = 64 seqs x 256 tokens = 16,384 tokens/iter
dtype                       = 'bfloat16'
device                      = 'cuda'
compile                     = False

# -----------------------------------------------------------------------------
# Regularization — low dropout for SFT, we want the model to reliably follow format
# -----------------------------------------------------------------------------
dropout      = 0.1
weight_decay = 0.1

# -----------------------------------------------------------------------------
# Learning Rate
# - Very low: preserve fluency learned in Stage 1, just teach discourse format
# - Constant LR (no decay): dataset is small, stable training is more important
# -----------------------------------------------------------------------------
learning_rate  = 3e-5
min_lr         = 3e-5               # same as LR — constant (decay_lr=False)
decay_lr       = False
warmup_iters   = 100                # short warmup
beta1          = 0.9
beta2          = 0.95
grad_clip      = 1.0

# -----------------------------------------------------------------------------
# Iterations
# ROCStories ~5M tokens / 16,384 tokens per iter ≈ 305 iters per epoch
# 5,000 iters ≈ ~16 epochs — enough for the model to learn the 5-tag format
#
# NOTE: iter_num is loaded from the Stage 1 checkpoint.
# Set max_iters = stage1_checkpoint_iter_num + 5000.
# Check with:
#   python -c "import torch; ck=torch.load('out-tinystories-ft/ckpt_best.pt'); print(ck['iter_num'])"
# -----------------------------------------------------------------------------
max_iters       = 52500             # adjust: stage1_iter_num + 5000
lr_decay_iters  = 52500
eval_interval   = 250
eval_iters      = 100
log_interval    = 25
