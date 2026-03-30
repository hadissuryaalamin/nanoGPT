# =============================================================================
# Stage 1 Fine-Tuning: ROCStories pretrained model -> TinyStories
# =============================================================================
# Goal    : improve fluency and coherence on high-quality GPT-4 story data
# Base    : out-rocstories/25_62.pt  (n_layer=6, n_head=6, n_embd=384, block_size=256)
#
# BEFORE RUNNING:
#   mkdir -p out-tinystories-ft
#   cp out-rocstories/25_62.pt out-tinystories-ft/ckpt.pt
#
# Then run:
#   python train.py config/finetune_tinystories.py
# =============================================================================

import torch

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
out_dir   = 'out-tinystories-ft'
init_from = 'resume'                # loads out-tinystories-ft/ckpt.pt (= 25_62.pt)
always_save_checkpoint = False      # only save when val loss improves

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
wandb_log      = True
wandb_project  = 'tinystories-ft'
wandb_run_name = 'stage1-tinystories-ft'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset = 'tinystories'             # -> data/tinystories/train.bin & val.bin

# -----------------------------------------------------------------------------
# Architecture — loaded from checkpoint, listed here for reference only
# n_layer=6, n_head=6, n_embd=384, bias=False, vocab_size=50257
block_size = 256  # must match checkpoint — also controls data loader sequence length
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
batch_size                  = 32
gradient_accumulation_steps = 8     # effective batch = 128 seqs x 256 tokens = 32,768 tokens/iter
dtype                       = 'float16'  # use float16 for faster training and lower memory usage
device                      = 'cuda'
compile                     = False

# -----------------------------------------------------------------------------
# Regularization — reduce dropout vs pretrain (0.2) to allow more generalization
# -----------------------------------------------------------------------------
dropout      = 0.1
weight_decay = 0.1

# -----------------------------------------------------------------------------
# Learning Rate
# - Higher than late-stage pretraining LR (9e-6) to adapt to new domain
# - With warmup to avoid destabilizing pretrained weights early on
# -----------------------------------------------------------------------------
learning_rate  = 1e-4
min_lr         = 1e-5               # 10% of learning_rate
decay_lr       = True
warmup_iters   = 200
beta1          = 0.9
beta2          = 0.95
grad_clip      = 1.0

# -----------------------------------------------------------------------------
# Iterations
# TinyStories ~475M tokens / 32,768 tokens per iter ≈ 14,495 iters per epoch
# 20,000 iters ≈ ~1.4 epochs — enough to adapt without overfitting
#
# NOTE: iter_num is loaded from checkpoint. If 25_62.pt was saved at iter X,
# training continues from iter X. Set max_iters = X + 20000 to get 20k new steps.
# Check iter_num by running:
#   python -c "import torch; ck=torch.load('out-rocstories/25_62.pt'); print(ck['iter_num'])"
# -----------------------------------------------------------------------------
max_iters       = 47600             # adjust: checkpoint_iter_num + 20000
lr_decay_iters  = 47600
eval_interval   = 500
eval_iters      = 200
log_interval    = 50
