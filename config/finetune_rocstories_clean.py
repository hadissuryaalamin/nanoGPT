# =============================================================================
# Stage 3 Fine-Tuning: ROCStories Discourse model -> Clean Story Generation
# =============================================================================
# Goal    : adapt the discourse model to generate stories WITHOUT discourse tags
#           so inference works with standard tiktoken (no <|s1|> special tokens)
#
# Base    : out-rocstories-discourse/ckpt.pt
#
# BEFORE RUNNING:
#   mkdir -p out-rocstories-clean
#   cp out-rocstories-discourse/ckpt.pt out-rocstories-clean/ckpt.pt
#
#   Check the discourse checkpoint iter_num:
#   python -c "import torch; ck=torch.load('out-rocstories-discourse/ckpt.pt'); print(ck['iter_num'])"
#   Then set max_iters = that_number + 1500
#
# Then run:
#   python train.py config/finetune_rocstories_clean.py
#
# INFERENCE after training:
#   python sample.py --out_dir=out-rocstories-clean --start="\n"
#   python eval.py --init_from=resume --out_dir=out-rocstories-clean --input_file=data/rocstories/eval_stories.txt
# =============================================================================

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
out_dir   = 'out-rocstories-clean'
init_from = 'resume'
always_save_checkpoint = True
eval_input_file = 'data/rocstories/test.txt'

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
wandb_log      = True
wandb_project  = 'rocstories-clean'
wandb_run_name = 'stage3-clean-sft'

# -----------------------------------------------------------------------------
# Dataset  — clean rocstories, no discourse tags
# -----------------------------------------------------------------------------
dataset = 'rocstories'

# -----------------------------------------------------------------------------
# Architecture — loaded from checkpoint (n_layer=6, n_head=6, n_embd=384,
#                block_size=256, bias=False, vocab_size=50257)
# -----------------------------------------------------------------------------
block_size = 256

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
batch_size                  = 32
gradient_accumulation_steps = 2     # effective batch = 64 seqs x 256 tokens = 16,384 tokens/iter
dtype                       = 'bfloat16'
device                      = 'cuda'
compile                     = False

# -----------------------------------------------------------------------------
# Regularization
# -----------------------------------------------------------------------------
dropout      = 0.1
weight_decay = 0.1

# -----------------------------------------------------------------------------
# Learning Rate  — keep very low: we only want to unteach the tag format,
#                  not disturb the fluency learned in earlier stages
# -----------------------------------------------------------------------------
learning_rate  = 1e-5
min_lr         = 1e-5
decay_lr       = False
warmup_iters   = 50
beta1          = 0.9
beta2          = 0.95
grad_clip      = 1.0

# -----------------------------------------------------------------------------
# Iterations
# rocstories ~5M tokens / 16,384 tokens per iter ≈ 305 iters per epoch
# +1500 iters ≈ ~5 epochs — enough to unteach the tag format
#
# Set max_iters = discourse_checkpoint_iter_num + 1500  (see instructions above)
# -----------------------------------------------------------------------------
max_iters       = 60000             # adjust: discourse_iter_num + 1500
lr_decay_iters  = 60000
eval_interval   = 200
eval_iters      = 100
log_interval    = 25
