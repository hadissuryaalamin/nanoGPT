# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# Based on 26_36.pt architecture (n_layer=24, n_head=16, n_embd=256)
# launch:
# $ py train.py config/train_rocstories_26_36.py

# =============================================================================
# Output & Checkpointing
# =============================================================================
out_dir = 'out-rocstories'
init_from = 'resume'                # 'scratch' or 'resume' from out_dir checkpoint
always_save_checkpoint = False       # save every checkpoint for quick runs

# =============================================================================
# Logging (Weights & Biases)
# =============================================================================
wandb_log = True
wandb_project = 'rocstories'
wandb_run_name = 'rocstories_no_split_each_batch_start_with_beginning'

dataset = 'rocstories'

# =============================================================================
# Architecture  (~19M params, under 32M limit)
# =============================================================================
block_size = 128    # larger = more context understanding, smaller = faster runs
n_layer = 6        # deeper model, better reasoning & complexity understanding
n_head = 6         # more attention patterns = better perspectives
n_embd = 384        # representation size (n_embd must be divisible by n_head)
bias = False

# =============================================================================
# Training Batch & Memory
# =============================================================================
batch_size = 16                     # effective batch = 64 * 4 = 256, stable training
gradient_accumulation_steps = 8     # effective batch = 64 * 4 = 256, stable training
dtype = 'bfloat16'
device = 'cuda'
compile = False

# =============================================================================
# Regularization
# =============================================================================
dropout = 0.2                       # from 26_36.pt — less dropout for deeper model
weight_decay = 0.2                  # regularization to reduce overfitting

# =============================================================================
# Learning Rate Schedule
# =============================================================================
learning_rate = 9e-6                # higher LR suits deeper model from scratch
min_lr = 9e-7                       # 10% of learning_rate
warmup_iters = 400                  # stable warmup before full LR
beta2 = 0.95                        # stable gradient accumulation

# =============================================================================
# Iteration & Evaluation
# =============================================================================
max_iters = 12000                    # total training iterations (adjust for quick runs)
lr_decay_iters = 12000               # decay LR across full training run
eval_interval = 100                 # evaluate every N iters
eval_iters = 400                    # iters to average for stable val loss estimate
log_interval = 50
grad_clip = 1.0                     # clip gradients to prevent exploding