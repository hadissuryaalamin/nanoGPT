# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# Based on 26_36.pt architecture (n_layer=24, n_head=16, n_embd=256)
# launch:
# $ py train.py config/train_rocstories_26_36.py

# =============================================================================
# Output & Checkpointing
# =============================================================================
out_dir = 'out-rocstories'
# init_from = 'resume'                # 'scratch' or 'resume' from out_dir checkpoint
always_save_checkpoint = True       # save every checkpoint for quick runs

# =============================================================================
# Logging (Weights & Biases)
# =============================================================================
wandb_log = True
wandb_project = 'rocstories'
wandb_run_name = 'rocstories'

dataset = 'rocstories'

# =============================================================================
# Architecture  (26_36.pt config — ~19M params, under 32M limit)
# =============================================================================
block_size = 256    # larger = more context understanding, smaller = faster runs
n_layer = 24        # deeper model, better reasoning & complexity understanding
n_head = 16         # more attention patterns = better perspectives
n_embd = 256        # representation size (n_embd must be divisible by n_head)
bias = False

# =============================================================================
# Training Batch & Memory
# =============================================================================
batch_size = 16                     # smaller to fit deeper model in GPU memory
gradient_accumulation_steps = 8     # effective batch = 16 * 8 = 128, stable training
dtype = 'float16'
device = 'cuda'
compile = True

# =============================================================================
# Regularization
# =============================================================================
dropout = 0.15                      # from 26_36.pt — less dropout for deeper model
weight_decay = 0.2                  # regularization to reduce overfitting

# =============================================================================
# Learning Rate Schedule
# =============================================================================
learning_rate = 6e-4                # higher LR suits deeper model from scratch
min_lr = 6e-5                       # 10% of learning_rate
warmup_iters = 500                  # stable warmup before full LR
beta2 = 0.99                        # stable gradient accumulation

# =============================================================================
# Iteration & Evaluation
# =============================================================================
max_iters = 30000
lr_decay_iters = 30000              # decay LR across full training run
eval_interval = 400                 # evaluate every N iters
eval_iters = 200                    # iters to average for stable val loss estimate
log_interval = 10
grad_clip = 1.0                     # clip gradients to prevent exploding