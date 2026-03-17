# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# launch:
# $ py train.py config/train_rocstories.py

out_dir = 'out-rocstories'
eval_interval = 20  # more frequent evals
eval_iters = 5      # fewer eval batches
log_interval = 2    # more frequent logs

# save only when validation improves
always_save_checkpoint = True  # save every checkpoint for quick runs

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'rocstories-gpt'

dataset = 'rocstories'

# effective tokens/iter = grad_accum * batch * block
# 8 * 8 * 512 = 32,768 tokens/iter on single GPU
gradient_accumulation_steps = 2  # smaller for quick runs
batch_size = 4                   # smaller batch
block_size = 256                 # smaller context


# quick training: much smaller model
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.2
bias = False


learning_rate = 1e-3
max_iters = 100   # much fewer iters
lr_decay_iters = 100
min_lr = 1e-4
warmup_iters = 10


weight_decay = 0.01
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# windows/pytorch-friendly defaults
# set device='cpu' if no GPU
device = 'cuda'
compile = False
