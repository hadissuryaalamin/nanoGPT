# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# launch:
# $ py train.py config/train_rocstories.py

out_dir = 'out-rocstories'
eval_interval = 100  # moderately frequent evals
eval_iters = 20      # moderate eval batches
log_interval = 10    # moderate logging

# save only when validation improves
always_save_checkpoint = True  # save every checkpoint for quick runs

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'rocstories-gpt'

dataset = 'rocstories'

# effective tokens/iter = grad_accum * batch * block
# 8 * 8 * 512 = 32,768 tokens/iter on single GPU
gradient_accumulation_steps = 4  # moderate
batch_size = 8                   # moderate batch
block_size = 512                 # moderate context



# medium model for not-too-quick training
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15
bias = False



learning_rate = 5e-4
max_iters = 1000   # more iterations
lr_decay_iters = 1000
min_lr = 5e-5
warmup_iters = 100



weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# windows/pytorch-friendly defaults
# set device='cpu' if no GPU
device = 'cuda'
compile = False
