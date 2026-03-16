# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# launch:
# $ py train.py config/train_rocstories.py

out_dir = 'out-rocstories'
eval_interval = 200
eval_iters = 40
log_interval = 10

# save only when validation improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'rocstories-gpt'

dataset = 'rocstories'

# effective tokens/iter = grad_accum * batch * block
# 8 * 8 * 512 = 32,768 tokens/iter on single GPU
gradient_accumulation_steps = 8
batch_size = 8
block_size = 512

# model size: balanced baseline for 8-16GB GPUs
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

learning_rate = 6e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 6e-5
warmup_iters = 1000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# windows/pytorch-friendly defaults
# set device='cpu' if no GPU
device = 'cuda'
compile = False
