# Pre-train LLaMA-style model on OpenWebText
# launch: py train_llama.py config/pretrain_owt_llama.py

out_dir = 'out-llama-pretrain'
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'llama-qa'
wandb_run_name = 'llama-pretrain'

dataset = 'openwebtext'

# effective tokens/iter = grad_accum * batch * block
# 8 * 8 * 512 = 32,768 tokens/iter on single GPU
gradient_accumulation_steps = 8
batch_size = 8
block_size = 512

# ~30M param LLaMA-style model (fits 8-16GB GPU)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-5
warmup_iters = 1000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

device = 'cuda'
compile = False
