# Fine-tune LLaMA on GPQA for QA
# launch: py train_llama.py config/finetune_gpqa_llama.py

out_dir = 'out-llama-gpqa'
eval_interval = 50
eval_iters = 20
log_interval = 5

always_save_checkpoint = False

wandb_log = False
wandb_project = 'llama-qa'
wandb_run_name = 'llama-gpqa-finetune'

dataset = 'gpqa'

# Resume from pretrained LLaMA checkpoint
init_from = 'resume'

# Smaller batches for fine-tuning (GPQA is small)
gradient_accumulation_steps = 4
batch_size = 4
block_size = 512

# Architecture must match pre-trained model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1  # add dropout for fine-tuning on small data

# Fine-tuning hyperparameters: lower LR, fewer iters
learning_rate = 1e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-5
warmup_iters = 100

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

device = 'cuda'
compile = False
