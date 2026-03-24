# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# launch:
# $ py train.py config/train_rocstories.py

out_dir = 'out-rocstories'
# init_from = 'resume'  
always_save_checkpoint = False  # save every checkpoint for quick runs

wandb_log = True # set to True to log training and validation metrics to Weights and Biases (wandb)
wandb_project = 'rocstories'
wandb_run_name = 'rocstories-gpt'

dataset = 'rocstories'

# Main architectural parameters

batch_size = 8     # larger stable training, smaller for saving memory GPU
# Parameters that affect the model size
block_size = 512    # larger more understanding context, smaller faster runs
n_layer = 128         # larger for better reasoning(complexity understanding)
n_head = 64         # larger for more various attention patterns(better prespectives) 
n_embd = 128        # larger for more representation

# Fine-tuning parameters

eval_interval = 400                 # higher for faster runs, smaller for better monitoring
eval_iters = 200                    # higher for stable loss validation
log_interval = 50                   # higher more frequent logging
gradient_accumulation_steps = 8     # higher for stable training, smaller for faster runs

dropout = 0.1                     # larger for less overfitting, smaller for overfitting
learning_rate = 6e-4                # larger for faster training, smaller for better accuracy
max_iters = 6000                    # larger for longer training(might increase accuracy), smaller for faster runs
lr_decay_iters = 6000               # starting step for learning rate decay
min_lr = 6e-5                      # learning rate after decay
warmup_iters = 200                 # larger for more stable training - might spike at the start 
weight_decay = 0.1                 # larger for more regularization(less overfitting), smaller for less
beta1 = 0.9                         # larger for more stable training, smaller for faster convergence
beta2 = 0.95                        # larger for more stable training, smaller for faster convergence
grad_clip = 1                       # larger for more stable training(gradient might be exploding), smaller for faster convergence

bias = False        
device = 'cuda'
compile = False