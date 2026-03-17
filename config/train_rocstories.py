# train a GPT on ROCStories token binaries (data/rocstories/train.bin, val.bin)
# launch:
# $ py train.py config/train_rocstories.py

out_dir = 'out-rocstories'
eval_interval = 200  # higher for faster runs, lower for better monitoring
eval_iters = 40      # higher for stable loss validation
log_interval = 20    # higher more frequent logging

# save only when validation improves
always_save_checkpoint = True  # save every checkpoint for quick runs

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'rocstories-gpt'

dataset = 'rocstories'

gradient_accumulation_steps = 8  # higher for stable training, lower for faster runs
batch_size = 16                  # larger stable training, lower for saving memory GPU
block_size = 1024                # larger more understanding context, lower faster runs

# Parameters that affect the model size
n_layer = 8     # larger for better reasoning(complexity understanding)
n_head = 8      # larger for more various attention patterns(better prespectives) 
n_embd = 320    # larger for more representation

dropout = 0.1 #larger for less overfitting, lower for overfitting

bias = False

learning_rate = 3e-5  # larger for faster training, lower for better accuracy
max_iters = 5000      # larger for longer training(might increase accuracy), lower for faster runs
lr_decay_iters = 4000 # Starting step for learning rate decay
min_lr = 4e-6         # learning rate after decay
warmup_iters = 500    # larger for more stable training - might spike at the start 

weight_decay = 0.1  # larger for more regularization(avoid overfitting), lower for less
beta1 = 0.9         # larger for more stable training, lower for faster convergence
beta2 = 0.95        # larger for more stable training, lower for faster convergence
grad_clip = 1.0     # larger for more stable training(gradient might be exploding), lower for faster convergence

device = 'cuda'
compile = False