# Config SFT: ROCStories Ending Prediction
# Fine-tune dari checkpoint pretrained kamu
#
# Jalankan:
#   python train.py config/train_sft.py
#
# Pastikan struktur folder:
#   out-rocstories/ckpt.pt          -> checkpoint pretrained kamu
#   data/rocstories_sft/train.bin   -> output dari prepare_sft.py
#   data/rocstories_sft/val.bin

# =============================================================================
# I/O
# =============================================================================
out_dir    = 'out-rocstories-sft'
init_from  = 'resume'               # load dari checkpoint pretrained
# PENTING: ganti path ini ke lokasi ckpt pretrained kamu
# train.py membaca dari out_dir saat init_from='resume',
# jadi copy dulu ckpt.pt ke out-rocstories-sft/ sebelum mulai,
# ATAU override di bawah dengan cara set out_dir ke folder checkpoint lama
# kemudian ganti out_dir setelah load (tidak didukung native nanoGPT).
#
# Cara paling simpel: 
#   cp out-rocstories/ckpt.pt out-rocstories-sft/ckpt.pt
# lalu jalankan script ini.

always_save_checkpoint = True

# =============================================================================
# Logging
# =============================================================================
wandb_log      = True
wandb_project  = 'rocstories-sft'
wandb_run_name = 'sft-ending-prediction'

# =============================================================================
# Dataset
# =============================================================================
dataset = 'rocstories_sft'          # → data/rocstories_sft/train.bin & val.bin

# =============================================================================
# Arsitektur — HARUS sama persis dengan checkpoint pretrained!
# =============================================================================
block_size = 256
n_layer    = 6
n_head     = 6
n_embd     = 384
bias       = False

# =============================================================================
# Training
# =============================================================================
batch_size                  = 32
gradient_accumulation_steps = 4    # effective batch = 128
dtype                       = 'bfloat16'   # sama dengan pretrain
device                      = 'cuda'
compile                     = False        # sama dengan pretrain

# =============================================================================
# Regularization
# =============================================================================
dropout      = 0.1          # turunkan dari pretrain (0.2) → lebih bebas generalize
weight_decay = 0.1

# =============================================================================
# Learning Rate
# Pretrain kamu pakai lr=3e-6 (sudah sangat kecil / late stage).
# SFT mulai dari sana, jadi kita pakai lr sedikit lebih tinggi
# agar model bisa adapt ke format baru, tapi tidak terlalu besar
# supaya tidak merusak representasi yang sudah terbentuk.
# =============================================================================
learning_rate  = 5e-5       # cukup untuk adapt format SFT
min_lr         = 5e-6       # 10% dari learning_rate
warmup_iters   = 40700        # warmup singkat
beta1          = 0.9
beta2          = 0.95       # sama dengan pretrain

# =============================================================================
# Iterasi — SFT jauh lebih pendek dari pretrain
# =============================================================================
max_iters       = 40600      # ~5 epoch untuk dataset ukuran ini
lr_decay_iters  = 40600
eval_interval   = 250
eval_iters      = 200
log_interval    = 50
grad_clip       = 1.0