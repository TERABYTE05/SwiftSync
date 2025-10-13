import os
import torch

# model to be fine tuned
model = "facebook/wav2vec2-xls-r-300m"
repo_name = "jager26/wav2vec2-xls-r-300m-common-voice-fr-ft"

# Training configuration
USE_FP16 = torch.cuda.is_available()

# initial hyperparameters
train_batch_size = 16
eval_batch_size = 8
gradient_accumulation_steps = 2
learning_rate = 3e-4
num_train_epochs = 2
warmup_steps = 500
save_steps = 500

