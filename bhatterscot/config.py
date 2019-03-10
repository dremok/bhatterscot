import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.9
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
print_every = 1
save_every = 500
checkpoint_to_load = 5000

save_dir = '/Users/max.leander/src/random/data/save'

CORPUS_NAME = 'vg'
