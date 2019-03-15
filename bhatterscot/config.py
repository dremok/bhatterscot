import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure models
MODEL_NAME = 'cb_model'
ATTN_MODEL = 'dot'
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64

# Configure training/optimization
CLIP = 50.0
TEACHER_FORCING_RATIO = 0.9
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_ITERATIONS = 5000
PRINT_EVERY = 10
SAVE_EVERY = 500

# Configure inference
CHECKPOINT_TO_LOAD = 5000

# General config
BASE_DIR = '/Users/max.leander/src/random/data'
MODEL_DIR = f'{BASE_DIR}/models'
CORPUS_NAME = 'video_games'
