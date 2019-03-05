import codecs
import csv
import os
from io import open

import torch
import torch.nn as nn

from bhatterscot.config import device, hidden_size, encoder_n_layers, dropout, attn_model, decoder_n_layers, \
    learning_rate, decoder_learning_ratio, model_name, n_iteration, batch_size, print_every, save_every, clip, \
    CORPUS_NAME, save_dir
from bhatterscot.custom_data import load_sentence_pairs
from bhatterscot.model import EncoderRNN, LuongAttnDecoderRNN, trainIters
from bhatterscot.text_data import load_prepare_data, trim_rare_words
from bhatterscot.vocabulary import Vocabulary

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

corpus = os.path.join('data', CORPUS_NAME)
voc = Vocabulary(CORPUS_NAME)

# Set checkpoint to load from; set to None if starting from scratch
# load_filename = os.path.join(save_dir, model_name, CORPUS_NAME,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))
load_filename = None

checkpoint = None
# Load model if a load_filename is provided
if load_filename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(load_filename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

sentence_pairs = []
datafile = os.path.join(corpus, 'formatted_sentence_pairs.txt')
if os.path.isfile(datafile):
    with open(datafile, 'r', encoding='utf-8') as inputfile:
        reader = csv.reader(inputfile, delimiter=delimiter)
        for row in reader:
            sentence_pairs.append(row)
else:
    sentence_pairs = load_sentence_pairs(CORPUS_NAME)
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in sentence_pairs:
            writer.writerow(pair)

# Load/Assemble vocabulary and pairs
voc, pairs = load_prepare_data(CORPUS_NAME, datafile)
pairs = trim_rare_words(voc, pairs, 1)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if load_filename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if load_filename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if load_filename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, CORPUS_NAME, load_filename, checkpoint)
