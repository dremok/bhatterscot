from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import os
import random
import re

import torch
import torch.nn as nn
from playsound import playsound

from bhatterscot.config import save_dir, model_name, corpus_name, encoder_n_layers, decoder_n_layers, hidden_size, \
    dropout, attn_model, device
from bhatterscot.model import EncoderRNN, LuongAttnDecoderRNN
from bhatterscot.text_data import normalize_string, MAX_LENGTH, indexes_from_sentence
from bhatterscot.vocabulary import SOS_token, Vocabulary

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

vocabulary = Vocabulary(corpus_name)

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 100
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))

checkpoint = None
# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    vocabulary.__dict__ = checkpoint['voc_dict']
    # vocabulary.replace_word('s', 'is')

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(vocabulary.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocabulary.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

XPLORE_PROB = 0.2


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            if random.random() > 1 - XPLORE_PROB:
                decoder_input = torch.multinomial(decoder_output, 1)[0]
                decoder_scores = torch.Tensor(1)
            else:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(searcher, voc: Vocabulary, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, input_sentence):
    try:
        # Normalize sentence
        input_sentence = normalize_string(input_sentence)
        # Evaluate sentence
        output_words = evaluate(searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        output_sentence = re.sub(r' ([.?!])', '\g<1>', ' '.join(output_words))
        output_sentence = re.sub(r'\.+', '.', output_sentence)
        output_sentence = output_sentence.replace('i m', "i'm").replace('can t', "can't").replace('you re', "you're") \
            .replace('i ve', "i've").replace('can t', "can't").replace('i ll', "i'll").replace('it s', "it's") \
            .replace('don t', "don't")
        for sep in ['.', '!', '?']:
            output_sentence = f'{sep} '.join(x.strip()[0].upper() + x.strip()[1:]
                                             for x in output_sentence.split(sep) if len(x) > 0)
        print('Bot:', f'{output_sentence}.')

    except KeyError:
        output_sentence = 'Förlåt.'
        print(f'Bot: {output_sentence}')
    return output_sentence


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

import pygame
from gtts import gTTS
import speech_recognition as sr

SPEECH = False

if SPEECH:
    pygame.mixer.init()
    r = sr.Recognizer()
    mic = sr.Microphone()
    tts = gTTS('hej.', lang='sv')
    tts.save('bot.mp3')
    playsound('bot.mp3')

corpus = os.path.join('data', corpus_name)
new_conversation_file = os.path.join(corpus, 'new_conversations.txt')

while True:
    # query = input('Prompt: ')
    # if query == 'q':
    #     break

    if SPEECH:
        with mic as source:
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language='sv-SE')
        except:
            query = 'vad sa du?'
    else:
        query = input('Prompt:')
    print(query)

    # Begin chatting (uncomment and run the following line to begin)
    response = evaluateInput(searcher, vocabulary, query)

    if SPEECH:
        tts = gTTS(response, lang='sv')
        tts.save('bot.mp3')

        playsound('bot.mp3')

    with open(new_conversation_file, 'a') as f:
        f.write(f'{query}\n')
        f.write(f'{response}\n')
