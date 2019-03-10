import os
import random
import re

import pygame
import speech_recognition as sr
import torch
from torch.nn import Embedding, Module

from bhatterscot.config import CORPUS_NAME, encoder_n_layers, decoder_n_layers, hidden_size, \
    dropout, attn_model, device
from bhatterscot.model import EncoderRNN, LuongAttnDecoderRNN
from bhatterscot.text_data import normalize_string, MAX_LENGTH, indexes_from_sentence
from bhatterscot.vocabulary import SOS_token, Vocabulary

XPLORE_PROB = 0.1
SPEECH = False
LANGUAGE = 'en'


class ChatBot:
    def __init__(self, load_filename):
        self._vocabulary = Vocabulary(CORPUS_NAME)

        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(load_filename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        self._vocabulary.__dict__ = checkpoint['voc_dict']
        self._vocabulary.replace_word('s', 'is')

        print('Building _encoder and _decoder ...')
        # Initialize word embeddings
        embedding = Embedding(self._vocabulary.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)
        # Initialize _encoder & _decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, self._vocabulary.num_words, decoder_n_layers,
                                      dropout)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        self._encoder = encoder.to(device)
        self._decoder = decoder.to(device)
        print('Models built and ready to go!')

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        self._searcher = GreedySearchDecoder(encoder, decoder)

        pygame.mixer.init()
        if SPEECH:
            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone()

        corpus = os.path.join('data', CORPUS_NAME)
        self._new_conversation_file = os.path.join(corpus, 'new_conversations.txt')

    def chat(self, query):
        response = self._evaluate_input(query)
        return response

        # tts = gTTS(response, lang=LANGUAGE)
        # tts.save('bot.mp3')
        # playsound('bot.mp3')

        # with open(new_conversation_file, 'a') as f:
        #     f.write(f'{query}\n')
        #     f.write(f'{response}\n')

    def _evaluate(self, sentence, max_length=MAX_LENGTH):
        indexes_batch = [indexes_from_sentence(self._vocabulary, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        tokens, scores = self._searcher(input_batch, lengths, max_length)
        decoded_words = [self._vocabulary.index2word[token.item()] for token in tokens]
        return decoded_words

    def _evaluate_input(self, input_sentence):
        try:
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = self._evaluate(input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not x == 'PAD']
            output_words[:] = output_words[:output_words.index('EOS')]
            output_sentence = re.sub(r' ([.?!])', '\g<1>', ' '.join(output_words))
            output_sentence = re.sub(r'\.+', '.', output_sentence)
            output_sentence = output_sentence.replace('i m', "i am").replace('you re', "you are") \
                .replace('i ve', "i have").replace('can t', "cannot").replace('i ll', "i will") \
                .replace('it s', "it is").replace('don t', "do not").replace('couldn t', "could not") \
                .replace(' i ', ' I ').replace('we re', "we are").replace('we ll', "we will") \
                .replace('didn t', "did not")
            output_sentence = output_sentence.capitalize()
            for sep in ['.', '!', '?']:
                if sep in output_sentence:
                    output_sentence = f'{sep} '.join(x.strip()[0].upper() + x.strip()[1:]
                                                     for x in output_sentence.split(sep) if len(x) > 0)
            output_sentence += '.'

        except KeyError:
            output_sentence = 'Excuse me?'
        print(f'Bot: {output_sentence}')
        return output_sentence


class GreedySearchDecoder(Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through _encoder model
        encoder_outputs, encoder_hidden = self._encoder(input_seq, input_length)
        # Prepare _encoder's final hidden layer to be first hidden input to the _decoder
        decoder_hidden = encoder_hidden[:self._decoder.n_layers]
        # Initialize _decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through _decoder
            decoder_output, decoder_hidden = self._decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            if random.random() > 1 - XPLORE_PROB:
                decoder_input = torch.multinomial(decoder_output, 1)[0]
                decoder_scores = torch.Tensor(1)
            else:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next _decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
