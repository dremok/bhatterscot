import os
import random
import re

import pygame
import speech_recognition as sr
import torch
from torch.nn import Embedding, Module

from bhatterscot.config import CORPUS_NAME, ENCODER_N_LAYERS, DECODER_N_LAYERS, HIDDEN_SIZE, DROPOUT, ATTN_MODEL, DEVICE
from bhatterscot.model import EncoderRNN, LuongAttnDecoderRNN
from bhatterscot.text_data import normalize_string, MAX_LENGTH, indexes_from_sentence
from bhatterscot.vocabulary import SOS_token, Vocabulary

EXPLORE_PROB = 0.1
SPEECH = False


class ChatBot:
    def __init__(self, model_file, use_device='cpu'):
        self._vocabulary = Vocabulary(CORPUS_NAME)

        if use_device == 'gpu':
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_file)

        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']
        self._vocabulary.__dict__ = checkpoint['voc_dict']
        self._vocabulary.replace_word('s', 'is')

        print('Building _encoder and _decoder ...')
        embedding = Embedding(self._vocabulary.num_words, HIDDEN_SIZE)
        embedding.load_state_dict(embedding_sd)
        encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT)
        decoder = LuongAttnDecoderRNN(ATTN_MODEL, embedding, HIDDEN_SIZE, self._vocabulary.num_words, DECODER_N_LAYERS,
                                      DROPOUT)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

        self._encoder = encoder.to(DEVICE)
        self._decoder = decoder.to(DEVICE)

        print('Models built and ready to go!')

        encoder.eval()
        decoder.eval()

        self._search_decoder = GreedySearchDecoder(encoder, decoder)

        pygame.mixer.init()
        if SPEECH:
            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone()

        corpus = os.path.join('data', CORPUS_NAME)
        self._conversation_logfile = os.path.join(corpus, 'conversations.txt')

    def chat(self, query):
        response = self._evaluate_input(query)
        return response

    def _evaluate_input(self, input_sentence):
        try:
            input_sentence = normalize_string(input_sentence)
            output_words = self._evaluate(input_sentence)
            output_sentence = self._sentence_from_words(output_words)
        except KeyError:
            output_sentence = 'Excuse me?'
        print(f'Bot: {output_sentence}')
        return output_sentence

    def _evaluate(self, sentence):
        indexes_batch = [indexes_from_sentence(self._vocabulary, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(DEVICE)
        lengths = lengths.to(DEVICE)
        tokens, scores = self._search_decoder(input_batch, lengths)
        decoded_words = [self._vocabulary.index2word[token.item()] for token in tokens]
        return decoded_words

    def _sentence_from_words(self, output_words):
        output_words[:] = [x for x in output_words if not x == 'PAD']
        output_words[:] = output_words[:output_words.index('EOS')]
        output_sentence = re.sub(r' ([.?!])', '\g<1>', ' '.join(output_words))
        output_sentence = re.sub(r'\.+', '.', output_sentence)
        output_sentence = output_sentence.replace('i m', "i am").replace('you re', "you are") \
            .replace('i ve', "i have").replace('can t', "cannot").replace('i ll', "i will") \
            .replace('it s', "it is").replace('don t', "do not").replace('couldn t', "could not") \
            .replace('we re', "we are").replace('we ll', "we will") \
            .replace('didn t', "did not").replace('they ll', "they will").replace('they re', 'they are')
        output_sentence = output_sentence.capitalize().replace(' i ', ' I ')
        for sep in ['.', '!', '?']:
            if sep in output_sentence:
                output_sentence = f'{sep} '.join(x.strip()[0].upper() + x.strip()[1:]
                                                 for x in output_sentence.split(sep) if len(x) > 0)
        output_sentence += '.'
        return output_sentence


class GreedySearchDecoder(Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, input_seq, input_length):
        encoder_outputs, encoder_hidden = self._encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self._decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)

        for _ in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self._decoder(decoder_input, decoder_hidden, encoder_outputs)
            if random.random() > 1 - EXPLORE_PROB:
                decoder_input = torch.multinomial(decoder_output, 1)[0]
                decoder_scores = torch.Tensor(1)
            else:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores
