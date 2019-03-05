from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import re
import unicodedata
from io import open

import torch

from bhatterscot.vocabulary import Vocabulary, EOS_token, PAD_token

MAX_LENGTH = 20


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?åäöÅÄÖ]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    s = re.sub(r'([(\[<]).*?([)\]>])', '\g<1>\g<2>', s)
    s = re.sub('[\[\]()<>]', '', s)
    return s


def read_vocs(datafile, corpus_name):
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[normalize_string(s)
              for s in l.split('\t')]
             for l in lines]
    voc = Vocabulary(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


# Using the functions defined above, return a populated vocabulary object and pairs list
def load_prepare_data(corpus_name, datafile):
    print('Start preparing text data ...')
    voc, pairs = read_vocs(datafile, corpus_name)
    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print('Counted words:', voc.num_words)
    return voc, pairs


def trim_rare_words(voc, pairs, min_count):
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep = True
        for word in input_sentence.split(' ') + output_sentence.split(' '):
            if word not in voc.word2index:
                keep = False
                break
        if keep:
            keep_pairs.append(pair)
    print(
        f'Trimmed {(len(pairs) - len(keep_pairs)) / len(pairs):.2%} % of pairs, from {len(pairs)} to {len(keep_pairs)}')
    return keep_pairs


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zero_padding(sentence_list, fillvalue=PAD_token):
    return list(itertools.zip_longest(*sentence_list, fillvalue=fillvalue))


def binary_matrix(batch, value=PAD_token):
    m = []
    for i, seq in enumerate(batch):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def input_var(sentence_list, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in sentence_list]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    inp = torch.LongTensor(pad_list)
    return inp, lengths


def output_var(sentence_list, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in sentence_list]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(mask)
    out = torch.LongTensor(pad_list)
    return out, mask, max_target_len


def batch2train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    out, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, out, mask, max_target_len
