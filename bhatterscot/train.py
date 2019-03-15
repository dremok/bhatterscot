import os
import random

import torch
import torch.nn as nn

from bhatterscot.config import DEVICE, HIDDEN_SIZE, ENCODER_N_LAYERS, DROPOUT, ATTN_MODEL, DECODER_N_LAYERS, \
    LEARNING_RATE, DECODER_LEARNING_RATIO, MODEL_NAME, N_ITERATIONS, BATCH_SIZE, PRINT_EVERY, SAVE_EVERY, CLIP, \
    CORPUS_NAME, MODEL_DIR, TEACHER_FORCING_RATIO
from bhatterscot.model import EncoderRNN, LuongAttnDecoderRNN, maskNLLLoss
from bhatterscot.text_data import load_prepare_data, trim_rare_words, load_sentence_pairs, batch2train_data
from bhatterscot.vocabulary import Vocabulary, SOS_token


def main():
    vocabulary = Vocabulary(CORPUS_NAME)
    sentence_pairs = load_sentence_pairs(CORPUS_NAME)
    pairs = load_prepare_data(vocabulary, sentence_pairs)
    pairs = trim_rare_words(vocabulary, pairs, 2)

    print(' Some sentence pairs:')
    for pair in pairs[:10]:
        print(pair)

    print('Building _encoder and _decoder ...')
    embedding = nn.Embedding(vocabulary.num_words, HIDDEN_SIZE)
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT)
    decoder = LuongAttnDecoderRNN(ATTN_MODEL, embedding, HIDDEN_SIZE, vocabulary.num_words, DECODER_N_LAYERS, DROPOUT)

    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    print('Models built and ready to go!')

    encoder.train()
    decoder.train()

    print('Building optimizers ...')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    print("Starting Training!")
    run_training(vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding)


def run_training(vocabulary, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding):
    training_batches = [batch2train_data(vocabulary, [random.choice(pairs) for _ in range(BATCH_SIZE)])
                        for _ in range(N_ITERATIONS)]

    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    print("Training...")
    for iteration in range(start_iteration, N_ITERATIONS + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, target_variable, mask, lengths, max_target_len, encoder, decoder,
                     encoder_optimizer, decoder_optimizer)
        print_loss += loss

        if iteration % PRINT_EVERY == 0:
            print_loss_avg = print_loss / PRINT_EVERY
            print(
                f'Iteration: {iteration}; '
                f'Percent complete: {iteration / N_ITERATIONS * 100:.1%}; '
                f'Average loss: {print_loss_avg:.4f}')
            print_loss = 0

        if iteration % SAVE_EVERY == 0:
            directory = os.path.join(MODEL_DIR, MODEL_NAME, CORPUS_NAME,
                                     f'{ENCODER_N_LAYERS}-{DECODER_N_LAYERS}_{HIDDEN_SIZE}')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocabulary.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def train(input_variable, target_variable, mask, lengths, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(DEVICE)
    lengths = lengths.to(DEVICE)
    target_variable = target_variable.to(DEVICE)
    mask = mask.to(DEVICE)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(BATCH_SIZE)]])
    decoder_input = decoder_input.to(DEVICE)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        if use_teacher_forcing:
            decoder_input = target_variable[t].view(1, -1)
        else:
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(BATCH_SIZE)]])
            decoder_input = decoder_input.to(DEVICE)
        mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * n_total)
        n_totals += n_total

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


if __name__ == '__main__':
    main()
