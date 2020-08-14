import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from random import shuffle

from models import EncoderRNN, AttnDecoderRNN
from data_manager_sentences import DataManager
from utils import showPlot, asMinutes, timeSince

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path

MAX_LENGTH = 15
dm = DataManager(max_seq_len=MAX_LENGTH)
train_d, test_d, dev_d = dm.get_batched_data(batch_sz=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

poly_path = '/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2'
embedding_path = str(Path.home()) + poly_path
polyglot_emb = Embedding.load(embedding_path)
polyglot_emb.apply_expansion(CaseExpander)
emb_sz = polyglot_emb.shape[1]
voc_sz = len(dm.word2idx)  # the size of vocabulary

embeddings = torch.zeros(voc_sz, emb_sz)
for w, i in dm.word2idx.items():
    if w in polyglot_emb:
        embeddings[i] = torch.tensor(polyglot_emb[w])
    else:
        embeddings[i] = torch.randn(emb_sz, dtype=torch.float)

hid_sz = 256
encoder = EncoderRNN(voc_sz, hid_sz, emb_sz, embeddings=embeddings).to(device)
decoder = AttnDecoderRNN(voc_sz, hid_sz, emb_sz, embeddings=embeddings, max_length=MAX_LENGTH).to(device)

teacher_forcing_ratio = 0.5

def train(batch, encoder, decoder,
          optim_enc, optim_dec, criterion, max_length=MAX_LENGTH):
    encoder.train()
    decoder.train()
    optim_enc.zero_grad()
    optim_dec.zero_grad()

    input_length = batch.shape[0]
    target_length = batch.shape[0]
    batch_sz = batch.shape[1]

    encoder_hidden = encoder.init_hidden(batch_sz).to(device)
    encoder_outputs = torch.zeros(max_length, batch_sz, encoder.hid_sz, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            batch[ei,:], encoder_hidden)
        encoder_outputs[ei,:] += encoder_output[0,0]

    decoder_input = torch.full((1, batch_sz), dm.SOS_IDX, dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, batch[di])
            decoder_input = batch[di].view(1, -1)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.view(-1, batch_sz).detach()  # detach from history as input

            loss += criterion(decoder_output, batch[di])
            # if decoder_input.item() == dm.EOS_IDX:
            #     break

    loss.backward()

    optim_enc.step()
    optim_dec.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters,
               print_every=200, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optim_enc = optim.SGD(encoder.parameters(), lr=learning_rate)
    optim_dec = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    try:
        data = [random.choice(train_d) for _ in range(n_iters)]
        for epoch in range(1, n_iters + 1):

            sentence = data[epoch - 1].to(device)
            loss = train(sentence, encoder, decoder,
                         optim_enc, optim_dec, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_iters),
                                             epoch, epoch / n_iters * 100, print_loss_avg))
                evaluateRandomly(encoder, decoder, 1)

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    encoder.eval()
    encoder.eval()
    with torch.no_grad():
        input_tensor = dm.to_idxs(sentence).view(-1, 1).to(device)
        input_length = input_tensor.shape[0]
        encoder_hidden = encoder.init_hidden(1).to(device)

        encoder_outputs = torch.zeros(max_length, 1, encoder.hid_sz, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei,:] += encoder_output[0, 0]

        decoder_input = torch.tensor([[dm.SOS_IDX]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length, device=device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == dm.EOS_IDX:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dm.idx2word[topi.item()])

            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        ex = random.choice(test_d)
        ex = dm.to_sentence(ex, as_list=True)[0]
        pair = (ex, ex)
        print('>', ' '.join(ex))
        output_words, attentions = evaluate(encoder, decoder, ex)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# encoder.load_state_dict(torch.load("saved_models/EncoderRNN_batch64_20000_iters_part4_batch2_new_data.pt"))
# decoder.load_state_dict(torch.load("saved_models/AttnDecoder_batch64_20000_iters_part4_batch2_new_data.pt"))
encoder.load_state_dict(torch.load("saved_models/encoder_exp.pt"))
decoder.load_state_dict(torch.load("saved_models/decoder_exp.pt"))
trainIters(encoder, decoder, 20000)
torch.save(encoder.state_dict(), "saved_models/encoder_exp.pt")
torch.save(decoder.state_dict(), "saved_models/decoder_exp.pt")
evaluateRandomly(encoder, decoder)
