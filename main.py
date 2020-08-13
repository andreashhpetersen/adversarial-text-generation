import os
import time
import math
import torch
import torch.nn as nn
from random import shuffle
import random
import matplotlib.pyplot as plt
import json

from models import TransformerModel, Generator, Discriminator
from reader_sentences import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path

def run(padding_eos):
    dm = DataManager(normalize_to_max_seq_len_and_eos=padding_eos)
    train_d, test_d, dev_d = dm.get_batched_data(batch_sz=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(random.choice(dm.train_data))

    poly_path = '/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2'
    embedding_path = str(Path.home()) + poly_path
    polyglot_emb = Embedding.load(embedding_path)
    polyglot_emb.apply_expansion(CaseExpander)
    emsize = polyglot_emb.shape[1]
    ntokens = len(dm.word2idx)  # the size of vocabulary

    embeddings = torch.zeros(ntokens, emsize)
    for w, i in dm.word2idx.items():
        if w in polyglot_emb:
            embeddings[i] = torch.tensor(polyglot_emb[w])
        else:
            embeddings[i] = torch.randn(emsize, dtype=torch.float)

    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, embeddings, dropout).to(device)

    ######################################################################
    # Run the model
    # -------------
    #

    if padding_eos:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=dm.PAD_IDX)
    lr = 0.05  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    def train_epoch(epoch):
        model.train()  # Turn on the training mode
        total_loss = 0.
        start_time = time.time()
        batch_idxs = list(range(0, len(train_d)))
        shuffle(batch_idxs)
        for i, batch_idx in enumerate(batch_idxs):
            batch = train_d[batch_idx]
            data = batch.to(device)
            targets = batch.reshape(-1).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 100
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | batch {:5d} | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


    def accuracy(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_words = 0
        correct_words = 0
        with torch.no_grad():
            for i, batch in enumerate(data_source):
                batch = batch.to(device)
                y_pred = eval_model.predict(batch)

                batch.permute(1, 0)
                y_pred.permute(1, 0)

                for sentence, pred_sentence in zip(batch, y_pred):
                    for word, predicted in zip(sentence, pred_sentence):
                        word = word.item()
                        predicted = predicted.item()
                        if not padding_eos and word == dm.PAD_IDX:
                            continue

                        total_words += 1
                        if word == predicted:
                            correct_words += 1
        return correct_words / total_words

    def evaluate(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for i, batch in enumerate(data_source):
                data = batch.to(device)
                targets = batch.reshape(-1).to(device)
                output = eval_model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)


    ######################################################################
    # Loop over epochs. Save the model if the validation loss is the best
    # we've seen so far.

    def train_multiple_epochs(epochs, plot_name=""):
        best_val_loss = evaluate(model, dev_d)
        best_model = model
        stats = {'train_loss':[],'valid_loss':[], 'valid_acc': [], 'train_acc': []}
        if padding_eos:
            stats_path = f"saved_models/transformer-padding-eos.stats.json"
        else:
            stats_path = f"saved_models/transformer.stats.json"


        # You may bail early using ctrl+c
        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                train_epoch(epoch)
                train_loss = evaluate(model, train_d)
                val_loss = evaluate(model, dev_d)
                train_acc = accuracy(model, train_d)
                val_acc = accuracy(model, dev_d)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | train loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss), train_loss))

                batch = random.randint(0,len(test_d)-1)
                y_true = test_d[batch][:,:3].to(device)
                y_pred = model.predict(y_true)

                for yt, yp in zip(*list(map(dm.to_sentence, [y_true, y_pred]))):
                    print(f'\tIN:  {yt}')
                    print(f'\tOUT: {yp}')

                print('-' * 89)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

                stats['valid_loss'].append(val_loss)
                stats['train_loss'].append(train_loss)
                stats['valid_acc'].append(val_acc)
                stats['train_acc'].append(train_acc)

                plt.figure()
                plt.plot(stats['valid_loss'])
                plt.plot(stats['train_loss'])
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.legend(['validation', 'training'], loc='upper right')
                plt.savefig(f'plots/transformer{plot_name}.svg')
                plt.savefig(f'plots/transformer{plot_name}.png')
                plt.close()
                with open(stats_path, "w") as file:
                    file.write(json.dumps(stats))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        plt.figure()
        plt.plot(stats['valid_loss'])
        plt.plot(stats['train_loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['validation', 'training'], loc='upper right')
        plt.savefig(f'plots/transformer{plot_name}.svg')
        plt.savefig(f'plots/transformer{plot_name}.png')
        plt.close()

        stats['test_loss'] = evaluate(model, test_d)
        stats['test_acc'] = accuracy(model, test_d)

        with open(stats_path, "w") as file:
            file.write(json.dumps(stats))

        return best_model

    def human_eval(i):
        ex = test_d[i][:, 1].to(device)
        print("Actual:")
        print(' '.join(dm.idx2word[w.item()] for w in ex[ex != 0]))
        y_pred = model.predict(ex)
        print("Predicted:")
        print(' '.join(dm.idx2word[w.item()] for w in y_pred))

    if padding_eos: path = f"saved_models/transformer-padding-eos.pt"
    else: path = f"saved_models/transformer.pt"

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
    else:
        model = train_multiple_epochs(100, '_padding-eos' if padding_eos else '')
        torch.save(model.state_dict(), path)

    # model.load_state_dict(torch.load("saved_models/20epochs_with_max_seq_len128.pt"))
    # best_model = model
    # best_model = train_multiple_epochs(200)

    ######################################################################
    # Evaluate the model with the test dataset
    # -------------------------------------
    #
    # Apply the best model to check the result with the test dataset.

    test_loss = evaluate(model, test_d)
    print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    print('| End of training | test loss {:5.2f}'.format(test_loss))
    print('=' * 89)

    return model