import time
import math
import torch
import torch.nn as nn
from random import shuffle

from models import TransformerModel, Generator, Discriminator
from reader import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path

dm = DataManager()
train_d, test_d, dev_d = dm.get_batched_data(batch_sz=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from polyglot.downloader import downloader
# downloader.download('embeddings2.en')

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

criterion = nn.CrossEntropyLoss(ignore_index=dm.PAD_IDX, reduction='mean')
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

def train_multiple_epochs():
    best_val_loss = evaluate(model, dev_d)
    epochs = 20  # The number of epochs
    best_model = model

    # You may bail early using ctrl+c
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_epoch(epoch)
            val_loss = evaluate(model, dev_d)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return best_model


model.load_state_dict(torch.load("saved_models/20epochs_with_max_seq_len128.pt"))
best_model = model
# best_model = train_multiple_epochs()

######################################################################
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.

test_loss = evaluate(best_model, test_d)
print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
print('| End of training | test loss {:5.2f}'.format(test_loss))
print('=' * 89)


def human_eval(i):
    ex = test_d[i][:, 1].to(device)
    print("Actual:")
    print(' '.join(dm.idx2word[w.item()] for w in ex[ex != 0]))
    y_pred = model.predict(ex)
    print("Predicted:")
    print(' '.join(dm.idx2word[w.item()] for w in y_pred))
