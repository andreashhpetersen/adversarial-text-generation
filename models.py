import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, embeddings=None, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        if embeddings is not None:
            assert ntoken == embeddings.shape[0]
            assert ninp == embeddings.shape[1]
            self.encoder.load_state_dict({ 'weight': embeddings })

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def predict(self, inp):
        remove_batch = False
        if inp.dim() == 1:
            inp = inp.view(-1,1)
            remove_batch = True

        out = self.forward(inp)
        probs = F.softmax(out, dim=2)
        if remove_batch:
            probs = probs.view(probs.shape[0], -1)
        return torch.argmax(probs, dim=1)[inp.view(-1) != 0]

    def predict_sentence(self, inp, idx2word):
        return ' '.join(idx2word[w] for w in self.predict(inp))

    def predict_and_compare(self, inp, idx2word):
        y_pred = self.predict_sentence(inp, idx2word).split(' ')
        inp = inp.view(-1)
        y_true = [idx2word[w] for w in inp[inp != 0]]
        return [(yt, yp, yt == yp) for yt, yp in zip(y_true, y_pred)]
