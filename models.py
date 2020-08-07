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
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def predict(self, inp):
        device = inp.device
        remove_batch = False
        if inp.dim() == 1:
            inp = inp.view(-1, 1)
            remove_batch = True

        out = self.forward(inp.to(device))
        probs = F.softmax(out, dim=2)
        if remove_batch:
            probs = probs.view(probs.shape[0], -1)
        return torch.argmax(probs, dim=1)[inp.view(-1) != 0]


class TransformerModel2(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, embeddings=None, dropout=0.5):
        super(TransformerModel2, self).__init__()
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

        ### We need to do something here...
        self.encode_sentence = nn.Linear(128*ninp, 128)
        self.ninp = ninp
        # ... here ...
        self.decoder = nn.Linear(128, ntoken)

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
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        import ipdb; ipdb.set_trace()

        # ... and here ...
        sent_enc = self.encode_sentence(output.view(1, src.shape[1], -1))
        # ... and here.
        output = self.decoder(sent_enc)

        return output

    def predict(self, inp):
        device = inp.device
        remove_batch = False
        if inp.dim() == 1:
            inp = inp.view(-1, 1)
            remove_batch = True

        out = self.forward(inp.to(device))
        probs = F.softmax(out, dim=2)
        if remove_batch:
            probs = probs.view(probs.shape[0], -1)
        return torch.argmax(probs, dim=1)[inp.view(-1) != 0]


class Generator(nn.Module):
    def __init__(self, voc_sz, out_sz, inp_sz=3, max_seq_len=128, hid_sz=512, emb_sz=None, embs=None):
        super(Generator, self).__init__()

        self.hid_sz = hid_sz
        self.max_seq_len = max_seq_len

        self.embeddings = nn.Embedding(voc_sz, emb_sz, padding_idx=0)
        self.sequence_constructor = nn.Linear(emb_sz*inp_sz, max_seq_len*64)
        self.recurrent = nn.GRU(max_seq_len, hid_sz)
        self.linear = nn.Linear(hid_sz, out_sz)

    def init_hidden(self, batch_sz):
        return torch.randn(1, batch_sz, self.hid_sz)

    def forward(self, inp):
        batch_sz = inp.shape[1]
        self.hidden = self.init_hidden(batch_sz)

        embs = self.embeddings(inp)
        X = F.leaky_relu(self.sequence_constructor(embs.view(-1, batch_sz).T))
        import ipdb; ipdb.set_trace()
        X, _ = self.recurrent(X.view(self.max_seq_len, batch_sz, -1), self.hidden)
        X = F.leaky_relu(self.linear2(X.view(-1, 1)))

        return X


class Discriminator(nn.Module):
    def __init__(self, enc_sz):

        # Number of input features is 12.
        self.layer_1 = nn.Linear(enc_sz, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
