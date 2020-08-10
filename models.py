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
            self.encoder.load_state_dict({'weight': embeddings})

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

    def encode(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src, self.src_mask)
        return encoded

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, src):
        encoded = self.encode(src)
        output = self.decode(encoded)
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


class SimpleGenerator(nn.Module):
    def __init__(self, embeddings, hidden_size=512):
        super(SimpleGenerator, self).__init__()

        vocabulary_size = embeddings.size(0)
        embedding_size = embeddings.size(1)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.embedding.load_state_dict({'weight': embeddings})

        self.layer1 = nn.Linear(embedding_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.leaky_relu(x)  # TODO: is this activation necessary?
        x = self.layer1(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)

        return x

class EncoderGenerator(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, embeddings=None, dropout=0.5):
        super(EncoderGenerator, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        if embeddings is not None:
            assert ntoken == embeddings.shape[0]
            assert ninp == embeddings.shape[1]
            self.encoder.load_state_dict({'weight': embeddings})

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ninp)

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

    def encode(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src, self.src_mask)
        return encoded

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, src):
        encoded = self.encode(src)
        output = self.decode(encoded)
        return output


class EncoderDiscriminator(nn.Module):
    def __init__(self, embedding_size, sentence_size, head_count, hidden_size, layer_count, dropout=0.5):
        super(EncoderDiscriminator, self).__init__()
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_size, head_count, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layer_count)
        self.embedding_size = embedding_size
        self.decoder = nn.Linear(embedding_size * sentence_size, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        batch_size = x.size(1)
        sentence_size = x.size(0)
        embedding_size = x.size(2)

        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        # x = self.encoder(x) * math.sqrt(self.embedding_size)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)

        x = x.permute(1, 0, 2)  # (batch, sentence, embedding)
        x = x.reshape(-1, sentence_size * embedding_size)  # (batch, sentence * embedding)
        x = self.decoder(x)
        x = F.sigmoid(x).view(-1)

        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, embedding_size, sentence_size, hidden_size=1024):
        super(SimpleDiscriminator, self).__init__()

        self.layer1 = nn.Linear(embedding_size * sentence_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        # doesn't work

    def forward(self, x):
        batch_size = x.size(1)
        sentence_size = x.size(0)
        embedding_size = x.size(2)

        x = x.permute(1, 0, 2)  # (batch, sentence, embedding)
        x = x.reshape(-1, sentence_size * embedding_size)  # (batch, sentence * embedding)
        x = F.leaky_relu(x)  # TODO: is this activation necessary?
        x = self.layer1(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)
        x = F.leaky_relu(x)
        x = self.layer3(x)
        x = F.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self, voc_sz, out_sz, inp_sz=3, max_seq_len=128, hid_sz=512, emb_sz=None, embs=None):
        super(Generator, self).__init__()

        self.hid_sz = hid_sz
        self.max_seq_len = max_seq_len

        self.embeddings = nn.Embedding(voc_sz, emb_sz, padding_idx=0)
        self.sequence_constructor = nn.Linear(emb_sz * inp_sz, max_seq_len * emb_sz)
        self.recurrent = nn.GRU(emb_sz, hid_sz)
        self.linear = nn.Linear(hid_sz, out_sz)

    def init_hidden(self, batch_sz):
        return torch.randn(1, batch_sz, self.hid_sz)

    def forward(self, inp):
        batch_sz = inp.shape[1]
        self.hidden = self.init_hidden(batch_sz)

        embs = self.embeddings(inp).view(1, -1)  # concat embeddings
        X = F.leaky_relu(self.sequence_constructor(embs))
        X = X.view(self.max_seq_len, batch_sz, -1)  # make sequence
        X, _ = self.recurrent(X, self.hidden)
        X = F.leaky_relu(self.linear(X))

        return X


class Discriminator(nn.Module):
    def __init__(self, ntokens, max_seq_len=128, hid_sz=64):
        super(Discriminator, self).__init__()

        self.layer_1 = nn.Linear(ntokens, hid_sz)
        self.layer_2 = nn.Linear(max_seq_len * hid_sz, hid_sz)
        self.layer_out = nn.Linear(hid_sz, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = x.view(1, -1)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        x = x.view(-1)

        return x
