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
    def __init__(self, voc_sz, emb_sz, nhead, hid_sz, nlayers, embeddings=None, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.encoder = nn.Embedding(voc_sz, emb_sz, padding_idx=0)
        if embeddings is not None:
            assert voc_sz == embeddings.shape[0]
            assert emb_sz == embeddings.shape[1]
            self.encoder.load_state_dict({'weight': embeddings})

        self.pos_encoder = PositionalEncoding(emb_sz, dropout)
        encoder_layers = TransformerEncoderLayer(emb_sz, nhead, hid_sz, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.emb_sz = emb_sz
        self.decoder = nn.Linear(emb_sz, voc_sz)

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

        src = self.encoder(src) * math.sqrt(self.emb_sz)
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
        out = self.forward(inp.to(device))
        probs = F.softmax(out, dim=2)
        return torch.argmax(probs, dim=2)


class EncoderRNN(nn.Module):
    def __init__(self, voc_sz, emb_sz, hid_sz, embeddings=None):
        super(EncoderRNN, self).__init__()
        self.hid_sz = hid_sz

        self.embedding = nn.Embedding(voc_sz, emb_sz, padding_idx=0)
        if embeddings is not None:
            assert voc_sz == embeddings.shape[0]
            assert emb_sz == embeddings.shape[1]
            self.embedding.load_state_dict({'weight': embeddings})

        self.gru = nn.GRU(emb_sz, hid_sz)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hid_sz)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hid_sz, voc_sz, emb_sz, embeddings=None, dropout_p=0.1, max_length=128):
        super(AttnDecoderRNN, self).__init__()
        self.hid_sz = hid_sz
        self.voc_sz = voc_sz
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.voc_sz, emb_sz, padding_idx=0)
        if embeddings is not None:
            assert voc_sz == embeddings.shape[0]
            assert emb_sz == embeddings.shape[1]
            self.embedding.load_state_dict({'weight': embeddings})

        self.attn = nn.Linear(emb_sz + hid_sz, self.max_length)
        self.attn_combine = nn.Linear(emb_sz + hid_sz, emb_sz)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(emb_sz, self.hid_sz)
        self.out = nn.Linear(self.hid_sz, self.voc_sz)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hid_sz)


class SimpleGenerator(nn.Module):
    def __init__(self, embeddings, hid_sz=512):
        super(SimpleGenerator, self).__init__()

        vocabulary_size = embeddings.size(0)
        emb_sz = embeddings.size(1)
        self.embedding = nn.Embedding(vocabulary_size, emb_sz, padding_idx=0)
        self.embedding.load_state_dict({'weight': embeddings})

        self.layer1 = nn.Linear(emb_sz, hid_sz)
        self.layer2 = nn.Linear(hid_sz, emb_sz)

    def forward(self, x):
        x = self.embedding(x)
        x = F.leaky_relu(x)  # TODO: is this activation necessary?
        x = self.layer1(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)

        return x


class EncoderGenerator(nn.Module):
    def __init__(self, voc_sz, emb_sz, nhead, hid_sz, nlayers, dropout=0.5):
        super(EncoderGenerator, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.linear1 = nn.Linear(emb_sz, 128)
        self.pos_encoder = PositionalEncoding(128, dropout)
        encoder_layers = TransformerEncoderLayer(128, nhead, hid_sz, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.emb_sz = emb_sz
        self.decoder = nn.Linear(128, emb_sz)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def encode(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.linear1(src)
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
    def __init__(self, emb_sz, seq_sz, nhead, hid_sz, layer_count, dropout=0.5):
        super(EncoderDiscriminator, self).__init__()
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(emb_sz, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            emb_sz, nhead, hid_sz, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layer_count)
        self.emb_sz = emb_sz
        self.decoder = nn.Linear(emb_sz * seq_sz, 1)

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
        seq_sz = x.size(0)
        emb_sz = x.size(2)

        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)

        x = x.permute(1, 0, 2)  # (batch, sentence, embedding)
        x = x.reshape(-1, seq_sz * emb_sz)  # (batch, sentence * embedding)
        x = self.decoder(x)
        x = torch.sigmoid(x).view(-1)

        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, emb_sz, seq_sz, hid_sz=1024):
        super(SimpleDiscriminator, self).__init__()

        self.layer1 = nn.Linear(emb_sz * seq_sz, hid_sz)
        self.layer2 = nn.Linear(hid_sz, hid_sz)
        self.layer3 = nn.Linear(hid_sz, 1)
        # doesn't work

    def forward(self, x):
        batch_size = x.size(1)
        seq_sz = x.size(0)
        emb_sz = x.size(2)

        x = x.permute(1, 0, 2)  # (batch, sentence, embedding)
        x = x.reshape(-1, seq_sz * emb_sz)  # (batch, sentence * embedding)
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
    def __init__(self, voc_szs, max_seq_len=128, hid_sz=64):
        super(Discriminator, self).__init__()

        self.layer_1 = nn.Linear(voc_szs, hid_sz)
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
