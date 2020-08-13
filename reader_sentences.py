import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DataManager:
    SOS_TOK, EOS_TOK, UNK_TOK = '<SOS>', '<EOS>', '<unk>'
    PAD_TOK, PAD_IDX = '<PAD>', 0

    def __init__(self, root_dir='./data/', lang='en', min_seq_len=3, max_seq_len=128, one_size=False,
                 normalize_to_max_seq_len_and_eos=True, eos=None):
        if eos is None:
            eos = normalize_to_max_seq_len_and_eos
        self.eos = eos
        data_dir = f'{root_dir}{lang}'
        self.path = f'{data_dir}/sentences.txt'

        self.normalize_to_max_seq_len_and_eos = normalize_to_max_seq_len_and_eos

        self.all_data, self.vocab = self.get_sentences(self.path)
        # split 80:10:10
        self.train_data = self.all_data[int(len(self.all_data) / 5):]
        dev_test_data = self.all_data[:int(len(self.all_data) / 5)]
        self.test_data = dev_test_data[int(len(dev_test_data) / 2):]
        self.dev_data = dev_test_data[:int(len(dev_test_data) / 2)]

        print(f"len(train) = {len(self.train_data)}")
        print(f"len(test) = {len(self.test_data)}")
        print(f"len(dev) = {len(self.dev_data)}")

        mappings = self.make_idx_mappings(self.vocab)
        self.word2idx = mappings[0]
        self.idx2word = mappings[1]
        self.UNK_IDX = self.word2idx[self.UNK_TOK]
        self.SOS_IDX = self.word2idx[self.SOS_TOK]
        self.EOS_IDX = self.word2idx[self.EOS_TOK]

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.one_size = one_size

    def get_sentences(self, filename):
        sentences = []
        words, all_words = [], [self.UNK_TOK, self.EOS_TOK, self.SOS_TOK]
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip('\n')
            words = line.split(' ')
            if self.eos:
                words += [self.EOS_TOK]
            sentences.append(words)
            for word in words:
                all_words.append(word)

        return sentences, self.unique(all_words)

    @staticmethod
    def unique(list):
        seen = set()
        seen_add = seen.add
        return [x for x in list if not (x in seen or seen_add(x))]

    def make_idx_mappings(self, elements):
        """
        Create mappings from elements in a list to indicies (and the other way
        around)
        """
        el2idx, idx2el = {}, {}
        if self.PAD_IDX is not None:
            el2idx[self.PAD_TOK] = self.PAD_IDX
            idx2el[self.PAD_IDX] = self.PAD_TOK

        for el in elements:
            el2idx[el] = len(el2idx)
            idx2el[len(idx2el)] = el
        return el2idx, idx2el

    def to_sentence(self, idxs, as_list=False):
        if type(idxs) == torch.Tensor:
            if len(idxs.shape) > 1:
                return [
                    self.to_sentence(idxs[:, i], as_list=as_list)
                    for i in range(idxs.shape[1])
                ]
            idxs = idxs.tolist()

        if as_list:
            return [self.idx2word[i] for i in idxs]

        return ' '.join([self.idx2word[i] for i in idxs])

    def to_idxs(self, X):
        """
        Convert a list X (eg. tags or words) to a list of indicies
        """
        if not isinstance(X, list):
            X = X.split(' ')
        return torch.tensor([self.word2idx.get(x, self.UNK_IDX) for x in X])

    def batchify(self, X, batch_sz=32):
        """
        Split X into padded batches and return a list of tuples ([batch_data],
        [batch_lengths]), where batch_data is a tensor of shape
        (batch_sz * max_seq_len), with max_seq_len being the length of the longest
        sequence in the batch, and batch_lengths is a list of the lengths of the
        sequences in the batch. The sequences are sorted in decreasing order
        according to their length, ie. max_seq_len == batch_lengths[0] ==
        len(batch_data[0]). The shorter sequences are end-padded with PAD_IDX.

        As an example,

            X = [ [1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14] ]

        would become

        [
            (torch.tensor([
                [4, 5, 6, 7],
                [1, 2, 3, PAD_IDX]
            ]), [4, 3]),
            (torch.tensor([
                [8, 9, 10, 11, 12],
                [13, 14, PAD_IDX, PAD_IDX, PAD_IDX]
            ]), [5, 2])
        ]

        for batch_sz=2.

        Note that len(X) % batch_sz data points are dropped, as I am too lazy to
        handle that properly.
        """
        batches = []
        for i in range(0, len(X), batch_sz):

            # get the data points
            batch = X[i:i + batch_sz]

            # discard last data points if they are fewer than batch_sz
            if len(batch) < batch_sz:
                continue

            # sort the data and get a list of lengths
            batch.sort(key=lambda x: len(x), reverse=True)
            lengths = [len(x) for x in batch]

            # pad it
            if self.normalize_to_max_seq_len_and_eos:
                max_len = self.max_seq_len
            else:
                max_len = lengths[0]

            padded = torch.zeros(batch_sz, max_len, dtype=torch.long) + self.PAD_IDX
            for j in range(batch_sz):
                padded[j, range(lengths[j])] = torch.tensor(batch[j], dtype=torch.long)

            batches.append(padded.T)

        return batches

    def prepare_data(self, data, batch_sz=32):
        """
        Convert raw list of pairs of (words, tags) into their corresponding
        indicies according to word2idx and tag2idx, and return a tuple of
        (X_batches, Y_batches)

        :param data:      list of sentences
        :param word2idx:  dictionary mapping words from string to integers

        :return:          list of batches
        """
        X = [
            self.to_idxs(sen) for sen in data
            if len(sen) == self.max_seq_len
        ] if self.one_size else [
            self.to_idxs(sen) for sen in data
            if self.min_seq_len < len(sen) < self.max_seq_len
        ]
        return self.batchify(X, batch_sz=batch_sz)

    def get_batched_data(self, batch_sz=32):
        train_d = self.prepare_data(self.train_data, batch_sz=batch_sz)
        test_d = self.prepare_data(self.test_data, batch_sz=batch_sz)
        dev_d = self.prepare_data(self.dev_data, batch_sz=batch_sz)
        return train_d, test_d, dev_d

    def compare(self, y_true, y_pred):
        y_true = self.to_sentence(y_true, as_list=True)
        y_pred = self.to_sentence(y_pred, as_list=True)
        return [(yt, yp, yt == yp) for yt, yp in zip(y_true, y_pred)]


if __name__ == '__main__':
    dm = DataManager()
    print(dm.to_idxs("hello my darling , hello my rag time gal"))
