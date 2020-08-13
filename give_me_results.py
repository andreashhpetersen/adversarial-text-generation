import os
import time
import math
import torch
import torch.nn as nn
from random import shuffle

from models import TransformerModel, Generator, Discriminator
from reader import DataManager

from polyglot.mapping import Embedding, CaseExpander
from pathlib import Path
from models import TransformerModel


def transformer_model(padding_eos):
    import main
    main.run(padding_eos)


def rnn_model_results():
    pass


def gan_results():
    pass


transformer_model(False)
transformer_model(True)
