import numpy as np
import jieba
from collections import Counter  # 计数器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer  # 分词器

UNK_IDX = 0  # 未知
PAD_IDX = 1  #
BATCH_SIZE = 64
EPOCHS = 30
DROPOUT = 0.2
ENC_HIDDEN_SIZE = DEC_HIDDEN_SIZE = 100
EMBED_SIZE = 100
N_HEADS = 5  # EMBED_SIZE整除N_HEADS
N_LAYERS = 6  # encoder 和 decoder 层数
DEBUG = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, train_data, loss_fn):
    model.train()
    losses = 0
    for x, y, x_lengths, y_lengths in train_data:
        '''
        x: [batch_size, src_len]
        y: [batch_size, tgt_len]

        '''
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # x_lengths = x_lengths.to(DEVICE)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(x, y)

        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_data)


def evaluate(model, dev_data, loss_fn):
    model.train()
    losses = 0
    for x, y, x_lengths, y_lengths in dev_data:
        '''
        x: [batch_size, src_len]
        y: [batch_size, tgt_len]

        '''
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(x, y)

        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

        losses += loss.item()

    return losses / len(dev_data)