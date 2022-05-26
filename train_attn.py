# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 23:25
# @Author  : HCY
# @File    : train_attn.py
# @Software: PyCharm

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
DEBUG = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, train_data, loss_fn):
    model.train()
    losses = 0
    for batch in train_data:
        x = batch["en"]
        y = batch["cn"]
        x_lengths = batch["en_len"]
        y_lengths = batch["cn_len"]
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x_lengths = x_lengths.to(DEVICE)

        y_input = y[:, :-1]  # 将前seq-1个单词作为输入
        y_output = y[:, 1:]  # 将后seq-1个单词作为输出，相当于前一个单词预测后一个单词
        y_lengths = (y_lengths - 1).to(DEVICE)

        logits, _ = model(x, x_lengths, y_input, y_lengths)  # batch_size, max(y_lengths), vocab_size
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_data)


def evaluate(model, dev_data, loss_fn):
    model.train()
    losses = 0
    for x, y, x_lengths, y_lengths in dev_data:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x_lengths = x_lengths.to(DEVICE)

        y_input = y[:, :-1]
        y_output = y[:, 1:]
        y_lengths = (y_lengths - 1).to(DEVICE)
        logits, _ = model(x, x_lengths, y_input, y_lengths)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_output.reshape(-1))

        losses += loss.item()

    return losses / len(dev_data)
