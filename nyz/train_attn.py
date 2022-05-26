# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 23:25
# @Author  : HCY
# @File    : train_attn.py
# @Software: PyCharm
import seaborn
import torch
from tqdm import tqdm
from thop import profile

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
    counter = 0
    bar = tqdm(train_data)
    for batch in bar:
        optimizer.zero_grad()
        x = batch["en"].to(DEVICE)
        y = batch["cn"].to(DEVICE)
        x_lengths = batch["en_len"].to(DEVICE)
        y_lengths = (batch["cn_len"] - 1).to(DEVICE)
        logits, _ = model(x, x_lengths, y[:, :-1], y_lengths)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1))

        bar.set_postfix(loss = loss.item(), dim_x = x.shape[1], dim_y = y.shape[1])
        loss.backward()
        optimizer.step()
        losses += loss.item()
        counter += 1
    return losses / counter


seaborn.heatmap


def evaluate(model, dev_data, loss_fn):
    model.train()
    losses = 0
    counter = 0
    for batch in tqdm(dev_data):
        x = batch["en"].to(DEVICE)
        y = batch["cn"].to(DEVICE)
        x_lengths = batch["en_len"].to(DEVICE)
        y_lengths = (batch["cn_len"] - 1).to(DEVICE)
        logits, _ = model(x, x_lengths, y[:, :-1], y_lengths)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1))

        losses += loss.item()
        counter += 1
    return losses / counter
