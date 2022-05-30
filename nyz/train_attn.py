# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 23:25
# @Author  : HCY
# @File    : train_attn.py
# @Software: PyCharm
from tqdm import tqdm


def train_epoch(model, optimizer, train_data, loss_fn, args):
    model.train()
    losses = 0
    counter = 0
    bar = tqdm(train_data)
    for x, x_length, y, y_length in bar:
        optimizer.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)
        x_lengths = x_length.to(args.device)
        y_lengths = (y_length - 1).to(args.device)
        logits, _ = model(x, x_lengths, y[:, :-1], y_lengths)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1))

        bar.set_postfix(loss = loss.item())
        loss.backward()
        optimizer.step()
        losses += loss.item()
        counter += 1
    return losses / counter


def evaluate(model, dev_data, loss_fn, args):
    model.train()
    losses = 0
    counter = 0
    for batch in tqdm(dev_data):
        x = batch["en"].to(args.device)
        y = batch["cn"].to(args.device)
        x_lengths = batch["en_len"].to(args.device)
        y_lengths = (batch["cn_len"] - 1).to(args.device)
        logits, _ = model(x, x_lengths, y[:, :-1], y_lengths)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1))

        losses += loss.item()
        counter += 1
    return losses / counter
