# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 22:48
# @Author  : nieyuzhou
# @File    : textDataset.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
from torchtext import transforms
from torchtext.functional import to_tensor


# 用于collate_fn
def pad_text(data, padding_idx):
    length = len(data)
    x = to_tensor([data[i][0] for i in range(length)], padding_value = padding_idx)
    y = to_tensor([data[i][2] for i in range(length)], padding_value = padding_idx)
    x_length = torch.tensor([data[i][1] for i in range(length)]).long()
    y_length = torch.tensor([data[i][3] for i in range(length)]).long()
    return x, x_length, y, y_length


class textDataset(Dataset):
    def __init__(self, args, en_file, cn_file, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn):
        super(textDataset, self).__init__()
        # self.raw_en_data = open(args.train_en_file, 'r').read().splitlines()
        # self.raw_cn_data = open(args.train_cn_file, 'r').read().splitlines()
        self.raw_en_data = open(en_file, 'r').readlines()
        self.raw_cn_data = open(cn_file, 'r').readlines()
        self.vocab_en = vocab_en
        self.vocab_cn = vocab_cn
        self.tokenizer_en = tokenizer_en
        self.tokenizer_cn = tokenizer_cn
        self.transform = transforms.Sequential(
            transforms.Truncate(args.truncate_size - 2),
            transforms.AddToken(token = args.bos_idx, begin = True),
            transforms.AddToken(token = args.eos_idx, begin = False),
        )
        self.padding_idx = args.padding_idx

    def __getitem__(self, idx):
        x = self.transform(self.vocab_en(self.tokenizer_en(self.raw_en_data[idx][:-1])))
        y = self.transform(self.vocab_cn(self.tokenizer_cn(self.raw_cn_data[idx][:-1])))
        x_length = len(x)
        y_length = len(y)
        # x = to_tensor(x, padding_value = self.padding_idx)
        # y = to_tensor(y, padding_value = self.padding_idx)
        return x, x_length, y, y_length

    def __len__(self):
        return len(self.raw_en_data)
