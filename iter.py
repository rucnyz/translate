# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 9:05
# @Author  : HCY
# @File    : iter.py
# @Software: PyCharm

import os.path
import time

import jieba
import torch
from torch import Tensor
from torch.utils.data import dataset
from torchdata.datapipes.iter import FileOpener, FileLister
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

vocab_en = torch.load("vocab_en.pt")
vocab_cn = torch.load("vocab_cn.pt")

train_en_file = 'en-zh/train.en'
train_cn_file = 'en-zh/train.zh'
test_en_file = 'en-zh/test.en'
test_cn_file = 'en-zh/test.zh'

tokenizer_en = get_tokenizer('basic_english')
tokenizer_cn = get_tokenizer(jieba.lcut)  # 进行结巴分词

# 开始训练重新读数据
train_en_iter = FileLister([train_en_file])
train_en_iter = FileOpener(train_en_iter).readlines().map(lambda x: x[1])
train_en_iter = train_en_iter.map(tokenizer_en).map(vocab_en)

train_cn_iter = FileLister([train_cn_file])
train_cn_iter = FileOpener(train_cn_iter).readlines().map(lambda x: x[1])
train_cn_iter = train_cn_iter.map(tokenizer_cn).map(vocab_cn)

test_en_iter = FileLister([test_en_file])
test_en_iter = FileOpener(test_en_iter).readlines().map(lambda x: x[1])
test_en_iter = test_en_iter.map(tokenizer_en).map(vocab_en)

test_cn_iter = FileLister([test_cn_file])
test_cn_iter = FileOpener(test_cn_iter).readlines().map(lambda x: x[1])
test_cn_iter = test_cn_iter.map(tokenizer_cn).map(vocab_cn)

def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    # data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) for item in raw_text_iter]
    for item in raw_text_iter:
        tmp = torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_en_data = data_process(train_en_iter, vocab_en, tokenizer_en)
train_cn_data = data_process(train_cn_iter, vocab_cn, tokenizer_cn)
test_en_data = data_process(test_en_iter, vocab_en, tokenizer_en)
test_cn_data = data_process(test_cn_iter, vocab_cn, tokenizer_cn)