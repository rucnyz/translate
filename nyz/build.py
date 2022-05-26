# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 5:09
# @Author  : nieyuzhou
# @File    : build.py
# @Software: PyCharm
import argparse
import os
import random
import time

import jieba
import numpy as np
import torch
from torchdata.datapipes.iter import FileOpener, FileLister
from torchtext.data import get_tokenizer
from torchtext.functional import to_tensor
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms


def seed_everything(seed = 123):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_args(data_root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = "Transformer", choices = ["Transformer"])
    parser.add_argument('--batch_size', type = int, default = 50)
    parser.add_argument('--vocab_size', type = int, default = 50000)
    parser.add_argument('--truncate_size', type = int, default = 128)
    parser.add_argument('--device', type = str, default = "cpu", choices = ["cpu", "cuda"])
    parser.add_argument('--seed', type = int, default = 123)
    args = parser.parse_args()
    # 设置随机数种子
    seed_everything(args.seed)
    # 文件路径

    args.train_en_file = data_root + 'train.en'
    args.train_cn_file = data_root + 'train.zh'
    args.test_en_file = data_root + 'test.en'
    args.test_cn_file = data_root + 'test.zh'
    # 特殊标识设置
    args.unk_idx = 0
    args.padding_idx = 1
    args.bos_idx = 2
    args.eos_idx = 3
    # 使用设备
    if args.device != "cpu":
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备: {}".format(args.device))
    return args


def build_vocab(args):
    # 用训练集构建字典
    en_iter = FileLister([args.train_en_file])
    en_iter = FileOpener(en_iter).readlines().map(lambda x: x[1])

    cn_iter = FileLister([args.train_cn_file])
    cn_iter = FileOpener(cn_iter).readlines().map(lambda x: x[1])

    start_time = time.time()
    tokenizer_en = get_tokenizer('basic_english')
    tokenizer_cn = get_tokenizer(jieba.lcut)  # 结巴分词
    if os.path.exists("en-zh/vocab_en.pt"):
        vocab_en = torch.load("en-zh/vocab_en.pt")
    else:
        # 字典中插入
        vocab_en = build_vocab_from_iterator(map(tokenizer_en, en_iter), max_tokens = args.vocab_size,
                                             specials = ['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab_en.set_default_index(vocab_en['<unk>'])
        torch.save(vocab_en, "en-zh/vocab_en.pt")

    print("英文读入完成")

    if os.path.exists("en-zh/vocab_cn.pt"):
        vocab_cn = torch.load("en-zh/vocab_cn.pt")
    else:
        vocab_cn = build_vocab_from_iterator(map(tokenizer_cn, cn_iter), max_tokens = args.vocab_size,
                                             specials = ['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab_cn.set_default_index(vocab_cn['<unk>'])
        torch.save(vocab_cn, "en-zh/vocab_cn.pt")
    print("中文读入完成")
    print("所用时间: {:.4f}".format(time.time() - start_time))
    return tokenizer_en, tokenizer_cn, vocab_en, vocab_cn


def build_pipe(args, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn, en_file, cn_file):
    # 构建插入开始结尾符转换器
    transform = transforms.Sequential(
        transforms.Truncate(args.truncate_size - 2),
        transforms.AddToken(token = args.bos_idx, begin = True),
        transforms.AddToken(token = args.eos_idx, begin = False),
    )
    # 开始训练重新读数据
    en_iter = FileLister([en_file])
    en_iter = FileOpener(en_iter).readlines().map(lambda x: x[1])
    en_iter = en_iter.map(tokenizer_en).map(vocab_en).map(transform)

    cn_iter = FileLister([cn_file])
    cn_iter = FileOpener(cn_iter).readlines().map(lambda x: x[1])
    cn_iter = cn_iter.map(tokenizer_cn).map(vocab_cn).map(transform)
    # padding补全
    concat_iter = en_iter.zip(cn_iter).batch(args.batch_size).rows2columnar(["english", "chinese"]).map(
        lambda x: {"en": to_tensor(x["english"], padding_value = args.padding_idx),
                   "en_len": torch.tensor([len(en) for en in x["english"]]).long(),
                   "cn": to_tensor(x["chinese"], padding_value = args.padding_idx),
                   "cn_len": torch.tensor([len(cn) for cn in x["chinese"]]).long()})
    return concat_iter
