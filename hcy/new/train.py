# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 13:36
# @Author  : nieyuzhou
# @File    : train.py
# @Software: PyCharm
import os

from torch.utils.data import DataLoader

from build import build_pipe, build_vocab, compute_args


# 可视化迭代器的前num个元素
def visualize(iters, num = 5):
    for i, j in enumerate(iters):
        print(j)
        if i == num:
            break


if __name__ == '__main__':
    if os.getcwd().endswith("nyz"):
        os.chdir("..")
    # 基本配置
    data_root = "en-zh/"
    args = compute_args(data_root)
    # 构建字典
    tokenizer_en, tokenizer_cn, vocab_en, vocab_cn = build_vocab(args)
    # 构建iterator和loader
    train_iter = build_pipe(args, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn, args.train_en_file,
                            args.train_cn_file)
    test_iter = build_pipe(args, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn, args.test_en_file, args.test_cn_file)
    train_data = DataLoader(dataset = train_iter, batch_size = None)
    test_data = DataLoader(dataset = test_iter, batch_size = None)

    # 开始训练
    for i, batch in enumerate(train_data):
        # 每个batch的原长度
        print(batch["en_len"])
        print(batch["cn_len"])
        # 每个batch中补全后的中英文数据
        print(batch["en"].shape)
        print(batch["cn"].shape)
        print("----------------")
        if i > 5:
            break

    # def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    #     """Converts raw text into a flat Tensor."""
    #     data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) for item in raw_text_iter]
    #     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
