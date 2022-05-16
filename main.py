import numpy as np
import jieba
from collections import Counter#计数器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence ,pack_padded_sequence,pad_packed_sequence
from torchtext.data.utils import get_tokenizer#分词器

UNK_IDX = 0 #未知
PAD_IDX = 1  #
BATCH_SIZE = 64
EPOCHS  = 30
DROPOUT = 0.2
ENC_HIDDEN_SIZE = DEC_HIDDEN_SIZE = 100
EMBED_SIZE = 100
## 数据集文件
train_en_file = 'en-zh/train.en'
train_cn_file = 'en-zh/train.zh'
test_en_file = 'en-zh/test.en'
test_cn_file = 'en-zh/test.zh'
save_file = 'model.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#分词器
tokenizer_en = get_tokenizer('basic_english')#按空格进行分割
tokenizer_cn = get_tokenizer(jieba.lcut)#进行结巴分词


# 加载文件
def load_data(path, language):
    text = []
    # 语言为中文
    if language == 'cn':
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                i = i + 1
                if i % 1000000 == 0:
                    print(str(i))
                try:

                    line = line.strip().split('\t')
                    text.append(["BOS"] + tokenizer_cn(line[0]) + ["EOS"])
                except Exception as e:
                    print("error raise!")

        return text

    # 语言为英文
    elif language == 'en':
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                i = i + 1
                if i % 1000000 == 0:
                    print(str(i))
                try:
                    line = line.strip().split('\t')
                    text.append(["BOS"] + tokenizer_en(line[0].lower()) + ["EOS"])  # 小写
                except Exception as e:
                    print("error raise!")

        return text

    # 语言非中文和非英文
    else:
        print("Can't handle the language!")
        return

train_en = load_data(test_en_file, 'en')
train_zh = load_data(train_cn_file, 'cn')