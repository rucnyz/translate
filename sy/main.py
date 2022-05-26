import numpy as np
import jieba
from collections import Counter  #计数器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer  #分词器

from train import train_epoch, evaluate
from model import myTransformer
import torch.optim as optim


UNK_IDX = 0  #未知
PAD_IDX = 1  #
BATCH_SIZE = 64
EPOCHS = 30
DROPOUT = 0.2
ENC_HIDDEN_SIZE = DEC_HIDDEN_SIZE = 100
EMBED_SIZE = 100
N_HEADS = 5#EMBED_SIZE整除N_HEADS
N_LAYERS = 6 #encoder 和 decoder 层数
D_FF = 2048 # 前馈全连接神经网络的维度
DEBUG = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEBUG:

    train_file = 'attn_data/train_mini.txt'
    dev_file = 'attn_data/dev_mini.txt'
    test_file = 'attn_data/test_mini.txt'
    save_file = 'model.pt'
else:

    train_file = '../attn_data/train.txt'
    dev_file = '../attn_data/dev.txt'
    test_file = '../attn_data/test.txt'
    save_file = '../large_model.pt'

#分词器
tokenizer_en = get_tokenizer('basic_english')  #按空格进行分割
tokenizer_cn = get_tokenizer(jieba.lcut)  #进行结巴分词


#加载文件
def load_data(path):
    en = []
    cn = []
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            en.append(["BOS"] + tokenizer_en(line[0].lower()) + ["EOS"])  #小写
            cn.append(["BOS"] + tokenizer_cn(line[1]) + ["EOS"])
    return en, cn


train_en, train_zh = load_data(train_file)
dev_en, dev_zh = load_data(dev_file)
test_en, test_zh = load_data(test_file)

#构建词汇表
def build_dict(sentences, max_words = 50000):
    vocab = Counter(np.concatenate(sentences)).most_common(max_words)  #最大单词数是50000
    word_to_id = {w[0]: index + 2 for index, w in enumerate(vocab)}
    word_to_id['UNK'] = UNK_IDX  #0
    word_to_id['PAD'] = PAD_IDX  #1
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


en_wtoi, en_itow = build_dict(train_en)
zh_wtoi, zh_itow = build_dict(train_zh)


# 利用词典对原始句子编码 单词->数字
def encode(en_sentences, ch_sentences, en_wtoi, zh_wtoi, sort_by_len = True):
    out_en_sentences = [[en_wtoi.get(w, UNK_IDX) for w in sent] for sent in en_sentences]
    out_ch_sentences = [[zh_wtoi.get(w, UNK_IDX) for w in sent] for sent in ch_sentences]

    #返回w对应的值，否则返回UNK_IDX
    def len_argsort(seq):  #按照长度进行排序
        return sorted(range(len(seq)), key = lambda x: len(seq[x]))

    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_ch_sentences = [out_ch_sentences[i] for i in sorted_index]

    return out_en_sentences, out_ch_sentences


train_en_encode, train_zh_encode = encode(train_en, train_zh, en_wtoi, zh_wtoi)
dev_en_encode, dev_zh_encode = encode(dev_en, dev_zh, en_wtoi, zh_wtoi)
test_en_encode, test_zh_encode = encode(test_en, test_zh, en_wtoi, zh_wtoi)


#返回每个batch的id
def get_minibatches(n, minibatch_size, shuffle = True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

#将句子对划分到batch
def get_batches(en_encode, ch_encode):
    batch_indexs = get_minibatches(len(en_encode), BATCH_SIZE)

    batches = []
    for batch_index in batch_indexs:
        batch_en = [torch.tensor(en_encode[index]).long() for index in batch_index]  #每一个idx对应的句子，转为tensor格式
        batch_zh = [torch.tensor(ch_encode[index]).long() for index in batch_index]
        length_en = torch.tensor([len(en) for en in batch_en]).long()  #每一个句子的长度
        length_zh = torch.tensor([len(zh) for zh in batch_zh]).long()

        batch_en = pad_sequence(batch_en, padding_value = PAD_IDX, batch_first = True)  #讲一个batch中的句子padding为相同长度
        batch_zh = pad_sequence(batch_zh, padding_value = PAD_IDX, batch_first = True)

        batches.append((batch_en, batch_zh, length_en, length_zh))
    return batches


train_data = get_batches(train_en_encode, train_zh_encode)
dev_data = get_batches(dev_en_encode, dev_zh_encode)

# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model

model = myTransformer(src_vocab_size = len(en_itow),\
                        tgt_vocab_size = len(zh_itow),\
                        embed_size = EMBED_SIZE,\
                        n_heads = N_HEADS,\
                        n_layers = N_LAYERS,\
                        d_ff = D_FF,\
                        dropout = DROPOUT).cuda()
model = model.to(DEVICE)

           
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  #忽略padding位置的损失
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# 训练模型
from timeit import default_timer as timer

for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, train_data, loss_fn)
    end_time = timer()
    val_loss = evaluate(model, dev_data, loss_fn)
    print((
              f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

