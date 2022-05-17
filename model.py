# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 23:19
# @Author  : HCY
# @File    : model.py
# @Software: PyCharm
import numpy as np
import jieba
from collections import Counter  #计数器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer  #分词器

UNK_IDX = 0  #未知
PAD_IDX = 1  #
BATCH_SIZE = 64
EPOCHS = 30
DROPOUT = 0.2
ENC_HIDDEN_SIZE = DEC_HIDDEN_SIZE = 100
EMBED_SIZE = 100
DEBUG = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LuongEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(LuongEncoder, self).__init__()

        # 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值: vocab_size 词典的大小尺寸, embed_size 嵌入向量的维度
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size,
                          bidirectional=True, batch_first=True)  # 双向GRU（embed_size输入特征维度，enc_hidden_size输出特征维度）
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, x_lengths):  # x_lengths: 输入句子长度
        """
        input_seqs : batch_size,max(x_lengths)
        input_lengths: batch_size
        """
        embedded = self.dropout(self.embedding(x))  # batch_size,max(x_lengths),embed_size
        packed = pack_padded_sequence(embedded, x_lengths.long().cpu().data.numpy(), batch_first=True,
                                      enforce_sorted=False)
        # batch_first = False (seq, batch, feature)  batch_first = True (batch, seq, feature)

        # 压缩填充张量,压缩掉无效的填充值
        # enforce_sorted：如果是 True ，则输入应该是按长度降序排序的序列。如果是 False ，会在函数内部进行排序
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, padding_value=PAD_IDX, batch_first=True)  # 还原

        # hidden (2, batch_size, enc_hidden_size)  # 2:双向
        # outputs (batch_size,seq_len, 2 * enc_hidden_size)  h_s

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # 变成一维
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # 修改成decoder可接受hidden size维度
        return outputs, hidden  # outputs为每一个time stamp的输出，hidden为最后一个time stamp的输出


class Attn(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attn, self).__init__()
        #general attention
        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias = False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, encoder_out, mask):
        """
        output:batch_size, max(y_lengths), dec_hidden_size  #(h_t)
        encoder_out:batch_size, max(x_lengths), 2 * enc_hidden_size  #(h_s)
        """
        batch_size = output.shape[0]
        output_len = output.shape[1]
        input_len = encoder_out.shape[1]

        encoder_out1 = self.linear_in(encoder_out.view(batch_size * input_len, -1)).view(batch_size, input_len, -1)
        #Wh_s
        #batch_size,max(x_lengths),dec_hidden_size
        score = torch.bmm(output, encoder_out1.transpose(1, 2))  #实现三维数组的乘法，而不用拆成二维数组使用for循环解决
        #[batch_size,max(y_lengths),dec_hidden_size] * [batch_size,dec_hidden_size,max(x_lengths)]
        #batch_size,max(y_lengths),max(x_lengths)  #score = h_t W h_s
        score.data.masked_fill(mask, -1e16)
        attn = F.softmax(score, dim = 2)  #attention系数矩阵, mask均为0
        # mask的size是(batch_size,n,m)
        # mask中，若该位置为True，就表明score中该位置要被mask掉，用-1e6来代替。
        # PS mask中，若某个位置K_sub(b,i,j)为True，表明这个batch中的第b句话的中文的第i个字是padding or 英文的第j个单词是padding or 两个都是padding
        # 若某个位置K_sub(b,i,j)为False，表明这个batch中的第b句话的中文的第i个字不是padding且英文的第j个单词也不是padding

        ct = torch.bmm(attn, encoder_out)  #ct = aths
        #[batch_size,max(y_lengths),max(x_lengths)] * [batch_size, max(x_lengths), 2 * enc_hidden_size]
        #batch_size, max(y_lengths), enc_hidden_size*2
        output = torch.cat((ct, output), dim = 2)  #batch_size, max(y_lengths), enc_hidden_size*2 + dec_hidden_size

        output = output.view(batch_size * output_len, -1)  #batch_size * max(y_lengths), enc_hidden_size*2 + dec_hidden_size
        output = torch.tanh(self.linear_out(output))  #batch_size * max(y_lengths), dec_hidden_size
        output = output.view(batch_size, output_len, -1)
        #batch_size, max(y_lengths), dec_hidden_size

        return output, attn


class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(LuongDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attn(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden_size, vocab_size)

    def creat_mask(self, x, y):
        x_mask = x.data != PAD_IDX  #batch_size,max(x_lengths)  # 不等于为true，不是padding
        y_mask = y.data != PAD_IDX  #batch_size,max(y_lengths)
        mask = (1 - (x_mask.unsqueeze(2) * y_mask.unsqueeze(1)).float()).bool()  # true为padding
        # unsqueeze增加维度
        #batch_size,max(x_lengths),max(y_lengths)
        #attn为batch_size,max(y_lengths),max(x_lengths)，因此y与x对调
        return mask

    def forward(self, encoder_out, x, y, y_lengths, hid):  ## (encoder_out. hid)对应LuongEncoder的输出(outputs, hidden)
        mask = self.creat_mask(y, x)
        y = self.dropout(self.embedding(y))
        packed = pack_padded_sequence(y, y_lengths.long().cpu().data.numpy(), batch_first = True,
                                      enforce_sorted = False)
        out, hid = self.rnn(packed, hid)  # x的 hid和y同时输入到decoder

        out, _ = pad_packed_sequence(out, padding_value = PAD_IDX, batch_first = True)

        output, attn = self.attention(out, encoder_out, mask)  # 输入enc和dec的hidden
        output = self.out(output)
        #batch_size, max(y_lengths), dec_hidden_size --> batch_size, max(y_lengths), vocab_size
        return output, hid, attn  # output为每一个time stamp的输出，hid为最后一个time stamp的输出(attention前)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(encoder_out,  #这里输出的hid是decoder_rnn的hid
                                         x = x,
                                         y = y,
                                         y_lengths = y_lengths,
                                         hid = hid)  #encoder的hid
        return output, attn

    def translate(self, x, x_lengths, y, max_length = 15):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for _ in range(max_length):
            output, hid, attn = self.decoder(encoder_out,
                                             x = x,
                                             y = y,
                                             y_lengths = torch.ones(batch_size).long().to(y.device),  # 逐字翻译！
                                             hid = hid)

            y = output.max(2)[1].view(batch_size, 1)

            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


