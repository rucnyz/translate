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
EPOCHS = 200
DROPOUT = 0.2
ENC_HIDDEN_SIZE = DEC_HIDDEN_SIZE = 100
EMBED_SIZE = 100
DEBUG = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


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

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # 变成一维  # (batch_size, 2* enc_hidden_size)
        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  # 修改成decoder可接受hidden size维度  # (1, 64, 100)
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


class BaudanauAttn(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(BaudanauAttn, self).__init__()
        self.W1 = nn.Linear(enc_hidden_size * 2 , dec_hidden_size)
        self.W2 = nn.Linear(dec_hidden_size, dec_hidden_size)
        # self.V = nn.Parameter(torch.rand(dec_hidden_size))

    def forward(self, encoder_hid, encoder_out, mask):
        """
        encoder_hid:max(y_lengths), batch_size, dec_hidden_size  #(h_t)
        encoder_out:batch_size, max(x_lengths), 2 * enc_hidden_size  #(h_s)
        """
        # encoder_hid = encoder_hid.permute(1, 0, 2)  # batch_size, max(y_lengths), dec_hidden_size
        batch_size = encoder_out.shape[0]
        input_len = encoder_hid.shape[1]

        encoder_out = encoder_out[:, -1, :].unsqueeze(1).repeat(1, input_len, 1)

        tanh_W_s_h = torch.tanh(self.W1(encoder_out)+self.W2(encoder_hid))  # (batch_size, max(x_lengths), dec_hidden_size)
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)  # [batch_size, dec_hidden_size, max(x_lengths)]
        # [batch_size,max(y_lengths),dec_hidden_size] * [batch_size,dec_hidden_size,max(x_lengths)]
        # V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [b, max(y_lengths), dec_hidden_size]
        e = torch.bmm(encoder_hid, tanh_W_s_h)  # [b, max(y_lengths), max(x_lengths)]

        e.data.masked_fill(mask, -1e16)
        attn = F.softmax(e, dim = 1)  # [b, max(y_lengths), max(x_lengths)]
        # attn = attn.unsqueeze(1)  # [b, 1, max(x_lengths)]

        ct = torch.bmm(attn, encoder_out)  #ct = at*hs
        #[batch_size,max(y_lengths),max(x_lengths)] * [batch_size, max(x_lengths), 2 * enc_hidden_size]
        #batch_size, max(y_lengths), enc_hidden_size*2

        return ct, attn


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
        # packed: batch_size,max(y_lengths),embed_size     hidden: (1, batch_size, dec_hidden_size)
        out, _ = pad_packed_sequence(out, padding_value = PAD_IDX, batch_first = True)

        output, attn = self.attention(out, encoder_out, mask)  # 输入enc和dec的hidden
        output = self.out(output)
        #batch_size, max(y_lengths), dec_hidden_size --> batch_size, max(y_lengths), vocab_size
        return output, hid, attn  # output为每一个time stamp的输出，hid为最后一个time stamp的输出(attention前)


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(PlainDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden_size, vocab_size)

    def forward(self, encoder_out, x, y, y_lengths, hid):  ## (encoder_out. hid)对应LuongEncoder的输出(outputs, hidden)
        y = self.dropout(self.embedding(y))
        packed = pack_padded_sequence(y, y_lengths.long().cpu().data.numpy(), batch_first = True,
                                      enforce_sorted = False)
        out, hid = self.rnn(packed, hid)  # x的 hid和y同时输入到decoder
        # packed: batch_size,max(y_lengths),embed_size     hidden: (1, batch_size, dec_hidden_size)
        out, _ = pad_packed_sequence(out, padding_value = PAD_IDX, batch_first = True)

        # output, attn = self.attention(out, encoder_out, mask)  # 输入enc和dec的hidden
        output = self.out(out)
        #batch_size, max(y_lengths), dec_hidden_size --> batch_size, max(y_lengths), vocab_size
        return output, hid, 0  # output为每一个time stamp的输出，hid为最后一个time stamp的输出(attention前)


class BaudanauDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(BaudanauDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BaudanauAttn(enc_hidden_size, dec_hidden_size)
        self.rnn1 = nn.GRU(embed_size, dec_hidden_size, batch_first = True)
        self.rnn2 = nn.GRU(embed_size + enc_hidden_size*2, dec_hidden_size, batch_first = True)
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
        # encoder_out (batch_size,seq_len, 2 * enc_hidden_size)
        # hid (1, b, dec_hidden_size)
        # y_lengths = torch.ones(y.shape[0]).long().to(y.device)
        mask = self.creat_mask(y, y)
        y = self.dropout(self.embedding(y))  # batch_size,max(y_lengths),embed_size
        # packed = pack_padded_sequence(y, y_lengths.long().cpu().data.numpy(), batch_first = True, enforce_sorted = False)  # batch_size,1,embed_size

        packed = pack_padded_sequence(y, y_lengths.long().cpu().data.numpy(), batch_first=True,
                                      enforce_sorted=False)
        out, hid = self.rnn1(packed, hid)  # x的 hid和y同时输入到decoder
        # packed: batch_size,max(y_lengths),embed_size     hidden: (1, batch_size, dec_hidden_size)
        out, _ = pad_packed_sequence(out, padding_value=PAD_IDX, batch_first=True)

        ct, attn = self.attention(out, encoder_out, mask)  # hid encoder_out, mask
        # ct: batch_size, 1, enc_hidden_size*2

        gru_input = torch.cat((y, ct), dim=2)  # [b, 1, embed_size + enc_hidden_size*2]

        out, hid = self.rnn2(gru_input, hid)
        # out: (batch_size, 1, dec_hidden_size)  hid: (1, batch_size, dec_hidden_size)
        # out, _ = pad_packed_sequence(out, padding_value = PAD_IDX, batch_first = True)
        # hid = hid.permute(1, 0, 2)  # [b, 1, dec_hidden_size]


        out = self.out(out)
        #batch_size, 1, dec_hidden_size --> batch_size, 1, vocab_size
        return out, hid, attn  # output为每一个time stamp的输出，hid为最后一个time stamp的输出(attention前)

    # def forward(self, encoder_out, x, y, y_lengths, hid):
    #     batch_size = encoder_out.size(0)
    #     decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=DEVICE).fill_(0)
    #     decoder_hidden = hid
    #     decoder_outputs = []
    #     attn_outputs = []
    #
    #     for i in range(y.size(1)):
    #         decoder_output, decoder_hidden, attn_weights = self.forward_step(
    #             encoder_out, x, decoder_input, y_lengths, decoder_hidden)
    #         decoder_outputs.append(decoder_output)
    #         attn_outputs.append(attn_weights)
    #         try:
    #             decoder_input = y[:, i].unsqueeze(1)  # Teacher forcing
    #         except:
    #             break
    #
    #     decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [B, Seq, OutVocab]
    #     return decoder_outputs, decoder_hidden, attn_outputs


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
        return torch.cat(preds, 1)
