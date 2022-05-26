
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        '''
        输入 = 输入 + PE
        PE(pos, 2i) = sin(pos/10000^(2i/embed_size))
        PE(pos, 2i+1) = cos(pos/10000^(2i/embed_size))
        pos 是位置索引(0,seq_len)，i是维度索引(0,embed_size)

        '''
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, embed_size]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class myTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, n_heads, n_layers, d_ff, dropout):
        super(myTransformer, self).__init__()
        '''
        
        embed_size: Embedding Size
        n_heads: the number of heads in Multi-Head Attention
        n_layers: the nuber of Encoder or Decoder Layer
        d_ff: FeedForward dimension
        '''
        # d_model:the number of expected features in the encoder/decoder inputs(embedding size)
        # nhead:the number of heads in the multiheadattention models
        # d_model 必须能够整除n_heads
        self.transformer = nn.Transformer(d_model = embed_size, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff)
        self.src_emb = nn.Embedding(src_vocab_size, embed_size)
        self.pos_emb = PositionalEncoding(embed_size,dropout)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, embed_size)
        self.projection = nn.Linear(embed_size, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        b, src_len = enc_inputs.shape[0], enc_inputs.shape[1]
        b, tgt_len = dec_inputs.shape[0], dec_inputs.shape[1]

        src_mask = self.transformer.generate_square_subsequent_mask(src_len).cuda()
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()
        memory_mask = None
        # 找到padding的位置
        src_key_padding_mask = enc_inputs.data.eq(0).cuda()                     # [N,S]
        tgt_key_padding_mask = dec_inputs.data.eq(0).cuda()                     # [N,T]
        memory_key_padding_mask = src_key_padding_mask                 # [N,S]
        
        # 嵌入向量
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, embed_size]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).cuda()  # [ src_len, batch_size, embed_size]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, embed_size]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).cuda()  # [ tgt_len,batch_size, embed_size]
        
        # 送入Transformer
        # encoder 部分Multihead-attention仅需要遮住padding部分；
        # decoder部分有self-attn和cross-attn两部分，
        # self-attn需要遮住dec_input的padding部分和未来信息部分，
        # cross-atten 部分 Q = tgt, K,V均来自encoder 称之为memory,multihead-attntion 部分仅需要遮住padding部分，
        dec_outputs  = self.transformer(src= enc_outputs, tgt = dec_outputs, src_mask = None, tgt_mask = tgt_mask,
                                        memory_mask = None, src_key_padding_mask = src_key_padding_mask,
                                        tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask)

        # 维度变换
        dec_logits = self.projection(dec_outputs.transpose(0,1))  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), None, None, None
