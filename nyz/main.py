import os
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from build import build_vocab, compute_args
from model import LuongEncoder, LuongDecoder, seq2seq
from textDataset import textDataset, pad_text
from train_attn import train_epoch, evaluate


# 可视化迭代器的前num个元素
def visualize(iters, num = 5):
    for i, j in enumerate(iters):
        print(j)
        if i == num:
            break


if os.getcwd().endswith("nyz"):
    os.chdir("..")

# 分布式训练
torch.distributed.init_process_group(backend = 'nccl')
# 基本配置
data_root = "en-zh/"
args = compute_args(data_root)
# 构建字典
tokenizer_en, tokenizer_cn, vocab_en, vocab_cn = build_vocab(args)
# 构建iterator和loader
train_dataset = textDataset(args, args.train_en_file, args.train_cn_file, vocab_en, vocab_cn, tokenizer_en,
                            tokenizer_cn)
test_dataset = textDataset(args, args.test_en_file, args.test_cn_file, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn)
# train_iter = build_pipe(args, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn, args.train_en_file,
#                         args.train_cn_file)
# test_iter = build_pipe(args, vocab_en, vocab_cn, tokenizer_en, tokenizer_cn, args.test_en_file, args.test_cn_file)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
# train_data = DataLoader(dataset = train_iter, batch_size = None, shuffle = False)
# dev_data = DataLoader(dataset = test_iter, batch_size = None, shuffle = False)

train_data = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False, sampler = train_sampler,
                        collate_fn = lambda x: pad_text(x, args.padding_idx), num_workers = args.num_workers)
dev_data = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, sampler = test_sampler,
                      collate_fn = lambda x: pad_text(x, args.padding_idx), num_workers = args.num_workers)
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model
encoder = LuongEncoder(vocab_size = len(vocab_en), embed_size = args.emb_size, enc_hidden_size = args.enc_hidden_size,
                       dec_hidden_size = args.dec_hidden_size, dropout = args.dropout)
decoder = LuongDecoder(vocab_size = len(vocab_cn), embed_size = args.emb_size, enc_hidden_size = args.enc_hidden_size,
                       dec_hidden_size = args.dec_hidden_size, dropout = args.dropout)
# decoder = PlainDecoder(vocab_size = len(zh_wtoi), embed_size = EMBED_SIZE, enc_hidden_size = ENC_HIDDEN_SIZE,
#                        dec_hidden_size = DEC_HIDDEN_SIZE, dropout = DROPOUT)
model = seq2seq(encoder, decoder)

# device_ids = [0, 1]
# model = torch.nn.DataParallel(model, device_ids = device_ids)
# 使用分布式训练
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index = args.padding_idx)  # 忽略padding位置的损失
optimizer = torch.optim.Adam(model.parameters())

# 画句子长度直方图
# bar = tqdm(train_data)
# en = torch.tensor([])
# cn = torch.tensor([])
# counter = 0
# for batch in bar:
#     en = torch.cat([en, batch["en_len"]])
#     cn = torch.cat([cn, batch["cn_len"]])
#     if counter == 20000:
#         break

# 看句子最长有多长
# max_length = 0
# bar = tqdm(train_data)
# for batch in bar:
#     if bar.last_print_n < 10000:
#         continue
#     x_lengths = batch["en_len"].to(DEVICE)
#     y_lengths = (batch["cn_len"] - 1).to(DEVICE)
#     length = max(x_lengths.max().item(), y_lengths.max().item())
#     if max_length < length:
#         max_length = length
# print(max_length)

# 训练流程


for epoch in range(1, args.epoch + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, train_data, loss_fn, args)
    end_time = timer()
    val_loss = evaluate(model, dev_data, loss_fn, args)
    print("Epoch: {}, Train loss: {:.3f}, Val loss: {:.3f}, Epoch time = {:.3f}s".format(epoch, train_loss, val_loss,
                                                                                         (end_time - start_time)))

# def translate_dev(i):
#     model.eval()
#
#     en_sent = " ".join([en_itow[word] for word in test_en_encode[i]])
#     print('英文原句：', en_sent)
#     print('标准中文翻译：', " ".join([zh_itow[word] for word in test_zh_encode[i]]))
#
#     bos = torch.Tensor([[zh_wtoi["BOS"]]]).long().to(DEVICE)
#     x = torch.Tensor(test_en_encode[i]).long().to(DEVICE).reshape(1, -1)
#     x_len = torch.Tensor([len(test_en_encode[i])]).long().to(DEVICE)
#
#     translation, _ = model.translate(x, x_len, bos)
#     translation = [zh_itow[i] for i in translation.data.cpu().numpy().reshape(-1)]
#
#     trans = []
#     for word in translation:
#         if word != "EOS":
#             trans.append(word)
#         else:
#             break
#     print('模型翻译结果：', " ".join(trans))

# for i in range(50, 100):
#     translate_dev(i)
#     print()
