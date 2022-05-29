## 运行

```shell
cd ~/autodl-tmp/attention/nyz/
#python -m torch.distributed.launch --nproc_per_node=2 main.py --vocab_size 50000 --batch_size 32 --truncate_size 128 --device cuda
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --vocab_size 50000 --batch_size 32 --truncate_size 128 --device cuda
```

参数量
Transformer 词向量 embedding*vocab_size attention embedding*(heads+1) 全连接层 2*embedding*feedforward

import opencc
cc = opencc.OpenCC('t2s')