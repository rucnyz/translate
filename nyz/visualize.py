# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 21:35
# @Author  : nieyuzhou
# @File    : visualize.py
# @Software: PyCharm
import matplotlib.pyplot as plt

hmodel = 64
feed = 2048
vocab = 50000
head = 8
layers = 12
p = (hmodel * vocab + (head + 1) * hmodel + 2 * hmodel * feed) * 12 / 1000000

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

x = [83.1, 62.3, 41.5, 15.3, 15.2]
y = [0.7, 0.64, 0.62, 0.41, 0.39]
label = ["Transformer 128", "Transformer 96", "Transformer 64", "Seq2Seq+Attn(Luong)", "Seq2Seq"]
plt.figure(dpi = 200)
scatter = plt.scatter(x, y, s = 140, marker = "*", c = range(1, len(label) + 1))
a, b = scatter.legend_elements()

plt.legend(a, label, loc = "lower right", title = "Models")

plt.grid()
plt.xlim([0,100])
plt.ylim([0.3,0.9])
plt.xlabel("参数量 （百万）")
plt.ylabel("Bleu分数")
# plt.show()
plt.savefig("./vis/vis_para.png", dpi = 300, bbox_inches = 'tight', transparent = True)
print("绘制完成")
# sns.heatmap(attn[1].cpu().detach().numpy()[:15,:13],xticklabels=['<bos>', 'adopted', 'by', 'the', 'security', 'council', 'at', 'its', '<unk>', 'meeting', ',', 'on', '17', 'may', '1994'],yticklabels=['1994', '年', '5', '月', '17', '日', '安全', '理事会', '第', '<unk>', '次', '会议', '通过'])