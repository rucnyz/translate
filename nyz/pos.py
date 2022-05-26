# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 23:47
# @Author  : nieyuzhou
# @File    : pos.py
# @Software: PyCharm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(dpi = 200)
d_model = 128
max_len = 128
pos_table = np.array([[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
                      if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
# 字嵌入维度为偶数时
pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
# 字嵌入维度为奇数时
sns.heatmap(pos_table, cmap = "YlGnBu_r", vmin = -1, vmax = 1)
plt.show()
