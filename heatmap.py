#-*- coding = utf-8 -*-
#@Time :2025/6/18 14:30
#@Author :郎皓东
#@File ：heatmap.py
#@Software:PyCharm

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
all_metrics = pd.read_excel("./data/heatmap.xlsx",index_col=0)
# 绘制热图
plt.figure(figsize=(8, 4))
sns.heatmap(all_metrics,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5)

plt.title("All Models Evaluation Metrics")
plt.tight_layout()
# plt.savefig("LR_metrics_heatmap.png")
plt.show()
