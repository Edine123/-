import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 读取数据，修正编码问题
try:
    df = pd.read_csv(r"output/high_correlations_filtered.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(r"output/high_correlations_filtered.csv")  # 尝试默认编码

# 清理数据中的 inf/nan
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
after = len(df)
if before != after:
    print(f"[警告] 有 {before-after} 行包含 NaN 或 inf 已被删除")

# 创建图对象
G = nx.Graph()

# 添加边
for _, row in df.iterrows():
    G.add_edge(row['变量A'], row['变量B'])

# 设置图形样式
plt.figure(figsize=(16, 12))
plt.style.use('ggplot')

# 使用圆形布局替代平面布局
pos = nx.circular_layout(G)

# 计算节点度数
degrees = dict(G.degree())
max_degree = max(degrees.values())

# 绘制图形
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, width=1.2, edge_color='gray', alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=10)

# 添加标题
plt.title('Variable Relationship Graph (Circular Layout)', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()