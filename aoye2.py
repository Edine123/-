import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
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

# 兼容不同相关系数列名
if 'Pearson相关系数' in df.columns:
    corr_col = 'Pearson相关系数'
elif '相关系数' in df.columns:
    corr_col = '相关系数'
else:
    corr_col = None

# 创建图对象
G = nx.Graph()
for _, row in df.iterrows():
    if corr_col:
        G.add_edge(row['变量A'], row['变量B'], weight=row[corr_col])
    else:
        G.add_edge(row['变量A'], row['变量B'])

# 函数：判断变量是否包含指定关键词
def is_geo_var(name):
    return any(keyword in name.lower() for keyword in ['radius', 'perimeter', 'area'])

# 分割图形
geo_nodes = [n for n in G.nodes() if is_geo_var(n)]
other_nodes = [n for n in G.nodes() if not is_geo_var(n)]

geo_graph = G.subgraph(geo_nodes)
other_graph = G.subgraph(other_nodes)

# 设置图形样式
plt.style.use('ggplot')

# 1. 绘制几何特征图
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(geo_graph, k=0.8, seed=42)

# 绘制节点和边
nx.draw_networkx_nodes(geo_graph, pos, node_size=1200, node_color='#66b3ff', alpha=0.9)
nx.draw_networkx_edges(geo_graph, pos, width=1.5, edge_color='#888888', alpha=0.7)
nx.draw_networkx_labels(geo_graph, pos, font_size=10, font_weight='bold')

plt.title('Geometric Features Relationship (radius/perimeter/area)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# 2. 绘制其他特征图
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(other_graph, k=0.8, seed=42)

# 按节点类型着色
node_colors = []
for node in other_graph.nodes():
    if 'texture' in node.lower():
        node_colors.append('#ff9999')  # 粉色
    elif 'concav' in node.lower():
        node_colors.append('#99ff99')  # 绿色
    elif 'compact' in node.lower():
        node_colors.append('#ffcc99')  # 橙色
    else:
        node_colors.append('#c2c2f0')  # 紫色

nx.draw_networkx_nodes(other_graph, pos, node_size=1200, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(other_graph, pos, width=1.5, edge_color='#888888', alpha=0.7)
nx.draw_networkx_labels(other_graph, pos, font_size=10, font_weight='bold')

plt.title('Other Features Relationship', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()