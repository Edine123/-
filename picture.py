# ======全局直方图=====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'data.csv') 

# 设置图形大小和布局（调整高度比例）
plt.figure(figsize=(20, 25))  # 减小高度以防止空白
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义要绘制的特征组
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# 计数器用于跟踪实际绘制的子图
plot_counter = 1

# 遍历每个特征组
for i, feature in enumerate(features, 1):
    # 获取该特征对应的三个变量
    cols = [f"{feature}_mean", f"{feature}_se", f"{feature}_worst"]
    
    # 检查这些列是否存在于数据中
    cols = [col for col in cols if col in data.columns]
    if not cols:
        continue
    
    # 创建子图
    for j, col in enumerate(cols, 1):
        # 使用更直接的子图定位方式
        ax = plt.subplot(10, 3, plot_counter)
        plot_counter += 1
        
        # 绘制直方图
        sns.histplot(data[col], kde=False, stat='density', color='skyblue', alpha=0.7, ax=ax)
        
        # 绘制PDF曲线
        sns.kdeplot(data[col], color='red', linewidth=2, ax=ax)
        
        # 计算并显示取值范围
        min_val = data[col].min()
        max_val = data[col].max()
        ax.text(0.95, 0.95, f"Range: [{min_val:.2f}, {max_val:.2f}]", 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Density')

plt.suptitle('Distribution of Numerical Features with PDF Curves', y=1.02, fontsize=16)

# 保存图片
plt.savefig('breast_cancer_features_distribution.png', dpi=300, bbox_inches='tight')
print("图片已保存为 breast_cancer_features_distribution.png")

plt.show()

# 描述性统计
desc_stats = data.describe().transpose()
print(desc_stats)

# =====良恶性分层直方图=====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'data.csv')  

diagnosis_col = 'diagnosis'  # 请根据实际数据修改

# 设置图形大小和布局
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义要绘制的特征组
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# 计数器用于跟踪实际绘制的子图
plot_counter = 1

# 遍历每个特征组
for i, feature in enumerate(features, 1):
    # 获取该特征对应的三个变量
    cols = [f"{feature}_mean", f"{feature}_se", f"{feature}_worst"]
    
    # 检查这些列是否存在于数据中
    cols = [col for col in cols if col in data.columns]
    if not cols:
        continue
    
    # 创建子图
    for j, col in enumerate(cols, 1):
        ax = plt.subplot(10, 3, plot_counter)
        plot_counter += 1
        
        # 分离良性和恶性数据
        benign = data[data[diagnosis_col] == 'B'][col]
        malignant = data[data[diagnosis_col] == 'M'][col]
        
        # 绘制直方图（叠加）
        sns.histplot(benign, kde=False, stat='density', color='skyblue', alpha=0.7, 
                    label='Benign', ax=ax)
        sns.histplot(malignant, kde=False, stat='density', color='salmon', alpha=0.7, 
                    label='Malignant', ax=ax)
        
        # 绘制PDF曲线（叠加）
        sns.kdeplot(benign, color='blue', linewidth=1.5, ax=ax)
        sns.kdeplot(malignant, color='red', linewidth=1.5, ax=ax)
        
        # 计算并显示取值范围
        min_benign, max_benign = benign.min(), benign.max()
        min_malignant, max_malignant = malignant.min(), malignant.max()
        
        text_str = f"B: [{min_benign:.2f}, {max_benign:.2f}]\nM: [{min_malignant:.2f}, {max_malignant:.2f}]"
        
        ax.text(0.95, 0.50, text_str, transform=ax.transAxes, 
                ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        ax.legend()

plt.suptitle('Distribution of Numerical Features (Benign vs Malignant)', y=1.02, fontsize=16)

# 保存图片
plt.savefig('breast_cancer_features_benign_malignant.png', dpi=300, bbox_inches='tight')
print("图片已保存为 breast_cancer_features_benign_malignant.png")

plt.show()

# =====良恶性分层箱线图=====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'data.csv')  # 请替换为您的文件路径

# 假设诊断结果列名为'diagnosis'，B=良性，M=恶性
diagnosis_col = 'diagnosis'  # 请根据实际数据修改

# 设置图形大小和布局
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义要绘制的特征组
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# 计数器用于跟踪实际绘制的子图
plot_counter = 1

# 遍历每个特征组
for i, feature in enumerate(features, 1):
    # 获取该特征对应的三个变量
    cols = [f"{feature}_mean", f"{feature}_se", f"{feature}_worst"]
    
    # 检查这些列是否存在于数据中
    cols = [col for col in cols if col in data.columns]
    if not cols:
        continue
    
    # 创建子图
    for j, col in enumerate(cols, 1):
        ax = plt.subplot(10, 3, plot_counter)
        plot_counter += 1
        
        # 绘制箱线图（按良恶性分组）
        sns.boxplot(x=diagnosis_col, y=col, data=data, 
                   palette={'B': 'skyblue', 'M': 'salmon'},
                   width=0.5, ax=ax)
        
        # 计算并显示取值范围
        benign = data[data[diagnosis_col] == 'B'][col]
        malignant = data[data[diagnosis_col] == 'M'][col]
        
        min_benign, max_benign = benign.min(), benign.max()
        min_malignant, max_malignant = malignant.min(), malignant.max()
        
        text_str = f"B: [{min_benign:.2f}, {max_benign:.2f}]\nM: [{min_malignant:.2f}, {max_malignant:.2f}]"
        
        # 调整文本位置（y=0.85向下移动）
        ax.text(0.95, 0.85, text_str, transform=ax.transAxes, 
                ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(col)
        ax.set_ylabel('Value')

plt.suptitle('Boxplot of Numerical Features (Benign vs Malignant)', y=1.02, fontsize=16)

# 保存图片
plt.savefig('breast_cancer_features_boxplot.png', dpi=300, bbox_inches='tight')
print("图片已保存为 breast_cancer_features_boxplot.png")

#=====全局箱线图=====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'data.csv')  

# 设置图形大小和布局
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义要绘制的特征组
features = ['cadius', 'texture', 'perimeter', 'area', 'smoothness', 
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# 计数器用于跟踪实际绘制的子图
plot_counter = 1

# 遍历每个特征组
for i, feature in enumerate(features, 1):
    # 获取该特征对应的三个变量
    cols = [f"{feature}_mean", f"{feature}_se", f"{feature}_worst"]
    
    # 检查这些列是否存在于数据中
    cols = [col for col in cols if col in data.columns]
    if not cols:
        continue
    
    # 创建子图
    for j, col in enumerate(cols, 1):
        ax = plt.subplot(10, 3, plot_counter)
        plot_counter += 1
        
        # 绘制箱线图（整体数据，不分组）
        sns.boxplot(y=data[col], color='skyblue', width=0.5, ax=ax)
        
        # 计算并显示取值范围
        min_val = data[col].min()
        max_val = data[col].max()
        
        ax.text(0.95, 0.85, f"Range: [{min_val:.2f}, {max_val:.2f}]", 
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Value')

plt.suptitle('Boxplot of Numerical Features (Overall Data)', y=1.02, fontsize=16)

# 保存图片
plt.savefig('breast_cancer_features_boxplot_overall.png', dpi=300, bbox_inches='tight')
print("图片已保存为 breast_cancer_features_boxplot_overall.png")

#=====柱状图=====
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv(r'data.csv')  
diagnosis_col = 'diagnosis'  # 请确认列名是否正确

# 统计良恶性数量和比例
counts = data[diagnosis_col].value_counts()
percentages = data[diagnosis_col].value_counts(normalize=True) * 100

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制数量柱状图
bars = plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'], width=0.6)

# 添加数量标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12)

# 添加百分比标签（显示在柱子上方）
for i, (idx, val) in enumerate(counts.items()):
    plt.text(i, val + max(counts)*0.05, 
             f'{percentages[idx]:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 美化图形
plt.title('Breast Cancer Diagnosis Distribution', fontsize=14, pad=20)
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Benign (B)', 'Malignant (M)'], fontsize=12)
plt.ylim(0, max(counts.values) * 1.2)  # 留出标签空间

# 显示网格线
plt.grid(axis='y', alpha=0.3)

# 保存图片
plt.savefig('diagnosis_distribution.png', dpi=300, bbox_inches='tight')
print("图片已保存为 diagnosis_distribution.png")

# =====散点图=====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv(r'data.csv')  # 请替换为您的文件路径
diagnosis_col = 'diagnosis'  # 诊断结果列名（B=良性，M=恶性）

# 设置图形大小和布局
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义要绘制的特征组
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
            'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# 生成随机x坐标（避免点重叠）
np.random.seed(42)
data['jitter'] = np.random.normal(0, 0.05, size=len(data))

# 计数器用于跟踪实际绘制的子图
plot_counter = 1

# 遍历每个特征组
for i, feature in enumerate(features, 1):
    cols = [f"{feature}_mean", f"{feature}_se", f"{feature}_worst"]
    cols = [col for col in cols if col in data.columns]
    
    for j, col in enumerate(cols, 1):
        ax = plt.subplot(10, 3, plot_counter)
        plot_counter += 1
        
        # 绘制散点图（按良恶性分组）
        scatter = sns.scatterplot(x='jitter', y=col, hue=diagnosis_col, data=data,
                                 palette={'B': 'skyblue', 'M': 'salmon'},
                                 alpha=0.7, ax=ax, legend=False)  # 先关闭默认图例
        
        # 手动添加图例到子图内部
        ax.plot([], [], 'o', color='skyblue', label='B')
        ax.plot([], [], 'o', color='salmon', label='M')
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
        
        # 美化图形
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.get_xaxis().set_visible(False)  # 隐藏x轴

plt.suptitle('Scatter Plot of Numerical Features (Benign vs Malignant)', y=1.02, fontsize=16)
plt.savefig('breast_cancer_features_scatter_internal_legend.png', dpi=300, bbox_inches='tight')
