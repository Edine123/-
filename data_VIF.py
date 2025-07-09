import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# 创建输出目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv('data.csv')

# 获取数值型变量（排除id列）
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['id']]

# 剔除非显著变量
non_significant_vars = ['smoothness_se', 'texture_se', 'fractal_dimension_mean']
filtered_columns = [col for col in numeric_columns if col not in non_significant_vars]

# 准备数据并添加截距项
X = df[filtered_columns].copy()
X = add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data['变量'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# 按VIF值降序排序
vif_data = vif_data.sort_values('VIF', ascending=False)

# 保存结果
vif_data.to_csv(os.path.join(output_dir, 'vif_values_filtered.csv'), index=False, encoding='utf-8-sig')

# 打印结果
print("\n各变量的VIF值（剔除非显著变量后，包含截距项）：")
print(vif_data.to_string(index=False))
print("\nVIF值大于10的变量可能存在严重的多重共线性问题")
