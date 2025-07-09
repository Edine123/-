import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from data_show import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv('data.csv')

# 检查是否存在缺失值
has_missing = df.isnull().any().any()

if has_missing:
    print("数据中存在缺失值")
else:
    print("数据中没有缺失值")

# 检查每列的缺失值数量
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print(missing_percentage)

# 获取所有数值型列（排除id列和diagnosis列）
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['id']]

# 生成数据描述性统计,生成各种各样的水箱图、直方图、柱状图和散点图
import picture

# 定义变量组
variable_groups = {
    'radius': ['radius_mean', 'radius_se', 'radius_worst'],
    'texture': ['texture_mean', 'texture_se', 'texture_worst'],
    'perimeter': ['perimeter_mean', 'perimeter_se', 'perimeter_worst'],
    'area': ['area_mean', 'area_se', 'area_worst'],
    'smoothness': ['smoothness_mean', 'smoothness_se', 'smoothness_worst'],
    'compactness': ['compactness_mean', 'compactness_se', 'compactness_worst'],
    'concavity': ['concavity_mean', 'concavity_se', 'concavity_worst'],
    'concave_points': ['concave points_mean', 'concave points_se', 'concave points_worst'],
    'symmetry': ['symmetry_mean', 'symmetry_se', 'symmetry_worst'],
    'fractal_dimension': ['fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst']
}


# 进行良恶性变量比较分析
print("\n进行良恶性变量比较分析...")
compare_results = compare_variables(df)

# 进行详细的良恶性差异分析
print("\n进行详细的良恶性差异分析...")
diff_analysis = analyze_group_differences(df)

# 进行变量相关性分析（计算皮尔孙相关系数>0.75，计算并比较vif值）
print("\n进行变量相关性分析...")
pearson_corr, spearman_corr, high_corr = analyze_correlations(df)

import graph #绘制变量间的关系图
import aoye2 #绘制变量间的关系图

# 进行单变量显著性检验
print("\n进行单变量显著性检验...")
univariate_results = perform_univariate_tests(df)

# 验证单变量检验结果
if univariate_results is not None:
    print("\n单变量检验完成，结果已保存到 output/univariate_test_results.csv")
    print(f"发现 {len(univariate_results[univariate_results['是否显著'] == '是'])} 个显著相关变量")
else:
    print("\n警告：单变量检验未能完成，请检查数据或错误信息")


# 进行变量相关性分析
print("\n进行变量相关性分析（剔除非显著变量后）...")
pearson_corr, spearman_corr, high_corr = analyze_correlations(df)

# 进行单变量显著性检验
print("\n进行单变量显著性检验...")
univariate_results = perform_univariate_tests(df)

# 验证单变量检验结果
if univariate_results is not None:
    print("\n单变量检验完成，结果已保存到 output/univariate_test_results.csv")
    print(f"发现 {len(univariate_results[univariate_results['是否显著'] == '是'])} 个显著相关变量")
else:
    print("\n警告：单变量检验未能完成，请检查数据或错误信息")

#计算并显示VIF值（剔除非显著变量后）

print("\n计算变量的方差膨胀因子（VIF）（剔除非显著变量后）...")
vif_data = calculate_vif(df, numeric_columns)
if vif_data is not None:
    print("\n各变量的VIF值：")
    print(vif_data.to_string(index=False))
    print("\nVIF值大于10的变量可能存在严重的多重共线性问题")
else:
    print("\n警告：VIF计算未能完成，请检查数据或错误信息")

# 数据标准化处理
print("\n正在进行数据标准化处理...")
df_standardized = standardize_data(df, output_dir)

#手动剔除后剩余的特征列表
features_list = ["texture_worst","texture_mean","compactness_se", "symmetry_worst","fractal_dimension_worst","smoothness_worst",
    "fractal_dimension_se", "smoothness_mean", "area_mean","concavity_se","concave points_se","area_se","symmetry_mean","symmetry_se"]

# 对每一个变量进行逻辑回归分析
print("\n进行逻辑回归分析...")
logistic_regression_results = perform_logistic_regression_analysis(df_standardized, features_list, output_dir)

# 进行基准逻辑回归分析（空白对照）
print("\n进行基准逻辑回归分析...")
baseline_results = perform_baseline_logistic_regression(df_standardized, output_dir)



# 进行多变量逻辑回归分析
print("\n进行多变量逻辑回归分析...")
multivariate_results = perform_multivariate_logistic_regression(df_standardized, features_list, output_dir)


AIC_list =[
    "compactness_worst","concave points_worst","area_worst","concavity_worst","area_mean","compactness_mean",
    "compactness_se","fractal_dimension_worst","concavity_se","texture_worst","fractal_dimension_se", "texture_mean",
    "concave points_se","symmetry_worst","smoothness_mean","smoothness_worst","area_se","symmetry_mean","symmetry_se"
]

# 进行逐步回归分析
print("\n进行逐步回归分析...")
stepwise_regression_analysis(df_standardized, AIC_list, 'diagnosis', output_dir, initial_bidirectional_features=features_list)



# 进行L1正则化逻辑回归分析
print("\n进行L1正则化逻辑回归分析（皮尔森系数<0.2）...")
l1_results = perform_l1_logistic_regression(df_standardized, numeric_columns, output_dir)

features_list_L1_2 = ['texture_mean', 'texture_se', 'texture_worst','perimeter_mean', 'perimeter_se', 'perimeter_worst','area_mean', 'area_se', 
                 'area_worst','concavity_mean', 'concavity_se', 'concavity_worst', 'symmetry_mean', 'symmetry_se', 'symmetry_worst']

# 进行L1正则化逻辑回归分析（使用指定的特征列表）
print("\n进行L1正则化逻辑回归分析（使用指定的特征列表）...")
l1_results_2 = perform_l1_logistic_regression_with_features(df_standardized, features_list_L1_2, output_dir)

# 进行标准逻辑回归分析（无正则化）
print("\n进行标准逻辑回归分析（无正则化）...")
standard_results = perform_standard_logistic_regression(df_standardized, numeric_columns, output_dir)


# 进行AIC向后回归的特征重要性分析
# 准备数据
X = df_standardized[AIC_list]
y = df_standardized['diagnosis']

# 添加常数项
X = sm.add_constant(X)

# 初始模型
model = sm.Logit(y, X)
results = model.fit()

current_features = AIC_list.copy()
best_aic = results.aic
best_model = results
removed_features = []

while len(current_features) > 1:
    current_aic = float('inf')
    feature_to_remove = None
    
    # 尝试移除每个特征
    for feature in current_features:
        temp_features = [f for f in current_features if f != feature]
        X_temp = sm.add_constant(df_standardized[temp_features])
        model_temp = sm.Logit(y, X_temp)
        results_temp = model_temp.fit()
        
        if results_temp.aic < current_aic:
            current_aic = results_temp.aic
            feature_to_remove = feature
    
    # 如果移除特征后AIC没有改善，则停止
    if current_aic >= best_aic:
        break
    
    # 更新最佳模型
    best_aic = current_aic
    current_features.remove(feature_to_remove)
    removed_features.append(feature_to_remove)
    
    # 重新拟合模型
    X = sm.add_constant(df_standardized[current_features])
    best_model = sm.Logit(y, X).fit()

# 输出最终模型结果
print("\nAIC向后回归分析结果：")
print(f"最终保留的特征：{current_features}")
print(f"移除的特征：{removed_features}")
print(f"最终AIC值：{best_aic:.4f}")

# 特征重要性分析
print("\n进行特征重要性分析...")

# 计算特征权重和显著性
conf_int = best_model.conf_int()
params = best_model.params[1:]  # 排除常数项
pvalues = best_model.pvalues[1:]
conf_int = conf_int.iloc[1:]  # 排除常数项的置信区间

feature_importance = pd.DataFrame({
    '特征': current_features,
    '系数': params,
    'P值': pvalues,
    'OR值': np.exp(params),
    '95%置信区间下限': np.exp(conf_int.iloc[:, 0]),
    '95%置信区间上限': np.exp(conf_int.iloc[:, 1])
})

# 按系数绝对值排序
feature_importance['系数绝对值'] = abs(feature_importance['系数'])
feature_importance = feature_importance.sort_values('系数绝对值', ascending=False)

# 保存特征重要性结果
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False, encoding='utf-8')

# 输出特征重要性分析结果
print("\n特征重要性分析结果：")
print(feature_importance[['特征', '系数', 'P值', 'OR值', '95%置信区间下限', '95%置信区间上限']])

# 进行ANOVA分析
print("\n进行ANOVA分析...")
anova_results = []
for feature in current_features:
    # 对每个特征进行ANOVA分析
    benign_data = df_standardized[df_standardized['diagnosis'] == 0][feature]
    malignant_data = df_standardized[df_standardized['diagnosis'] == 1][feature]
    
    # 计算F统计量和P值
    f_stat, p_val = stats.f_oneway(benign_data, malignant_data)
    
    # 计算效应量 (Eta-squared)
    ss_total = np.sum((df_standardized[feature] - df_standardized[feature].mean()) ** 2)
    ss_between = len(benign_data) * (benign_data.mean() - df_standardized[feature].mean()) ** 2 + \
                 len(malignant_data) * (malignant_data.mean() - df_standardized[feature].mean()) ** 2
    eta_squared = ss_between / ss_total
    
    # 计算Cohen's d效应量
    cohens_d = (malignant_data.mean() - benign_data.mean()) / np.sqrt((malignant_data.var() + benign_data.var()) / 2)
    
    # 计算描述性统计量
    benign_mean = benign_data.mean()
    malignant_mean = malignant_data.mean()
    benign_std = benign_data.std()
    malignant_std = malignant_data.std()
    
    # 计算95%置信区间
    benign_ci = stats.t.interval(0.95, len(benign_data)-1, 
                                loc=benign_mean, 
                                scale=benign_std/np.sqrt(len(benign_data)))
    malignant_ci = stats.t.interval(0.95, len(malignant_data)-1, 
                                   loc=malignant_mean, 
                                   scale=malignant_std/np.sqrt(len(malignant_data)))
    
    anova_results.append({
        '特征': feature,
        'F统计量': f_stat,
        'P值': p_val,
        'Eta-squared': eta_squared,
        "Cohen's d": cohens_d,
        '良性均值': benign_mean,
        '良性标准差': benign_std,
        '恶性均值': malignant_mean,
        '恶性标准差': malignant_std,
        '良性95%置信区间下限': benign_ci[0],
        '良性95%置信区间上限': benign_ci[1],
        '恶性95%置信区间下限': malignant_ci[0],
        '恶性95%置信区间上限': malignant_ci[1],
        '均值差异': malignant_mean - benign_mean,
        '效应量解释': '大' if abs(cohens_d) > 0.8 else '中' if abs(cohens_d) > 0.5 else '小'
    })

anova_df = pd.DataFrame(anova_results)
# 按F统计量排序
anova_df = anova_df.sort_values('F统计量', ascending=False)
anova_df.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False, encoding='utf-8')

print("\nANOVA分析结果：")
print(anova_df[['特征', 'F统计量', 'P值', 'Eta-squared', "Cohen's d", '效应量解释', '均值差异']])

# 输出详细的ANOVA分析报告
print("\n详细的ANOVA分析报告：")
for _, row in anova_df.iterrows():
    print(f"\n{row['特征']}:")
    print(f"F统计量: {row['F统计量']:.4f}")
    print(f"P值: {row['P值']:.4f}")
    print(f"Eta-squared: {row['Eta-squared']:.4f}")
    print(f"Cohen's d: {row['Cohen\'s d']:.4f}")
    print(f"效应量解释: {row['效应量解释']}")
    print(f"良性组: 均值 = {row['良性均值']:.4f}, 标准差 = {row['良性标准差']:.4f}")
    print(f"恶性组: 均值 = {row['恶性均值']:.4f}, 标准差 = {row['恶性标准差']:.4f}")
    print(f"均值差异: {row['均值差异']:.4f}")
    print(f"良性组95%置信区间: [{row['良性95%置信区间下限']:.4f}, {row['良性95%置信区间上限']:.4f}]")
    print(f"恶性组95%置信区间: [{row['恶性95%置信区间下限']:.4f}, {row['恶性95%置信区间上限']:.4f}]")

# 绘制特征重要性可视化
plt.figure(figsize=(12, 6))
sns.barplot(x='系数绝对值', y='特征', data=feature_importance)
plt.title('特征重要性排序')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# 输出特征解释
print("\n特征解释：")
for _, row in feature_importance.iterrows():
    feature = row['特征']
    coef = row['系数']
    or_value = row['OR值']
    p_value = row['P值']
    
    print(f"\n{feature}:")
    print(f"系数: {coef:.4f}")
    print(f"OR值: {or_value:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        if coef > 0:
            print(f"该特征与恶性肿瘤呈正相关，OR值大于1，表明该特征值增加会增加恶性肿瘤的风险")
        else:
            print(f"该特征与恶性肿瘤呈负相关，OR值小于1，表明该特征值增加会降低恶性肿瘤的风险")
    else:
        print("该特征对诊断结果的影响不具有统计学显著性")