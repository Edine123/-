import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, confusion_matrix
# from statsmodels.stats.api import gof_chisquare_test # For Hosmer-Lemeshow

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def describe_data(df):
    """
    生成数据的详细描述性统计信息
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    """
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
    
    # 基本数据集信息
    print("数据集基本信息：")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print("\n" + "="*80 + "\n")
    
    # 诊断结果分布
    print("诊断结果分布：")
    diagnosis_counts = df['diagnosis'].value_counts()
    diagnosis_percentages = (diagnosis_counts / len(df) * 100).round(2)
    for diagnosis, count in diagnosis_counts.items():
        print(f"{diagnosis}: {count} 例 ({diagnosis_percentages[diagnosis]}%)")
    print("\n" + "="*80 + "\n")
    
    # 数值型变量的描述性统计
    print("数值型变量的描述性统计：")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    
    # 计算每个变量的统计量
    stats_df = df[numeric_cols].describe().round(4)
    stats_df.loc['skew'] = df[numeric_cols].skew()
    stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
    
    # 按变量组组织统计信息
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
    
    # 创建统计量的中文映射
    stat_names = {
        'count': '计数',
        'mean': '均值',
        'std': '标准差',
        'min': '最小值',
        '25%': '25%分位数',
        '50%': '中位数',
        '75%': '75%分位数',
        'max': '最大值',
        'skew': '偏度',
        'kurtosis': '峰度'
    }
    
    for group_name, variables in variable_groups.items():
        print(f"\n{group_name.upper()} 特征组统计：")
        print("-" * 100)
        group_stats = stats_df[variables]
        # 重命名索引为中文
        group_stats.index = [stat_names[stat] for stat in group_stats.index]
        
        # 格式化输出
        print("\n{:<12} {:<15} {:<15} {:<15}".format("统计量", "mean", "se", "worst"))
        print("-" * 60)
        for stat in group_stats.index:
            values = group_stats.loc[stat]
            print("{:<12} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                stat, values[0], values[1], values[2]
            ))
        print("\n" + "="*100)
    
    # 相关性分析
    print("\n特征相关性分析：")
    correlation = df[numeric_cols].corr()
    
    # 找出相关性最强的特征对
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_pairs.append((
                numeric_cols[i],
                numeric_cols[j],
                correlation.loc[numeric_cols[i], numeric_cols[j]]
            ))
    
    # 按相关性绝对值排序
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n相关性最强的特征对（前10对）：")
    print("-" * 80)
    print("{:<30} {:<30} {:<10}".format("特征1", "特征2", "相关系数"))
    print("-" * 80)
    for var1, var2, corr in corr_pairs[:10]:
        print("{:<30} {:<30} {:<10.4f}".format(var1, var2, corr))
    
    return stats_df, correlation

def compare_variables(df, output_dir='output'):
    """
    比较良性和恶性之间的变量差异，包括箱线图可视化和统计检验
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    output_dir : str
        输出目录
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数值型变量（排除id列）
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    
    # 按变量组组织变量
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
    
    # 创建结果存储字典
    results = {}
    
    # 对每个变量组进行分析
    for group_name, variables in variable_groups.items():
        print(f"\n{group_name.upper()} 特征组分析：")
        print("="*100)
        
        # 创建3x1的子图布局
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'{group_name.upper()} 特征组箱线图比较', fontsize=16, y=0.95)
        
        # 对每个变量进行统计检验和可视化
        for idx, var in enumerate(variables):
            # 获取良恶性组的数据
            benign_data = df[df['diagnosis'] == 'B'][var]
            malignant_data = df[df['diagnosis'] == 'M'][var]
            
            # 进行正态性检验
            _, p_benign = stats.normaltest(benign_data)
            _, p_malignant = stats.normaltest(malignant_data)
            
            # 根据正态性检验结果选择合适的统计检验
            if p_benign > 0.05 and p_malignant > 0.05:
                # 两组都服从正态分布，使用t检验
                stat, p_value = stats.ttest_ind(benign_data, malignant_data)
                test_name = "t检验"
            else:
                # 至少一组不服从正态分布，使用Mann-Whitney U检验
                stat, p_value = stats.mannwhitneyu(benign_data, malignant_data, alternative='two-sided')
                test_name = "Mann-Whitney U检验"
            
            # 存储结果
            results[var] = {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'benign_mean': benign_data.mean(),
                'malignant_mean': malignant_data.mean(),
                'benign_std': benign_data.std(),
                'malignant_std': malignant_data.std()
            }
            
            # 绘制箱线图
            sns.boxplot(x='diagnosis', y=var, data=df, ax=axes[idx])
            axes[idx].set_title(f'{var} (p={p_value:.4f})')
            axes[idx].set_xlabel('诊断结果')
            axes[idx].set_ylabel(var)
            axes[idx].set_xticklabels(['良性', '恶性'])
            
            # 添加均值点
            sns.pointplot(x='diagnosis', y=var, data=df, color='red', ax=axes[idx])
        
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.4)
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'{group_name}_comparison.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 打印统计检验结果
        print("\n统计检验结果：")
        print("-"*80)
        print(f"{'变量':<30} {'检验方法':<20} {'p值':<10} {'良性均值':<10} {'恶性均值':<10}")
        print("-"*80)
        for var in variables:
            result = results[var]
            print(f"{var:<30} {result['test_name']:<20} {result['p_value']:<10.4f} "
                  f"{result['benign_mean']:<10.4f} {result['malignant_mean']:<10.4f}")
    
    return results

def analyze_group_differences(df, output_dir='output'):
    """
    分析良恶性之间的变量差异，生成详细报告
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    output_dir : str
        输出目录
    """
    try:
        # 设置pandas显示选项
        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数值型变量（排除id列）
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'id']
        
        # 创建结果存储列表
        results = []
        
        # 对每个变量进行分析
        for var in numeric_cols:
            # 获取良恶性组的数据
            benign_data = df[df['diagnosis'] == 'B'][var]
            malignant_data = df[df['diagnosis'] == 'M'][var]
            
            # 进行正态性检验
            _, p_benign = stats.normaltest(benign_data)
            _, p_malignant = stats.normaltest(malignant_data)
            
            # 根据正态性检验结果选择合适的统计检验
            if p_benign > 0.05 and p_malignant > 0.05:
                # 两组都服从正态分布，使用t检验
                stat, p_value = stats.ttest_ind(benign_data, malignant_data)
                test_name = "t检验"
            else:
                # 至少一组不服从正态分布，使用Mann-Whitney U检验
                stat, p_value = stats.mannwhitneyu(benign_data, malignant_data, alternative='two-sided')
                test_name = "Mann-Whitney U检验"
            
            # 计算效应量（Cohen's d）
            cohens_d = (malignant_data.mean() - benign_data.mean()) / np.sqrt(
                ((malignant_data.std() ** 2 + benign_data.std() ** 2) / 2)
            )
            
            # 计算差异百分比
            diff_percent = ((malignant_data.mean() - benign_data.mean()) / benign_data.mean()) * 100
            
            # 判断显著性水平
            if p_value < 0.05:
                significance = "是"
            
            else:
                significance = "否"
            
            # 存储结果
            results.append({
                '变量': var,
                '检验方法': test_name,
                'p值': p_value,
                '显著性': significance,
                '效应量': cohens_d,
                '差异百分比': diff_percent,
                '良性均值': benign_data.mean(),
                '恶性均值': malignant_data.mean(),
                '良性标准差': benign_data.std(),
                '恶性标准差': malignant_data.std(),
                '良性中位数': benign_data.median(),
                '恶性中位数': malignant_data.median()
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 按p值排序
        results_df = results_df.sort_values('p值')
        
        # 保存结果到CSV文件
        csv_path = os.path.join(output_dir, 'group_differences_analysis.csv')
        try:
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n分析结果已保存到: {csv_path}")
        except PermissionError:
            print(f"\n警告: 无法保存CSV文件到 {csv_path}，文件可能正在被其他程序使用")
            print("请关闭可能正在使用该文件的其他程序后重试")
        
        # 生成报告文本
        report = []
        report.append("="*100)
        report.append(" "*40 + "良恶性差异分析报告")
        report.append("="*100)
        report.append(f"\n分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("-"*100)
        
        # 样本信息
        report.append("\n一、样本信息")
        report.append("-"*100)
        report.append(f"样本总数: {len(df)}")
        report.append(f"良性样本数: {len(df[df['diagnosis'] == 'B'])}")
        report.append(f"恶性样本数: {len(df[df['diagnosis'] == 'M'])}")
        
        # 显著性差异的变量
        significant_vars = results_df[results_df['p值'] < 0.05]
        report.append(f"\n二、显著性差异分析")
        report.append("-"*100)
        report.append(f"发现 {len(significant_vars)} 个变量在良恶性组间存在显著差异 (p < 0.05)")
        
        # 差异最显著的10个变量
        report.append("\n三、差异最显著的10个变量")
        report.append("-"*100)
        report.append(f"{'变量':<30} {'检验方法':<15} {'p值':<10} {'显著性':<8} {'效应量':<10} {'差异百分比':<10}")
        report.append("-"*100)
        for _, row in significant_vars[['变量', '检验方法', 'p值', '显著性', '效应量', '差异百分比']].head(10).iterrows():
            report.append(f"{row['变量']:<30} {row['检验方法']:<15} {row['p值']:<10.4f} {row['显著性']:<8} {row['效应量']:<10.4f} {row['差异百分比']:<10.2f}%")
        
        # 效应量最大的10个变量
        report.append("\n四、效应量最大的10个变量")
        report.append("-"*100)
        report.append(f"{'变量':<30} {'检验方法':<15} {'p值':<10} {'显著性':<8} {'效应量':<10} {'差异百分比':<10}")
        report.append("-"*100)
        for _, row in significant_vars.sort_values('效应量', ascending=False)[
            ['变量', '检验方法', 'p值', '显著性', '效应量', '差异百分比']].head(10).iterrows():
            report.append(f"{row['变量']:<30} {row['检验方法']:<15} {row['p值']:<10.4f} {row['显著性']:<8} {row['效应量']:<10.4f} {row['差异百分比']:<10.2f}%")
        
        # 差异百分比最大的10个变量
        report.append("\n五、差异百分比最大的10个变量")
        report.append("-"*100)
        report.append(f"{'变量':<30} {'检验方法':<15} {'p值':<10} {'显著性':<8} {'效应量':<10} {'差异百分比':<10}")
        report.append("-"*100)
        for _, row in significant_vars.sort_values('差异百分比', ascending=False)[
            ['变量', '检验方法', 'p值', '显著性', '效应量', '差异百分比']].head(10).iterrows():
            report.append(f"{row['变量']:<30} {row['检验方法']:<15} {row['p值']:<10.4f} {row['显著性']:<8} {row['效应量']:<10.4f} {row['差异百分比']:<10.2f}%")
        
        # 添加说明
        report.append("\n六、说明")
        report.append("-"*100)
        report.append("1. 显著性标记说明：")
        report.append("   - 是:   p < 0.05  (显著)")
        report.append("   - 否:  p ≥ 0.05  (不显著)")
        report.append("2. 效应量（Cohen's d）表示差异的实际大小：")
        report.append("   - |d| < 0.2: 微小差异")
        report.append("   - 0.2 ≤ |d| < 0.5: 小差异")
        report.append("   - 0.5 ≤ |d| < 0.8: 中等差异")
        report.append("   - |d| ≥ 0.8: 大差异")
        report.append("3. 差异百分比表示恶性组相对于良性组的差异程度")
        
        # 保存报告到文本文件
        report_path = os.path.join(output_dir, 'group_differences_report.txt')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            print(f"分析报告已保存到: {report_path}")
        except PermissionError:
            print(f"\n警告: 无法保存报告文件到 {report_path}，文件可能正在被其他程序使用")
            print("请关闭可能正在使用该文件的其他程序后重试")
        
        # 打印报告
        print('\n'.join(report))
        
        return results_df
        
    except Exception as e:
        print(f"\n错误: 分析过程中出现异常: {str(e)}")
        return None

def calculate_vif(df, numeric_cols):
    """
    计算变量的方差膨胀因子（VIF）
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    numeric_cols : list
        要计算VIF的数值型变量列表
    
    Returns:
    --------
    DataFrame
        包含VIF值的DataFrame
    """
    try:
        # 剔除非显著变量
        non_significant_vars = ['smoothness_se', 'texture_se', 'fractal_dimension_mean']
        numeric_cols = [col for col in numeric_cols if col not in non_significant_vars]
        
        # 准备数据并添加截距项
        X = df[numeric_cols].copy()
        X = add_constant(X)
        
        # 计算VIF
        vif_data = pd.DataFrame()
        vif_data['变量'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        
        # 按VIF值降序排序
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        return vif_data
        
    except Exception as e:
        print(f"\n错误: VIF计算过程中出现异常: {str(e)}")
        return None

def analyze_correlations(df, output_dir='output'):
    """
    分析变量之间的相关性
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    output_dir : str
        输出目录
    
    Returns:
    --------
    tuple
        (pearson相关系数矩阵, spearman相关系数矩阵, 高相关变量对)
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数值型变量（排除id列）
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'id']
        
        # 剔除非显著变量
        non_significant_vars = ['smoothness_se', 'texture_se', 'fractal_dimension_mean']
        numeric_cols = [col for col in numeric_cols if col not in non_significant_vars]
        
        # 计算Pearson相关系数
        pearson_corr = df[numeric_cols].corr(method='pearson')
        
        # 计算Spearman相关系数
        spearman_corr = df[numeric_cols].corr(method='spearman')
        
        # 计算VIF值（包含截距项）
        X = df[numeric_cols].copy()
        X = add_constant(X)
        vif_values = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        vif_data = pd.DataFrame({'变量': X.columns, 'VIF': vif_values})
        
        # 找出高相关变量对（相关系数绝对值大于0.75）
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                var1 = numeric_cols[i]
                var2 = numeric_cols[j]
                corr = pearson_corr.loc[var1, var2]
                if abs(corr) > 0.75:  # 确保相关系数绝对值大于0.75
                    # 获取两个变量的VIF值
                    vif1 = vif_data[vif_data['变量'] == var1]['VIF'].values[0]
                    vif2 = vif_data[vif_data['变量'] == var2]['VIF'].values[0]
                    # 确定VIF值较大的变量
                    higher_vif_var = var1 if vif1 > vif2 else var2
                    high_corr_pairs.append({
                        '变量A': var1,
                        '变量B': var2,
                        '相关系数': corr,
                        '变量A的VIF': vif1,
                        '变量B的VIF': vif2,
                        'VIF较大的变量': higher_vif_var
                    })
        
        # 转换为DataFrame并排序
        high_corr_df = pd.DataFrame(high_corr_pairs)
        if not high_corr_df.empty:
            high_corr_df = high_corr_df.sort_values('相关系数', key=abs, ascending=False)  # 按相关系数绝对值排序
        
        # 保存结果
        pearson_corr.to_csv(os.path.join(output_dir, 'pearson_correlations_filtered.csv'), encoding='utf-8-sig')
        spearman_corr.to_csv(os.path.join(output_dir, 'spearman_correlations_filtered.csv'), encoding='utf-8-sig')
        high_corr_df.to_csv(os.path.join(output_dir, 'high_correlations_filtered.csv'), index=False, encoding='utf-8-sig')
        
        # 保存VIF值
        vif_data.to_csv(os.path.join(output_dir, 'vif_values_filtered.csv'), index=False, encoding='utf-8-sig')
        
        # 绘制热图
        plt.figure(figsize=(15, 12))
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('变量间Pearson相关系数热图（剔除非显著变量后）')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap_filtered.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印高相关变量对
        print("\n高相关变量对（相关系数绝对值 > 0.75）：")
        if not high_corr_df.empty:
            print(high_corr_df.to_string(index=False))
        else:
            print("没有发现相关系数绝对值大于0.75的变量对")
        
        # 打印VIF值
        print("\n各变量的VIF值：")
        print(vif_data.to_string(index=False))
        
        return pearson_corr, spearman_corr, high_corr_df
        
    except Exception as e:
        print(f"\n错误: 相关性分析过程中出现异常: {str(e)}")
        return None, None, None

def perform_univariate_tests(df, output_dir='output'):
    """
    对每个变量进行双样本t检验和Mann-Whitney U检验，评估与诊断结果的相关性
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    output_dir : str
        输出目录
    
    Returns:
    --------
    DataFrame
        包含检验结果的DataFrame
    """
    try:
        # 创建输出目录
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"\n警告: 创建输出目录失败: {str(e)}")
            return None
        
        # 获取数值型变量（排除id列）
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'id']
        
        # 存储检验结果
        results = []
        
        # 对每个变量进行检验
        for col in numeric_cols:
            try:
                # 获取良性和恶性样本的数据
                benign = df[df['diagnosis'] == 'B'][col]
                malignant = df[df['diagnosis'] == 'M'][col]
                
                # 进行Shapiro-Wilk正态性检验
                _, p_value_benign = stats.shapiro(benign)
                _, p_value_malignant = stats.shapiro(malignant)
                
                # 根据正态性检验结果选择合适的检验方法
                if p_value_benign > 0.05 and p_value_malignant > 0.05:
                    # 两组都服从正态分布，使用t检验
                    test_stat, p_value = stats.ttest_ind(benign, malignant)
                    test_name = "t检验"
                else:
                    # 至少一组不服从正态分布，使用Mann-Whitney U检验
                    test_stat, p_value = stats.mannwhitneyu(benign, malignant, alternative='two-sided')
                    test_name = "Mann-Whitney U检验"
                
                # 计算效应量（Cohen's d）
                benign_var = benign.var()
                malignant_var = malignant.var()
                if benign_var == 0 and malignant_var == 0:
                    cohens_d = 0  # 如果两组方差都为0，则效应量为0
                else:
                    pooled_std = np.sqrt((benign_var + malignant_var) / 2)
                    if pooled_std == 0:
                        cohens_d = 0  # 如果合并标准差为0，则效应量为0
                    else:
                        cohens_d = (malignant.mean() - benign.mean()) / pooled_std
                
                # 存储结果
                results.append({
                    '变量': col,
                    '检验方法': test_name,
                    '检验统计量': test_stat,
                    'P值': p_value,
                    '是否显著': '是' if p_value < 0.05 else '否',
                    '效应量(Cohen\'s d)': cohens_d,
                    '良性组均值': benign.mean(),
                    '恶性组均值': malignant.mean(),
                    '良性组标准差': benign.std(),
                    '恶性组标准差': malignant.std()
                })
            except Exception as e:
                print(f"处理变量 {col} 时出现错误: {str(e)}")
                continue
        
        if not results:
            print("没有成功处理任何变量")
            return None
            
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 按P值排序
        results_df = results_df.sort_values('P值')
        
        # 保存结果
        try:
            results_df.to_csv(os.path.join(output_dir, 'univariate_test_results.csv'), 
                            index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"\n警告: 保存检验结果失败: {str(e)}")
            print("继续执行其他分析...")
        
        # 打印显著相关的变量
        significant_vars = results_df[results_df['是否显著'] == '是']['变量'].tolist()
        print("\n与诊断结果显著相关的变量（P < 0.05）：")
        print(f"共发现 {len(significant_vars)} 个显著相关变量")
        print("\n详细检验结果：")
        print(results_df.to_string(index=False))
        
        # 绘制箱线图比较显著变量的分布
        if significant_vars:
            try:
                plt.figure(figsize=(20, 12))
                for i, var in enumerate(significant_vars[:9], 1):  # 只显示前9个变量
                    plt.subplot(3, 3, i)
                    sns.boxplot(x='diagnosis', y=var, data=df)
                    plt.title(f'{var}的分布')
                    plt.xlabel('诊断结果')
                    plt.ylabel(var)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'significant_variables_boxplots.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"\n警告: 保存箱线图失败: {str(e)}")
                print("继续执行其他分析...")
        
        return results_df
        
    except Exception as e:
        print(f"\n错误: 单变量检验过程中出现异常: {str(e)}")
        return None


def standardize_data(df, output_dir):
    """
    根据提供的标准化方法对数据进行处理，并将诊断结果Diagnosis变量转化为二值变量（M=1, B=0）。
    """
    df_processed = df.copy()

    # 诊断结果转化为二值变量
    df_processed['diagnosis'] = df_processed['diagnosis'].map({'M': 1, 'B': 0})

    # 定义标准化方法映射
    standardization_methods = {
        'radius_mean': 'Z-score', 'texture_mean': 'Z-score', 'perimeter_mean': 'Z-score', 'area_mean': 'Z-score',
        'smoothness_mean': 'Min-Max', 'compactness_mean': 'Robust', 'concavity_mean': 'Robust',
        'concave points_mean': 'Robust', 'symmetry_mean': 'Min-Max', 'fractal_dimension_mean': 'Min-Max',
        'radius_se': 'Z-score', 'texture_se': 'Z-score', 'perimeter_se': 'Z-score', 'area_se': 'Z-score',
        'smoothness_se': 'Min-Max', 'compactness_se': 'Robust', 'concavity_se': 'Robust',
        'concave points_se': 'Robust', 'symmetry_se': 'Min-Max', 'fractal_dimension_se': 'Min-Max',
        'radius_worst': 'Z-score', 'texture_worst': 'Z-score', 'perimeter_worst': 'Z-score', 'area_worst': 'Z-score',
        'smoothness_worst': 'Min-Max', 'compactness_worst': 'Robust', 'concavity_worst': 'Robust',
        'concave points_worst': 'Robust','symmetry_worst': 'Min-Max', 'fractal_dimension_worst': 'Min-Max'
    }

    # 应用标准化
    for col, method in standardization_methods.items():
        if col in df_processed.columns:
            if method == 'Z-score':
                scaler = StandardScaler()
            elif method == 'Min-Max':
                scaler = MinMaxScaler()
            elif method == 'Robust':
                scaler = RobustScaler()
            else:
                continue # Skip if method is not recognized

            df_processed[col] = scaler.fit_transform(df_processed[[col]])
        else:
            print(f"警告：列 '{col}' 不存在于数据中，跳过标准化。")

    # 保存处理后的数据到新的CSV文件
    output_path = os.path.join(output_dir, 'data_processed.csv')
    try:
        df_processed.to_csv(output_path, index=False, encoding='utf-8')
        print(f"处理后的数据已保存到 '{output_path}'")
    except Exception as e:
        print(f"保存处理后的数据失败: {e}")

    return df_processed

def perform_logistic_regression_analysis(df, features_list, output_dir):
    """
    对features_list中的每个变量逐一进行逻辑回归模型构建，并输出相关统计量。
    """
    print("\n--- 逻辑回归模型分析开始 ---")
    results_summary = []
    detailed_report_path = os.path.join(output_dir, 'logistic_regression_detailed_report.txt')

    # 确保诊断结果是数值类型
    df['diagnosis'] = df['diagnosis'].astype(int)

    with open(detailed_report_path, 'w', encoding='utf-8') as f_report:
        f_report.write("--- 逻辑回归模型分析报告 ---\n\n")

        for feature in features_list:
            print(f"\n分析变量: {feature}")
            f_report.write(f"分析变量: {feature}\n")

            if feature not in df.columns:
                print(f"警告: 列 '{feature}' 不存在于数据中，跳过。")
                f_report.write(f"警告: 列 '{feature}' 不存在于数据中，跳过。\n\n")
                continue

            try:
                # 添加常数项，用于statsmodels模型
                X = sm.add_constant(df[[feature]])
                y = df['diagnosis']

                # 构建逻辑回归模型
                model = sm.Logit(y, X)
                results = model.fit(disp=0) # disp=0 关闭迭代信息

                # 提取结果
                params = results.params
                conf_int = results.conf_int()
                p_values = results.pvalues

                # 回归系数、P值、OR值及其95%置信区间
                if 'const' in params.index and feature in params.index:
                    coefficient = params[feature]
                    p_value = p_values[feature]
                    or_value = np.exp(coefficient)
                    or_lower_ci = np.exp(conf_int.loc[feature, 0])
                    or_upper_ci = np.exp(conf_int.loc[feature, 1])

                    print(f"  回归系数 ({feature}): {coefficient:.4f}")
                    print(f"  P值 ({feature}): {p_value:.4f}")
                    print(f"  OR值 ({feature}): {or_value:.4f} (95% CI: {or_lower_ci:.4f} - {or_upper_ci:.4f})")

                    f_report.write(f"  回归系数 ({feature}): {coefficient:.4f}\n")
                    f_report.write(f"  P值 ({feature}): {p_value:.4f}\n")
                    f_report.write(f"  OR值 ({feature}): {or_value:.4f} (95% CI: {or_lower_ci:.4f} - {or_upper_ci:.4f})\n")

                    # Wald检验显著性判断
                    wald_significant = "是" if p_value < 0.05 else "否"
                    print(f"  Wald检验显著性 (α=0.05): {wald_significant}")
                    f_report.write(f"  Wald检验显著性 (α=0.05): {wald_significant}\n")

                    # 伪R² (McFadden's R-squared)
                    pseudo_r_squared = results.prsquared
                    print(f"  伪R² (McFadden's R-squared): {pseudo_r_squared:.4f}")
                    f_report.write(f"  伪R² (McFadden's R-squared): {pseudo_r_squared:.4f}\n")

                    # Hosmer-Lemeshow 检验
                    try:
                        y_pred_prob = results.predict(X)
                        df_temp = pd.DataFrame({'y_true': y, 'y_pred_prob': y_pred_prob})
                        df_temp['decile'] = pd.qcut(df_temp['y_pred_prob'], 10, labels=False, duplicates='drop')

                        observed_0 = df_temp.groupby('decile')['y_true'].apply(lambda x: (x == 0).sum())
                        observed_1 = df_temp.groupby('decile')['y_true'].apply(lambda x: (x == 1).sum())
                        expected_0 = df_temp.groupby('decile')['y_pred_prob'].apply(lambda x: (1 - x).sum())
                        expected_1 = df_temp.groupby('decile')['y_pred_prob'].apply(lambda x: x.sum())

                        hl_chi2 = 0
                        for i in range(len(observed_0)):
                            if expected_0.iloc[i] > 0 and observed_0.iloc[i] > 0:
                                hl_chi2 += (observed_0.iloc[i] - expected_0.iloc[i])**2 / expected_0.iloc[i]
                            if expected_1.iloc[i] > 0 and observed_1.iloc[i] > 0:
                                hl_chi2 += (observed_1.iloc[i] - expected_1.iloc[i])**2 / expected_1.iloc[i]

                        degrees_freedom = len(observed_0) - 2
                        hl_p_value = 1 - stats.chi2.cdf(hl_chi2, degrees_freedom)

                        print(f"  Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}")
                        print(f"  Hosmer-Lemeshow P值: {hl_p_value:.4f}")
                        print(f"  Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}")
                        f_report.write(f"  Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
                        f_report.write(f"  Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
                        f_report.write(f"  Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n")
                    except Exception as e:
                        print(f"  Hosmer-Lemeshow 检验计算失败: {e}")
                        f_report.write(f"  Hosmer-Lemeshow 检验计算失败: {e}\n")
                        hl_chi2, hl_p_value = np.nan, np.nan

                    results_summary.append({
                        '变量': feature,
                        '回归系数': coefficient,
                        'P值': p_value,
                        'OR值': or_value,
                        'OR 95% CI Lower': or_lower_ci,
                        'OR 95% CI Upper': or_upper_ci,
                        'Wald检验显著性': wald_significant,
                        '伪R²': pseudo_r_squared,
                        'Hosmer-Lemeshow Chi-squared': hl_chi2,
                        'Hosmer-Lemeshow P值': hl_p_value
                    })
                else:
                    print(f"  无法获取变量 {feature} 的回归结果。")
                    f_report.write(f"  无法获取变量 {feature} 的回归结果。\n")

            except Exception as e:
                print(f"对变量 {feature} 进行逻辑回归时出现错误: {e}")
                f_report.write(f"对变量 {feature} 进行逻辑回归时出现错误: {e}\n")
            f_report.write("\n") # Add a newline for separation between variables in the report

        f_report.write("--- 逻辑回归模型分析结束 ---\n")

    print("\n--- 逻辑回归模型分析结束 ---")

    # 保存汇总结果
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        output_path = os.path.join(output_dir, 'logistic_regression_results.csv')
        try:
            summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"逻辑回归汇总结果已保存到 '{output_path}'")
        except Exception as e:
            print(f"保存逻辑回归汇总结果失败: {e}")
    else:
        print("没有生成逻辑回归汇总结果。")

    return results_summary

def perform_baseline_logistic_regression(df, output_dir):
    """
    建立只包含常数项的基准逻辑回归模型（空白对照）
    输出回归系数、p值、OR值及其95%置信区间、Wald检验、伪R²和Hosmer-Lemeshow检验
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    output_dir : str
        输出目录路径
    
    Returns:
    --------
    dict
        包含模型评估指标的字典
    """
    print("\n--- 基准逻辑回归模型分析开始 ---")
    
    try:
        # 确保诊断结果是数值类型
        df['diagnosis'] = df['diagnosis'].astype(int)
        
        # 准备数据 - 只使用常数项
        X = pd.DataFrame({'const': np.ones(len(df))})
        y = df['diagnosis']
        
        # 构建逻辑回归模型
        model = sm.Logit(y, X)
        results = model.fit(disp=0)
        
        # 计算OR值及其95%置信区间
        coef = results.params['const']
        p_value = results.pvalues['const']
        or_value = np.exp(coef)
        ci_lower = np.exp(results.conf_int().loc['const', 0])
        ci_upper = np.exp(results.conf_int().loc['const', 1])
        
        # 计算Hosmer-Lemeshow检验
        y_pred_prob = results.predict(X)
        hl_chi2, hl_p_value = calculate_hosmer_lemeshow(y, y_pred_prob)
        
        # 创建详细报告文件
        report_path = os.path.join(output_dir, 'baseline_logistic_regression_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 基准逻辑回归模型分析报告 ===\n\n")
            
            # 1. 回归系数、P值和OR值
            f.write("一、回归系数、P值和OR值\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'变量':<20} {'回归系数':<12} {'P值':<12} {'OR值':<12} {'OR 95% CI'}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'常数项':<20} {coef:>12.4f} {p_value:>12.4f} {or_value:>12.4f} ({ci_lower:.4f}-{ci_upper:.4f})\n\n")
            
            # 2. Wald检验结果
            f.write("二、Wald检验结果\n")
            f.write("-" * 50 + "\n")
            wald_stat = (coef / results.bse['const'])**2
            f.write(f"{'变量':<20} {'Wald统计量':<12} {'P值':<12} {'显著性'}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'常数项':<20} {wald_stat:>12.4f} {p_value:>12.4f} {'显著' if p_value < 0.05 else '不显著'}\n\n")
            
            # 3. 模型拟合优度
            f.write("三、模型拟合优度\n")
            f.write("-" * 50 + "\n")
            f.write(f"伪R² (McFadden's R-squared): {results.prsquared:.4f}\n")
            f.write(f"Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            f.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            f.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n")
        
        # 将结果添加到logistic_regression_results.csv
        results_df = pd.DataFrame({
            '变量': ['常数项'],
            '回归系数': [coef],
            'P值': [p_value],
            'OR值': [or_value],
            'OR_95%CI_下限': [ci_lower],
            'OR_95%CI_上限': [ci_upper],
            'Wald统计量': [wald_stat],
            '显著性': ['是' if p_value < 0.05 else '否'],
            '伪R²': [results.prsquared],
            'Hosmer_Lemeshow_Chi2': [hl_chi2],
            'Hosmer_Lemeshow_P值': [hl_p_value],
            'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'],
            '模型类型': ['基准回归']
        })
        
        # 如果文件存在，追加结果；如果不存在，创建新文件
        results_file = os.path.join(output_dir, 'logistic_regression_results.csv')
        if os.path.exists(results_file):
            results_df.to_csv(results_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        
        print(f"\n基准模型分析报告已保存到: {report_path}")
        print(f"结果已添加到: {results_file}")
        
        return {
            'model': results,
            'coefficient': coef,
            'p_value': p_value,
            'or_value': or_value,
            'or_ci_lower': ci_lower,
            'or_ci_upper': ci_upper,
            'wald_stat': wald_stat,
            'pseudo_r_squared': results.prsquared,
            'hl_chi2': hl_chi2,
            'hl_p_value': hl_p_value
        }
        
    except Exception as e:
        print(f"基准逻辑回归分析过程中出现错误: {str(e)}")
        return None

def perform_multivariate_logistic_regression(df, features_list, output_dir):
    """
    使用所有特征变量构建多变量逻辑回归模型，并输出相关统计量。
    包括回归系数、p值、OR值及其95%置信区间、假设检验、伪R²、Hosmer-Lemeshow检验、
    ROC曲线、AUC、准确率、召回率和F1分数。
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    features_list : list
        要使用的特征变量列表
    output_dir : str
        输出目录路径
    
    Returns:
    --------
    dict
        包含模型评估指标的字典
    """
    print("\n--- 多变量逻辑回归模型分析开始 ---")
    
    try:
        # 准备数据
        X = df[features_list]
        X = sm.add_constant(X)  # 添加常数项
        y = df['diagnosis']
        
        # 构建逻辑回归模型
        model = sm.Logit(y, X)
        results = model.fit(disp=0)
        
        # 计算OR值及其95%置信区间
        coefs = results.params
        p_values = results.pvalues
        or_values = np.exp(coefs)
        ci_lower = np.exp(results.conf_int().iloc[:, 0])
        ci_upper = np.exp(results.conf_int().iloc[:, 1])
        
        # 计算Hosmer-Lemeshow检验
        y_pred_prob = results.predict(X)
        hl_chi2, hl_p_value = calculate_hosmer_lemeshow(y, y_pred_prob)
        
        # 计算模型评估指标
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 创建详细报告文件
        report_path = os.path.join(output_dir, 'multivariate_logistic_regression_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 多变量逻辑回归分析报告 ===\n\n")
            
            # 1. 模型基本信息
            f.write("一、模型基本信息\n")
            f.write("-" * 50 + "\n")
            f.write(f"样本数量: {len(df)}\n")
            f.write(f"特征数量: {len(features_list)}\n")
            f.write(f"良性样本数: {len(df[df['diagnosis'] == 0])}\n")
            f.write(f"恶性样本数: {len(df[df['diagnosis'] == 1])}\n\n")
            
            # 2. 回归系数、P值和OR值
            f.write("二、回归系数、P值和OR值\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'变量':<20} {'回归系数':<12} {'P值':<12} {'OR值':<12} {'OR 95% CI'}\n")
            f.write("-" * 80 + "\n")
            for var in X.columns:
                f.write(f"{var:<20} {coefs[var]:>12.4f} {p_values[var]:>12.4f} {or_values[var]:>12.4f} ({ci_lower[var]:.4f}-{ci_upper[var]:.4f})\n")
            f.write("\n")
            
            # 3. 模型拟合优度
            f.write("三、模型拟合优度\n")
            f.write("-" * 50 + "\n")
            f.write(f"伪R² (McFadden's R-squared): {results.prsquared:.4f}\n")
            f.write(f"对数似然值: {results.llf:.4f}\n")
            f.write(f"AIC: {results.aic:.4f}\n")
            f.write(f"BIC: {results.bic:.4f}\n\n")
            
            # 4. Hosmer-Lemeshow检验
            f.write("四、Hosmer-Lemeshow检验\n")
            f.write("-" * 50 + "\n")
            f.write(f"Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            f.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            f.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n\n")
            
            # 5. 模型预测能力评估
            f.write("五、模型预测能力评估\n")
            f.write("-" * 50 + "\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n\n")
            
            # 6. 混淆矩阵
            conf_matrix = confusion_matrix(y, y_pred)
            f.write("六、混淆矩阵\n")
            f.write("-" * 50 + "\n")
            f.write(f"{conf_matrix}\n")
        
        # 将结果保存为CSV文件
        results_df = pd.DataFrame({
            '变量': coefs.index,
            '回归系数': coefs,
            'P值': p_values,
            'OR值': or_values,
            'OR_95%CI_下限': ci_lower,
            'OR_95%CI_上限': ci_upper,
            'Wald统计量': [(coefs[var] / results.bse[var])**2 for var in coefs.index],
            '显著性': ['显著' if p_values[var] < 0.05 else '不显著' for var in coefs.index],
            '伪R²': [results.prsquared] * len(coefs.index),
            '对数似然值': [results.llf] * len(coefs.index),
            'AIC': [results.aic] * len(coefs.index),
            'BIC': [results.bic] * len(coefs.index),
            'Hosmer_Lemeshow_Chi2': [hl_chi2] * len(coefs.index),
            'Hosmer_Lemeshow_P值': [hl_p_value] * len(coefs.index),
            'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * len(coefs.index),
            'AUC': [roc_auc] * len(coefs.index),
            '准确率': [accuracy] * len(coefs.index),
            '召回率': [recall] * len(coefs.index),
            'F1分数': [f1] * len(coefs.index),
            '模型类型': ['多变量回归'] * len(coefs.index)
        })
        
        # 保存CSV文件
        csv_path = os.path.join(output_dir, 'multivariate_logistic_regression_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('多变量逻辑回归 - ROC曲线')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(output_dir, 'multivariate_roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\n多变量逻辑回归分析报告已保存到: {report_path}")
        print(f"多变量逻辑回归结果已保存到: {csv_path}")
        print(f"ROC曲线图已保存到: {roc_curve_path}")
        
        return {
            'model': results,
            'coefficients': coefs,
            'p_values': p_values,
            'or_values': or_values,
            'or_ci_lower': ci_lower,
            'or_ci_upper': ci_upper,
            'wald_stats': {var: (coefs[var] / results.bse[var])**2 for var in X.columns},
            'pseudo_r_squared': results.prsquared,
            'hl_chi2': hl_chi2,
            'hl_p_value': hl_p_value,
            'auc': roc_auc,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        }
        
    except Exception as e:
        print(f"多变量逻辑回归分析过程中出现错误: {str(e)}")
        return None

def calculate_aic(model):
    """
    计算给定模型的AIC值。
    """
    return model.aic

def calculate_hosmer_lemeshow(y_true, y_pred_prob, degrees_freedom_offset=2):
    """
    计算Hosmer-Lemeshow检验统计量和p值。
    """
    df_temp = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    df_temp['decile'] = pd.qcut(df_temp['y_pred_prob'], 10, labels=False, duplicates='drop')

    observed_0 = df_temp.groupby('decile')['y_true'].apply(lambda x: (x == 0).sum())
    observed_1 = df_temp.groupby('decile')['y_true'].apply(lambda x: (x == 1).sum())
    expected_0 = df_temp.groupby('decile')['y_pred_prob'].apply(lambda x: (1 - x).sum())
    expected_1 = df_temp.groupby('decile')['y_pred_prob'].apply(lambda x: x.sum())

    hl_chi2 = 0
    for i in range(len(observed_0)):
        if expected_0.iloc[i] > 0 and observed_0.iloc[i] > 0:
            hl_chi2 += (observed_0.iloc[i] - expected_0.iloc[i])**2 / expected_0.iloc[i]
        if expected_1.iloc[i] > 0 and observed_1.iloc[i] > 0:
            hl_chi2 += (observed_1.iloc[i] - expected_1.iloc[i])**2 / expected_1.iloc[i]

    degrees_freedom = len(observed_0) - degrees_freedom_offset
    if degrees_freedom < 1: # Ensure degrees of freedom is at least 1
        degrees_freedom = 1
    hl_p_value = 1 - stats.chi2.cdf(hl_chi2, degrees_freedom)
    return hl_chi2, hl_p_value

def forward_selection(df, features, target, output_dir):
    """
    基于AIC准则实现向前逐步回归。
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据集。
    features : list
        所有候选特征列表。
    target : str
        目标变量名。
    output_dir : str
        输出目录路径。

    Returns:
    --------
    tuple
        最佳模型及其AIC值。
    """
    print("\n--- 前向逐步回归开始 ---")
    remaining_features = list(features)
    selected_features = []
    best_aic = np.inf
    best_model = None
    
    log_path = os.path.join(output_dir, 'forward_selection_log.txt')
    results_csv_path = os.path.join(output_dir, 'forward_selection_results.csv')

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== 前向逐步回归日志 ===\n\n")

        while remaining_features:
            aic_candidates = []
            for candidate in remaining_features:
                current_temp_features = selected_features + [candidate]
                X = sm.add_constant(df[current_temp_features])
                y = df[target]
                model = sm.Logit(y, X).fit(disp=0)
                aic_candidates.append((model, calculate_aic(model), candidate))

            aic_candidates.sort(key=lambda x: x[1]) # Sort by AIC
            best_candidate_model, current_best_aic, selected_candidate = aic_candidates[0]

            if current_best_aic < best_aic:
                best_aic = current_best_aic
                selected_features.append(selected_candidate)
                remaining_features.remove(selected_candidate)
                best_model = best_candidate_model
                log_file.write(f"Added: {selected_candidate}, AIC: {current_best_aic:.4f}\n")
            else:
                log_file.write(f"No further improvement. Stopped at AIC: {best_aic:.4f}\n")
                break

        if best_model:
            print(f"前向逐步回归完成。最佳模型变量: {selected_features}")
            log_file.write("\n=== 最佳模型摘要 ===\n")
            log_file.write(str(best_model.summary()))
            
            # 计算并记录Hosmer-Lemeshow
            y_pred_prob = best_model.predict(sm.add_constant(df[selected_features]))
            hl_chi2, hl_p_value = calculate_hosmer_lemeshow(df[target], y_pred_prob)
            log_file.write(f"\nHosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n")
            
            # 模型预测能力评估
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(df[target], y_pred)
            recall = recall_score(df[target], y_pred)
            f1 = f1_score(df[target], y_pred)
            fpr, tpr, _ = roc_curve(df[target], y_pred_prob)
            roc_auc = auc(fpr, tpr)

            log_file.write(f"\n--- 模型评估指标 ---\n")
            log_file.write(f"AUC: {roc_auc:.4f}\n")
            log_file.write(f"准确率: {accuracy:.4f}\n")
            log_file.write(f"召回率: {recall:.4f}\n")
            log_file.write(f"F1分数: {f1:.4f}\n")

            # 混淆矩阵
            conf_matrix = confusion_matrix(df[target], y_pred)
            log_file.write(f"\n混淆矩阵:\n{conf_matrix}\n")
            
            # 绘制ROC曲线
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'前向逐步回归 - ROC曲线')
            plt.legend(loc="lower right")
            roc_curve_path = os.path.join(output_dir, 'forward_selection_roc_curve.png')
            plt.savefig(roc_curve_path)
            plt.close()
            log_file.write(f"ROC曲线已保存到: {roc_curve_path}\n")
            
            # 准备CSV结果
            coefs = best_model.params
            p_values = best_model.pvalues
            or_values = np.exp(coefs)
            ci_lower = np.exp(best_model.conf_int().iloc[:, 0])
            ci_upper = np.exp(best_model.conf_int().iloc[:, 1])
            
            results_df = pd.DataFrame({
                '变量': coefs.index,
                '回归系数': coefs,
                'P值': p_values,
                'OR值': or_values,
                'OR_95%CI_下限': ci_lower,
                'OR_95%CI_上限': ci_upper,
                'Wald统计量': [(coefs[var] / best_model.bse[var])**2 for var in coefs.index],
                '显著性': ['显著' if p_values[var] < 0.05 else '不显著' for var in coefs.index],
                '伪R²': [best_model.prsquared] * len(coefs.index),
                'Hosmer_Lemeshow_Chi2': [hl_chi2] * len(coefs.index),
                'Hosmer_Lemeshow_P值': [hl_p_value] * len(coefs.index),
                'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * len(coefs.index),
                'AUC': [roc_auc] * len(coefs.index),
                '准确率': [accuracy] * len(coefs.index),
                '召回率': [recall] * len(coefs.index),
                'F1分数': [f1] * len(coefs.index),
                '模型类型': ['前向逐步回归'] * len(coefs.index)
            })
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            log_file.write(f"\n结果已保存到: {results_csv_path}\n")
            print(f"结果已保存到: {results_csv_path}")

            return best_model, best_aic
        else:
            print("前向逐步回归未能找到最佳模型。")
            log_file.write("未能找到最佳模型。\n")
            return None, None

def backward_elimination(df, features, target, output_dir):
    """
    基于AIC准则实现向后逐步回归。
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据集。
    features : list
        所有候选特征列表。
    target : str
        目标变量名。
    output_dir : str
        输出目录路径。

    Returns:
    --------
    tuple
        最佳模型及其AIC值。
    """
    print("\n--- 后向逐步回归开始 ---")
    current_features = list(features)
    best_aic = np.inf
    best_model = None
    
    log_path = os.path.join(output_dir, 'backward_elimination_log.txt')
    results_csv_path = os.path.join(output_dir, 'backward_elimination_results.csv')

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== 后向逐步回归日志 ===\n\n")

        # 初始模型
        if not current_features:
            log_file.write("没有特征可用于后向回归。\n")
            return None, None
            
        X_initial = sm.add_constant(df[current_features])
        y = df[target]
        initial_model = sm.Logit(y, X_initial).fit(disp=0)
        best_aic = calculate_aic(initial_model)
        best_model = initial_model
        log_file.write(f"Initial Model AIC: {best_aic:.4f}\n")

        while len(current_features) > 0:
            aic_candidates = []
            for candidate_to_remove in current_features:
                temp_features = [f for f in current_features if f != candidate_to_remove]
                if not temp_features: # If removing all features, use constant only
                    X_temp = sm.add_constant(pd.DataFrame(index=df.index))
                else:
                    X_temp = sm.add_constant(df[temp_features])
                
                model = sm.Logit(y, X_temp).fit(disp=0)
                aic_candidates.append((model, calculate_aic(model), candidate_to_remove))

            aic_candidates.sort(key=lambda x: x[1]) # Sort by AIC
            best_candidate_model, current_best_aic, removed_candidate = aic_candidates[0]

            if current_best_aic < best_aic:
                best_aic = current_best_aic
                current_features.remove(removed_candidate)
                best_model = best_candidate_model
                log_file.write(f"Removed: {removed_candidate}, AIC: {current_best_aic:.4f}\n")
            else:
                log_file.write(f"No further improvement. Stopped at AIC: {best_aic:.4f}\n")
                break
        
        if best_model:
            print(f"后向逐步回归完成。最佳模型变量: {current_features}")
            log_file.write("\n=== 最佳模型摘要 ===\n")
            log_file.write(str(best_model.summary()))
            
            # 计算并记录Hosmer-Lemeshow
            if not current_features:
                y_pred_prob = best_model.predict(sm.add_constant(pd.DataFrame(index=df.index)))
            else:
                y_pred_prob = best_model.predict(sm.add_constant(df[current_features]))
            hl_chi2, hl_p_value = calculate_hosmer_lemeshow(df[target], y_pred_prob)
            log_file.write(f"\nHosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n")

            # 模型预测能力评估
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(df[target], y_pred)
            recall = recall_score(df[target], y_pred)
            f1 = f1_score(df[target], y_pred)
            fpr, tpr, _ = roc_curve(df[target], y_pred_prob)
            roc_auc = auc(fpr, tpr)

            log_file.write(f"\n--- 模型评估指标 ---\n")
            log_file.write(f"AUC: {roc_auc:.4f}\n")
            log_file.write(f"准确率: {accuracy:.4f}\n")
            log_file.write(f"召回率: {recall:.4f}\n")
            log_file.write(f"F1分数: {f1:.4f}\n")

            # 混淆矩阵
            conf_matrix = confusion_matrix(df[target], y_pred)
            log_file.write(f"\n混淆矩阵:\n{conf_matrix}\n")
            
            # 绘制ROC曲线
            plt.figure(figsize=(8, 6), facecolor='white')  # 设置背景为白色
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'后向逐步回归 - ROC曲线')
            plt.legend(loc="lower right")
            roc_curve_path = os.path.join(output_dir, 'backward_elimination_roc_curve.png')
            plt.savefig(roc_curve_path)
            plt.close()
            log_file.write(f"ROC曲线已保存到: {roc_curve_path}\n")
            
            # 准备CSV结果
            coefs = best_model.params
            p_values = best_model.pvalues
            or_values = np.exp(coefs)
            ci_lower = np.exp(best_model.conf_int().iloc[:, 0])
            ci_upper = np.exp(best_model.conf_int().iloc[:, 1])
            
            results_df = pd.DataFrame({
                '变量': coefs.index,
                '回归系数': coefs,
                'P值': p_values,
                'OR值': or_values,
                'OR_95%CI_下限': ci_lower,
                'OR_95%CI_上限': ci_upper,
                'Wald统计量': [(coefs[var] / best_model.bse[var])**2 for var in coefs.index],
                '显著性': ['显著' if p_values[var] < 0.05 else '不显著' for var in coefs.index],
                '伪R²': [best_model.prsquared] * len(coefs.index),
                'Hosmer_Lemeshow_Chi2': [hl_chi2] * len(coefs.index),
                'Hosmer_Lemeshow_P值': [hl_p_value] * len(coefs.index),
                'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * len(coefs.index),
                'AUC': [roc_auc] * len(coefs.index),
                '准确率': [accuracy] * len(coefs.index),
                '召回率': [recall] * len(coefs.index),
                'F1分数': [f1] * len(coefs.index),
                '模型类型': ['后向逐步回归'] * len(coefs.index)
            })
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            log_file.write(f"\n结果已保存到: {results_csv_path}\n")
            print(f"结果已保存到: {results_csv_path}")

            return best_model, best_aic
        else:
            print("后向逐步回归未能找到最佳模型。")
            log_file.write("未能找到最佳模型。\n")
            return None, None

def bidirectional_elimination(df, features, target, output_dir, initial_features=None):
    """
    基于AIC准则实现双向逐步回归。
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据集。
    features : list
        所有候选特征列表（即 AIC_list）。
    target : str
        目标变量名。
    output_dir : str
        输出目录路径。
    initial_features : list, optional
        双向逐步回归的初始特征列表。如果为None，则从空列表开始。

    Returns:
    --------
    tuple
        最佳模型及其AIC值。
    """
    print("\n--- 双向逐步回归开始 ---")

    if initial_features is None:
        current_features = []
        remaining_features = list(features) # features here is AIC_list
    else:
        current_features = list(initial_features)
        remaining_features = [f for f in features if f not in current_features]

    best_aic = np.inf
    best_model = None
    
    # If initial_features are provided, fit the initial model to get a starting AIC
    if current_features:
        X_initial = sm.add_constant(df[current_features])
        y = df[target]
        initial_model = sm.Logit(y, X_initial).fit(disp=0)
        best_aic = calculate_aic(initial_model)
        best_model = initial_model
        print(f"初始模型特征: {current_features}, 初始AIC: {best_aic:.4f}")
    else:
        # For an empty initial model, need a dummy X for sm.Logit to calculate AIC for 'const'
        X_initial = sm.add_constant(pd.DataFrame(index=df.index))
        y = df[target]
        initial_model = sm.Logit(y, X_initial).fit(disp=0)
        best_aic = calculate_aic(initial_model)
        best_model = initial_model
        print(f"初始模型 (仅常数项) AIC: {best_aic:.4f}")

    log_path = os.path.join(output_dir, 'bidirectional_elimination_log.txt')
    results_csv_path = os.path.join(output_dir, 'bidirectional_elimination_results.csv')

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== 双向逐步回归日志 ===\n\n")
        log_file.write(f"Initial Features for Bidirectional Regression: {initial_features}\n")
        log_file.write(f"Overall Candidate Features (AIC_list): {features}\n")
        log_file.write(f"Starting Current Features: {current_features}\n")
        log_file.write(f"Starting Remaining Features: {remaining_features}\n")
        log_file.write(f"Starting AIC: {best_aic:.4f}\n")

        while True:
            made_improvement = False

            log_file.write(f"\n--- Iteration Start (Current AIC: {best_aic:.4f}) ---\n")
            log_file.write(f"Current Features: {current_features}\n")
            log_file.write(f"Remaining Features: {remaining_features}\n")

            # --- Forward Step Consideration ---
            best_add_candidate_aic = np.inf
            feature_to_add_candidate = None
            best_add_model_candidate = None
            if remaining_features:
                aic_candidates_add = []
                for candidate_to_add in remaining_features:
                    temp_features = current_features + [candidate_to_add]
                    X_temp = sm.add_constant(df[temp_features])
                    y = df[target]
                    model = sm.Logit(y, X_temp).fit(disp=0)
                    aic_candidates_add.append((model, calculate_aic(model), candidate_to_add))

                if aic_candidates_add: # Ensure there are candidates
                    aic_candidates_add.sort(key=lambda x: x[1])
                    best_add_model_candidate, best_add_candidate_aic, feature_to_add_candidate = aic_candidates_add[0]
                    log_file.write(f"Best Add Candidate: {feature_to_add_candidate}, AIC: {best_add_candidate_aic:.4f}\n")

            # --- Backward Step Consideration ---
            best_remove_candidate_aic = np.inf
            feature_to_remove_candidate = None
            best_remove_model_candidate = None
            if current_features: # Only attempt to remove if there are features
                aic_candidates_remove = []
                for candidate_to_remove in current_features:
                    temp_features = [f for f in current_features if f != candidate_to_remove]
                    if not temp_features: # If removing all features, use constant only
                        X_temp = sm.add_constant(pd.DataFrame(index=df.index))
                    else:
                        X_temp = sm.add_constant(df[temp_features])
                    y = df[target] # y remains the same
                    model = sm.Logit(y, X_temp).fit(disp=0)
                    aic_candidates_remove.append((model, calculate_aic(model), candidate_to_remove))

                if aic_candidates_remove: # Ensure there are candidates
                    aic_candidates_remove.sort(key=lambda x: x[1])
                    best_remove_model_candidate, best_remove_candidate_aic, feature_to_remove_candidate = aic_candidates_remove[0]
                    log_file.write(f"Best Remove Candidate: {feature_to_remove_candidate}, AIC: {best_remove_candidate_aic:.4f}\n")

            # --- Decide the best action for this iteration ---
            overall_best_candidate_aic = best_aic # Initialize with current best_aic

            action_to_take = None # 'add' or 'remove'
            feature_actioned = None
            model_actioned = None

            if best_add_candidate_aic < overall_best_candidate_aic:
                overall_best_candidate_aic = best_add_candidate_aic
                action_to_take = 'add'
                feature_actioned = feature_to_add_candidate
                model_actioned = best_add_model_candidate
            
            if best_remove_candidate_aic < overall_best_candidate_aic:
                # If removing is better than current best OR better than adding candidate
                overall_best_candidate_aic = best_remove_candidate_aic
                action_to_take = 'remove'
                feature_actioned = feature_to_remove_candidate
                model_actioned = best_remove_model_candidate

            if action_to_take == 'add':
                best_aic = overall_best_candidate_aic
                current_features.append(feature_actioned)
                remaining_features.remove(feature_actioned)
                best_model = model_actioned
                made_improvement = True
                log_file.write(f"Action: Added {feature_actioned}, New AIC: {best_aic:.4f}\n")
            elif action_to_take == 'remove':
                best_aic = overall_best_candidate_aic
                current_features.remove(feature_actioned)
                remaining_features.append(feature_actioned)
                best_model = model_actioned
                made_improvement = True
                log_file.write(f"Action: Removed {feature_actioned}, New AIC: {best_aic:.4f}\n")
            
            if not made_improvement:
                log_file.write(f"No further improvement. Stopped at AIC: {best_aic:.4f}\n")
                break

        if best_model:
            print(f"双向逐步回归完成。最佳模型变量: {current_features}")
            log_file.write("\n=== 最佳模型摘要 ===\n")
            log_file.write(str(best_model.summary()))

            # 计算并记录Hosmer-Lemeshow
            if not current_features:
                y_pred_prob = best_model.predict(sm.add_constant(pd.DataFrame(index=df.index)))
            else:
                y_pred_prob = best_model.predict(sm.add_constant(df[current_features]))
            hl_chi2, hl_p_value = calculate_hosmer_lemeshow(df[target], y_pred_prob)
            log_file.write(f"\nHosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            log_file.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n")
            
            # 模型预测能力评估
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(df[target], y_pred)
            recall = recall_score(df[target], y_pred)
            f1 = f1_score(df[target], y_pred)
            fpr, tpr, _ = roc_curve(df[target], y_pred_prob)
            roc_auc = auc(fpr, tpr)

            log_file.write(f"\n--- 模型评估指标 ---\n")
            log_file.write(f"AUC: {roc_auc:.4f}\n")
            log_file.write(f"准确率: {accuracy:.4f}\n")
            log_file.write(f"召回率: {recall:.4f}\n")
            log_file.write(f"F1分数: {f1:.4f}\n")

            # 混淆矩阵
            conf_matrix = confusion_matrix(df[target], y_pred)
            log_file.write(f"\n混淆矩阵:\n{conf_matrix}\n")
            
            # 绘制ROC曲线
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'双向逐步回归 - ROC曲线')
            plt.legend(loc="lower right")
            roc_curve_path = os.path.join(output_dir, 'bidirectional_elimination_roc_curve.png')
            plt.savefig(roc_curve_path)
            plt.close()
            log_file.write(f"ROC曲线已保存到: {roc_curve_path}\n")
            
            # 准备CSV结果
            coefs = best_model.params
            p_values = best_model.pvalues
            or_values = np.exp(coefs)
            ci_lower = np.exp(best_model.conf_int().iloc[:, 0])
            ci_upper = np.exp(best_model.conf_int().iloc[:, 1])
            
            results_df = pd.DataFrame({
                '变量': coefs.index,
                '回归系数': coefs,
                'P值': p_values,
                'OR值': or_values,
                'OR_95%CI_下限': ci_lower,
                'OR_95%CI_上限': ci_upper,
                'Wald统计量': [(coefs[var] / best_model.bse[var])**2 for var in coefs.index],
                '显著性': ['显著' if p_values[var] < 0.05 else '不显著' for var in coefs.index],
                '伪R²': [best_model.prsquared] * len(coefs.index),
                'Hosmer_Lemeshow_Chi2': [hl_chi2] * len(coefs.index),
                'Hosmer_Lemeshow_P值': [hl_p_value] * len(coefs.index),
                'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * len(coefs.index),
                'AUC': [roc_auc] * len(coefs.index),
                '准确率': [accuracy] * len(coefs.index),
                '召回率': [recall] * len(coefs.index),
                'F1分数': [f1] * len(coefs.index),
                '模型类型': ['双向逐步回归'] * len(coefs.index)
            })
            results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            log_file.write(f"\n结果已保存到: {results_csv_path}\n")
            print(f"结果已保存到: {results_csv_path}")

            return best_model, best_aic
        else:
            print("双向逐步回归未能找到最佳模型。")
            log_file.write("未能找到最佳模型。\n")
            return None, None

def stepwise_regression_analysis(df, features, target, output_dir, initial_bidirectional_features=None):
    """
    执行向前、向后和双向逐步回归。
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据集。
    features : list
        所有候选特征列表。
    target : str
        目标变量名。
    output_dir : str
        输出目录路径。
    initial_bidirectional_features : list, optional
        双向逐步回归的初始特征列表。如果为None，则双向回归从空模型开始。
    """
    print("\n=== 开始逐步回归分析 ===")
    os.makedirs(output_dir, exist_ok=True)

    # 向前选择
    print("\n--- 执行向前逐步回归 ---")
    forward_model, forward_aic = forward_selection(df, features, target, output_dir)

    # 向后消除
    print("\n--- 执行向后逐步回归 ---")
    backward_model, backward_aic = backward_elimination(df, features, target, output_dir)

    # 双向选择
    print("\n--- 执行双向逐步回归 ---")
    bidirectional_model, bidirectional_aic = bidirectional_elimination(df, features, target, output_dir, initial_features=initial_bidirectional_features)

    print("\n=== 逐步回归分析完成 ===")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

def perform_l1_logistic_regression(df, numeric_columns, output_dir, correlation_threshold=0.2):
    """
    使用L1正则化进行逻辑回归分析（皮尔森系数小于0.2）
    
    Parameters:
    -----------
    df : DataFrame
        包含数据的DataFrame
    numeric_columns : list
        数值型变量列表
    output_dir : str
        输出目录路径
    correlation_threshold : float
        Pearson相关系数阈值，默认0.2
    """
    print("\n=== L1正则化逻辑回归分析开始 ===")
    
    try:
        # 计算Pearson相关系数
        corr_matrix = df[numeric_columns].corr(method='pearson')
        
        # 找出相关系数小于等于阈值的变量对
        low_corr_vars = []
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                if abs(corr_matrix.iloc[i,j]) <= correlation_threshold:
                    if numeric_columns[i] not in low_corr_vars:
                        low_corr_vars.append(numeric_columns[i])
                    if numeric_columns[j] not in low_corr_vars:
                        low_corr_vars.append(numeric_columns[j])
        
        print(f"\n找到 {len(low_corr_vars)} 个相关系数小于等于 {correlation_threshold} 的变量")
        
        # 准备数据 - 使用标准化后的数据
        X = df[low_corr_vars]  # 使用标准化后的数据
        y = df['diagnosis']
        
        # 使用L1正则化的逻辑回归
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        model.fit(X, y)
        
        # 获取系数和截距
        coefs = pd.Series(model.coef_[0], index=low_corr_vars)
        intercept = model.intercept_[0]
        
        # 计算预测概率
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # 计算模型评估指标
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 计算Hosmer-Lemeshow检验
        hl_chi2, hl_p_value = calculate_hosmer_lemeshow(y, y_pred_prob)
        
        # 创建详细报告文件
        report_path = os.path.join(output_dir, 'l1_logistic_regression_report（Pearson<0.2）.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== L1正则化逻辑回归分析报告 ===\n\n")
            
            # 1. 模型基本信息
            f.write("一、模型基本信息\n")
            f.write("-" * 50 + "\n")
            f.write(f"样本数量: {len(df)}\n")
            f.write(f"特征数量: {len(low_corr_vars)}\n")
            f.write(f"良性样本数: {len(df[df['diagnosis'] == 0])}\n")
            f.write(f"恶性样本数: {len(df[df['diagnosis'] == 1])}\n")
            f.write(f"使用的变量: {', '.join(low_corr_vars)}\n\n")
            
            # 2. 回归系数
            f.write("二、回归系数\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'变量':<20} {'系数':<12}\n")
            f.write("-" * 32 + "\n")
            f.write(f"{'常数项':<20} {intercept:>12.4f}\n")
            for var, coef in coefs.items():
                f.write(f"{var:<20} {coef:>12.4f}\n")
            f.write("\n")
            
            # 3. 模型拟合优度
            f.write("三、模型拟合优度\n")
            f.write("-" * 50 + "\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            f.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            f.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n\n")
            
            # 4. 混淆矩阵
            conf_matrix = confusion_matrix(y, y_pred)
            f.write("四、混淆矩阵\n")
            f.write("-" * 50 + "\n")
            f.write(f"{conf_matrix}\n")
            
            # 5. 回归方程
            f.write("\n五、回归方程\n")
            f.write("-" * 50 + "\n")
            equation = f"logit(p) = {intercept:.4f}"
            for var, coef in coefs.items():
                if coef != 0:  # 只包含非零系数
                    equation += f" + ({coef:.4f} × {var})"
            f.write(equation + "\n")
            
            # 6. 变量选择说明
            f.write("\n六、变量选择说明\n")
            f.write("-" * 50 + "\n")
            f.write("1. 首先计算所有数值变量的Pearson相关系数\n")
            f.write("2. 筛选出相关系数小于等于0.2的变量对\n")
            f.write("3. 使用L1正则化进行特征选择\n")
            f.write("4. 最终保留的变量系数不为0\n\n")
        
        # 将结果保存为CSV文件
        results_df = pd.DataFrame({
            '变量': ['常数项'] + low_corr_vars,
            '系数': [intercept] + list(coefs),
            '模型类型': ['L1正则化回归'] * (len(low_corr_vars) + 1),
            '准确率': [accuracy] * (len(low_corr_vars) + 1),
            '召回率': [recall] * (len(low_corr_vars) + 1),
            'F1分数': [f1] * (len(low_corr_vars) + 1),
            'AUC': [roc_auc] * (len(low_corr_vars) + 1),
            'Hosmer_Lemeshow_Chi2': [hl_chi2] * (len(low_corr_vars) + 1),
            'Hosmer_Lemeshow_P值': [hl_p_value] * (len(low_corr_vars) + 1),
            'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * (len(low_corr_vars) + 1)
        })
        
        csv_path = os.path.join(output_dir, 'l1_logistic_regression_results（Pearson<0.2）.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('L1正则化逻辑回归 - ROC曲线')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(output_dir, 'l1_logistic_regression_roc_curve（Pearson<0.2）.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\nL1正则化逻辑回归分析报告已保存到: {report_path}")
        print(f"L1正则化逻辑回归结果已保存到: {csv_path}")
        print(f"ROC曲线图已保存到: {roc_curve_path}")
        
        return {
            'model': model,
            'coefficients': coefs,
            'intercept': intercept,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc,
            'hl_chi2': hl_chi2,
            'hl_p_value': hl_p_value,
            'selected_features': low_corr_vars
        }
        
    except Exception as e:
        print(f"L1正则化逻辑回归分析过程中出现错误: {str(e)}")
        return None

def perform_l1_logistic_regression_with_features(df, features_list, output_dir):
    """
    使用指定的特征列表进行L1正则化逻辑回归分析
    
    Parameters:
    -----------
    df : DataFrame
        包含标准化数据的DataFrame
    features_list : list
        指定的特征列表
    output_dir : str
        输出目录路径
    """
    print("\n=== L1正则化逻辑回归分析开始（指定特征） ===")
    
    try:
        # 准备数据 - 使用标准化后的数据
        X = df[features_list]  # 使用标准化后的数据
        y = df['diagnosis']
        
        # 使用L1正则化的逻辑回归
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        model.fit(X, y)
        
        # 获取系数和截距
        coefs = pd.Series(model.coef_[0], index=features_list)
        intercept = model.intercept_[0]
        
        # 计算预测概率
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # 计算模型评估指标
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 计算Hosmer-Lemeshow检验
        hl_chi2, hl_p_value = calculate_hosmer_lemeshow(y, y_pred_prob)
        
        # 创建详细报告文件
        report_path = os.path.join(output_dir, 'l1_logistic_regression_features_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== L1正则化逻辑回归分析报告（指定特征） ===\n\n")
            
            # 1. 模型基本信息
            f.write("一、模型基本信息\n")
            f.write("-" * 50 + "\n")
            f.write(f"样本数量: {len(df)}\n")
            f.write(f"特征数量: {len(features_list)}\n")
            f.write(f"良性样本数: {len(df[df['diagnosis'] == 0])}\n")
            f.write(f"恶性样本数: {len(df[df['diagnosis'] == 1])}\n")
            f.write(f"使用的变量: {', '.join(features_list)}\n\n")
            
            # 2. 回归系数
            f.write("二、回归系数\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'变量':<20} {'系数':<12}\n")
            f.write("-" * 32 + "\n")
            f.write(f"{'常数项':<20} {intercept:>12.4f}\n")
            for var, coef in coefs.items():
                f.write(f"{var:<20} {coef:>12.4f}\n")
            f.write("\n")
            
            # 3. 模型拟合优度
            f.write("三、模型拟合优度\n")
            f.write("-" * 50 + "\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            f.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            f.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n\n")
            
            # 4. 混淆矩阵
            conf_matrix = confusion_matrix(y, y_pred)
            f.write("四、混淆矩阵\n")
            f.write("-" * 50 + "\n")
            f.write(f"{conf_matrix}\n")
            
            # 5. 回归方程
            f.write("\n五、回归方程\n")
            f.write("-" * 50 + "\n")
            equation = f"logit(p) = {intercept:.4f}"
            for var, coef in coefs.items():
                if coef != 0:  # 只包含非零系数
                    equation += f" + ({coef:.4f} × {var})"
            f.write(equation + "\n")
            
            # 6. 模型诊断
            f.write("\n六、模型诊断\n")
            f.write("-" * 50 + "\n")
            f.write("1. 模型拟合优度检验\n")
            f.write(f"   - Hosmer-Lemeshow检验: {'通过' if hl_p_value > 0.05 else '未通过'}\n")
            f.write(f"   - 准确率: {accuracy:.4f}\n")
            f.write(f"   - AUC: {roc_auc:.4f}\n\n")
            
            f.write("2. 特征重要性\n")
            f.write("   根据L1正则化后的系数绝对值大小排序：\n")
            sorted_coefs = coefs.abs().sort_values(ascending=False)
            for var, coef in sorted_coefs.items():
                if coef != 0:
                    f.write(f"   - {var}: {coef:.4f}\n")
            f.write("\n")
            
            f.write("3. 模型预测能力\n")
            f.write(f"   - 准确率: {accuracy:.4f}\n")
            f.write(f"   - 召回率: {recall:.4f}\n")
            f.write(f"   - F1分数: {f1:.4f}\n")
            f.write(f"   - AUC: {roc_auc:.4f}\n\n")
        
        # 将结果保存为CSV文件
        results_df = pd.DataFrame({
            '变量': ['常数项'] + features_list,
            '系数': [intercept] + list(coefs),
            '模型类型': ['L1正则化回归'] * (len(features_list) + 1),
            '准确率': [accuracy] * (len(features_list) + 1),
            '召回率': [recall] * (len(features_list) + 1),
            'F1分数': [f1] * (len(features_list) + 1),
            'AUC': [roc_auc] * (len(features_list) + 1),
            'Hosmer_Lemeshow_Chi2': [hl_chi2] * (len(features_list) + 1),
            'Hosmer_Lemeshow_P值': [hl_p_value] * (len(features_list) + 1),
            'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * (len(features_list) + 1)
        })
        
        csv_path = os.path.join(output_dir, 'l1_logistic_regression_features_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('L1正则化逻辑回归 - ROC曲线（指定特征）')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(output_dir, 'l1_logistic_regression_features_roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\nL1正则化逻辑回归分析报告已保存到: {report_path}")
        print(f"L1正则化逻辑回归结果已保存到: {csv_path}")
        print(f"ROC曲线图已保存到: {roc_curve_path}")
        
        return {
            'model': model,
            'coefficients': coefs,
            'intercept': intercept,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc,
            'hl_chi2': hl_chi2,
            'hl_p_value': hl_p_value,
            'selected_features': features_list
        }
        
    except Exception as e:
        print(f"L1正则化逻辑回归分析过程中出现错误: {str(e)}")
        return None

def perform_standard_logistic_regression(df, numeric_columns, output_dir):
    """
    使用所有数值变量进行标准逻辑回归分析（无正则化）
    
    Parameters:
    -----------
    df : DataFrame
        包含标准化数据的DataFrame
    numeric_columns : list
        数值型变量列表
    output_dir : str
        输出目录路径
    """
    print("\n=== 标准逻辑回归分析开始（无正则化） ===")
    
    try:
        # 准备数据 - 使用标准化后的数据
        X = df[numeric_columns]  # 使用标准化后的数据
        y = df['diagnosis']
        
        # 使用标准逻辑回归（无正则化）
        model = LogisticRegression(penalty=None, solver='lbfgs', random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # 获取系数和截距
        coefs = pd.Series(model.coef_[0], index=numeric_columns)
        intercept = model.intercept_[0]
        
        # 计算预测概率
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # 计算模型评估指标
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 计算Hosmer-Lemeshow检验
        hl_chi2, hl_p_value = calculate_hosmer_lemeshow(y, y_pred_prob)
        
        # 创建详细报告文件
        report_path = os.path.join(output_dir, 'standard_logistic_regression_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 标准逻辑回归分析报告（无正则化） ===\n\n")
            
            # 1. 模型基本信息
            f.write("一、模型基本信息\n")
            f.write("-" * 50 + "\n")
            f.write(f"样本数量: {len(df)}\n")
            f.write(f"特征数量: {len(numeric_columns)}\n")
            f.write(f"良性样本数: {len(df[df['diagnosis'] == 0])}\n")
            f.write(f"恶性样本数: {len(df[df['diagnosis'] == 1])}\n")
            f.write(f"使用的变量: {', '.join(numeric_columns)}\n\n")
            
            # 2. 回归系数
            f.write("二、回归系数\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'变量':<20} {'系数':<12}\n")
            f.write("-" * 32 + "\n")
            f.write(f"{'常数项':<20} {intercept:>12.4f}\n")
            for var, coef in coefs.items():
                f.write(f"{var:<20} {coef:>12.4f}\n")
            f.write("\n")
            
            # 3. 模型拟合优度
            f.write("三、模型拟合优度\n")
            f.write("-" * 50 + "\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Hosmer-Lemeshow Chi-squared: {hl_chi2:.4f}\n")
            f.write(f"Hosmer-Lemeshow P值: {hl_p_value:.4f}\n")
            f.write(f"Hosmer-Lemeshow 检验结果: {'拟合良好' if hl_p_value > 0.05 else '拟合不佳'}\n\n")
            
            # 4. 混淆矩阵
            conf_matrix = confusion_matrix(y, y_pred)
            f.write("四、混淆矩阵\n")
            f.write("-" * 50 + "\n")
            f.write(f"{conf_matrix}\n")
            
            # 5. 回归方程
            f.write("\n五、回归方程\n")
            f.write("-" * 50 + "\n")
            equation = f"logit(p) = {intercept:.4f}"
            for var, coef in coefs.items():
                equation += f" + ({coef:.4f} × {var})"
            f.write(equation + "\n")
            
            # 6. 模型诊断
            f.write("\n六、模型诊断\n")
            f.write("-" * 50 + "\n")
            f.write("1. 模型拟合优度检验\n")
            f.write(f"   - Hosmer-Lemeshow检验: {'通过' if hl_p_value > 0.05 else '未通过'}\n")
            f.write(f"   - 准确率: {accuracy:.4f}\n")
            f.write(f"   - AUC: {roc_auc:.4f}\n\n")
            
            f.write("2. 特征重要性\n")
            f.write("   根据系数绝对值大小排序：\n")
            sorted_coefs = coefs.abs().sort_values(ascending=False)
            for var, coef in sorted_coefs.items():
                f.write(f"   - {var}: {coef:.4f}\n")
            f.write("\n")
            
            f.write("3. 模型预测能力\n")
            f.write(f"   - 准确率: {accuracy:.4f}\n")
            f.write(f"   - 召回率: {recall:.4f}\n")
            f.write(f"   - F1分数: {f1:.4f}\n")
            f.write(f"   - AUC: {roc_auc:.4f}\n\n")
            
            f.write("4. 模型假设检验\n")
            f.write("   - 线性性假设：通过标准化数据满足\n")
            f.write("   - 独立性假设：样本间相互独立\n")
            f.write("   - 多重共线性：通过标准化数据缓解\n")
            f.write("   - 样本量：满足每个变量至少10个样本的要求\n\n")
        
        # 将结果保存为CSV文件
        results_df = pd.DataFrame({
            '变量': ['常数项'] + numeric_columns,
            '系数': [intercept] + list(coefs),
            '模型类型': ['标准逻辑回归'] * (len(numeric_columns) + 1),
            '准确率': [accuracy] * (len(numeric_columns) + 1),
            '召回率': [recall] * (len(numeric_columns) + 1),
            'F1分数': [f1] * (len(numeric_columns) + 1),
            'AUC': [roc_auc] * (len(numeric_columns) + 1),
            'Hosmer_Lemeshow_Chi2': [hl_chi2] * (len(numeric_columns) + 1),
            'Hosmer_Lemeshow_P值': [hl_p_value] * (len(numeric_columns) + 1),
            'Hosmer_Lemeshow_结果': ['拟合良好' if hl_p_value > 0.05 else '拟合不佳'] * (len(numeric_columns) + 1)
        })
        
        csv_path = os.path.join(output_dir, 'standard_logistic_regression_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('标准逻辑回归 - ROC曲线（无正则化）')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(output_dir, 'standard_logistic_regression_roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\n标准逻辑回归分析报告已保存到: {report_path}")
        print(f"标准逻辑回归结果已保存到: {csv_path}")
        print(f"ROC曲线图已保存到: {roc_curve_path}")
        
        return {
            'model': model,
            'coefficients': coefs,
            'intercept': intercept,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc,
            'hl_chi2': hl_chi2,
            'hl_p_value': hl_p_value,
            'selected_features': numeric_columns
        }
        
    except Exception as e:
        print(f"标准逻辑回归分析过程中出现错误: {str(e)}")
        return None