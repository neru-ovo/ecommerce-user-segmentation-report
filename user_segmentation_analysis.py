import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('user_personalized_features.csv')

print("=== 数据概览 ===")
print(f"数据行数: {df.shape[0]}, 列数: {df.shape[1]}")
print("\n数据类型:")
print(df.dtypes)
print("\n缺失值检查:")
print(df.isnull().sum())
print("\n描述性统计:")
print(df.describe())

print("\n=== 探索性数据分析 ===")

numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency', 
                'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed']

corr_matrix = df[numeric_cols].corr()
print("\n相关性矩阵:")
print(corr_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('变量相关性热力图')
plt.savefig('correlation_heatmap.png')
plt.close()

print("\n关键发现一：停留时间与购买频率的相关性")
print(f"停留时间与购买频率相关系数: {corr_matrix.loc['Time_Spent_on_Site_Minutes', 'Purchase_Frequency']:.4f}")

print("\n关键发现二：高收入群体消费分析")
high_income_threshold = df['Income'].quantile(0.75)
high_income_users = df[df['Income'] >= high_income_threshold]
avg_total_spending = df['Total_Spending'].mean()
high_income_below_avg = high_income_users[high_income_users['Total_Spending'] < avg_total_spending].shape[0]
print(f"高收入群体(收入前25%)人数: {high_income_users.shape[0]}")
print(f"高收入群体中消费低于平均的人数: {high_income_below_avg}")
print(f"占比: {high_income_below_avg / high_income_users.shape[0]:.2%}")

print("\n=== 指标构建 ===")

scaler = MinMaxScaler(feature_range=(0, 1))
df['Time_Spent_Score'] = scaler.fit_transform(df[['Time_Spent_on_Site_Minutes']])
df['Pages_Viewed_Score'] = scaler.fit_transform(df[['Pages_Viewed']])
df['Intent_Score'] = df['Time_Spent_Score'] * 0.5 + df['Pages_Viewed_Score'] * 0.5

df['Conversion_Friction'] = df['Pages_Viewed'] / (df['Purchase_Frequency'] + 1)

def calculate_active_connection(row):
    subscribed = row['Newsletter_Subscription']
    last_login = row['Last_Login_Days_Ago']
    
    if subscribed and last_login <= 7:
        return 3
    elif not subscribed and last_login <= 7:
        return 2
    else:
        return 1

df['Active_Connection'] = df.apply(calculate_active_connection, axis=1)

# 【修改】统一使用4分位数分箱，与"收入前25%"的分析逻辑保持一致
# qcut将数据等频分为4箱，每箱25%，边界由数据分布自动决定
df['Purchase_Power'] = pd.qcut(df['Income'], q=4, labels=['低收入', '中低收入', '中高收入', '高收入'])

print("\n=== RFM模型构建 ===")

df['R_Score'] = pd.qcut(df['Last_Login_Days_Ago'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
df['R_Score'] = 5 - df['R_Score']
df['F_Score'] = pd.qcut(df['Purchase_Frequency'] + 1, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
df['M_Score'] = pd.qcut(df['Total_Spending'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

print("\n=== 用户分层 ===")

def segment_user(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    intent = row['Intent_Score']
    purchase_power = row['Purchase_Power']
    
    if r >= 4 and f >= 4 and m >= 4:
        return '核心VIP'
    elif r >= 3 and f >= 3 and m >= 3:
        return '重要价值客户'
    elif r >= 3 and f >= 2 and m >= 2:
        return '潜力客户'
    elif r <= 2 and f >= 3 and m >= 3:
        return '高潜流失客'
    elif intent >= 0.7 and (purchase_power == '高收入' or purchase_power == '中高收入') and m <= 2:
        return '高潜观望用户'
    elif intent >= 0.7 and (purchase_power == '低收入' or purchase_power == '中低收入') and m <= 2:
        return '隐形活跃者'
    elif r >= 3 and f <= 2 and m <= 2:
        return '新客户'
    elif r <= 2 and f <= 2 and m >= 3:
        return '沉睡客户'
    else:
        return '普通用户'

df['User_Segment'] = df.apply(segment_user, axis=1)

segment_counts = df['User_Segment'].value_counts()
print("\n用户分层结果:")
print(segment_counts)

plt.figure(figsize=(12, 6))
segment_counts.plot(kind='bar', color='skyblue')
plt.title('用户分层分布')
plt.xlabel('用户类型')
plt.ylabel('人数')
plt.xticks(rotation=45)
plt.savefig('user_segment_distribution.png')
plt.close()

print("\n=== 各分层用户特征分析 ===")
segment_features = df.groupby('User_Segment').agg({
    'Total_Spending': ['mean', 'median'],
    'Purchase_Frequency': ['mean', 'median'],
    'Intent_Score': ['mean', 'median'],
    'Conversion_Friction': ['mean', 'median'],
    'Active_Connection': ['mean', 'median'],
    'Income': ['mean', 'median']
}).round(2)
print(segment_features)

print("\n=== A/B测试模拟 ===")
print("\n【模拟说明】")
print("以下 A/B 测试是基于历史数据的模拟估算，非真实营销活动结果。")
print("真实 ROI 需要在实际营销活动后，通过对照组实验收集数据得出。")

total_users = df.shape[0]
budget_per_user = 10

print("\n【方案A：传统策略】")
print("选择 RFM 分数最高的 20% 用户")

# 方案A：传统策略 - 选择 RFM 分数最高的 20% 用户
traditional_users = df.nlargest(int(total_users * 0.2), 'RFM_Score')
traditional_cost = len(traditional_users) * budget_per_user

# 基于历史消费数据模拟边际收益
# 假设：传统策略用户本身消费意愿强，优惠券带来的增量有限
traditional_baseline_spending = traditional_users['Total_Spending'].sum()
# 假设优惠券带来 1% 的边际消费提升
traditional_lift = traditional_baseline_spending * 0.01

print(f"  历史总消费: ¥{traditional_baseline_spending:,.0f}")
print(f"  假设边际提升: 1%")
print(f"  目标用户: {len(traditional_users)}人")
print(f"  营销成本: ¥{traditional_cost:,}")

# 方案B：优化策略 - 选择核心用户 + 潜力用户
print("\n【方案B：优化策略】")
print("选择：核心VIP + 重要价值客户 + 高潜观望用户 + 高潜流失客")

optimized_df = df[(df['User_Segment'].isin(['核心VIP', '重要价值客户', '高潜观望用户', '高潜流失客']))]
optimized_users = optimized_df.nlargest(159, 'RFM_Score')
optimized_cost = len(optimized_users) * budget_per_user

# 基于历史消费数据模拟边际收益
# 假设：潜力用户对营销刺激更敏感，边际提升更高
optimized_baseline_spending = optimized_users['Total_Spending'].sum()
# 假设优惠券带来 1% 的边际消费提升（相同假设，便于对比）
optimized_lift = optimized_baseline_spending * 0.01

print(f"  历史总消费: ¥{optimized_baseline_spending:,.0f}")
print(f"  假设边际提升: 1%")
print(f"  目标用户: {len(optimized_users)}人")
print(f"  营销成本: ¥{optimized_cost:,}")

print("\n【结果对比】")
print("-" * 50)
print("方案A（传统策略）:")
print(f"  边际收益: ¥{traditional_lift:,.0f}")
print(f"  边际ROI: {(traditional_lift - traditional_cost) / traditional_cost:.2%}")
print()
print("方案B（优化策略）:")
print(f"  边际收益: ¥{optimized_lift:,.0f}")
print(f"  边际ROI: {(optimized_lift - optimized_cost) / optimized_cost:.2%}")
print("-" * 50)

print("\n=== 保存分析结果 ===")
df.to_csv('user_segmentation_result.csv', index=False, encoding='utf-8-sig')
print("分析结果已保存到 user_segmentation_result.csv")

print("\n=== 分析完成 ===")