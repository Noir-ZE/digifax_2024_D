import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子保证每次运行结果相同
np.random.seed(42)

# 模拟省份列表
provinces = ['Beijing', 'Shanghai', 'Guangdong', 'Zhejiang', 'Sichuan', 'Yunnan', 'Henan', 'Shandong']

# 模拟年份
years = np.arange(1990, 2021)

# 模拟降水量数据 (单位：毫米)
precipitation_data = pd.DataFrame({
    'year': np.repeat(years, len(provinces)),
    'province': np.tile(provinces, len(years)),
    'precipitation_mm': np.random.normal(600, 100, len(provinces) * len(years))
})

# 模拟土地利用数据 (耕地、林地、草地比例)
land_use_types = ['farmland', 'forest', 'grassland']
land_use_data = pd.DataFrame({
    'year': np.repeat(years, len(provinces)),
    'province': np.tile(provinces, len(years)),
    'land_use_type': np.random.choice(land_use_types, len(provinces) * len(years)),
    'percentage': np.random.uniform(0.2, 0.8, len(provinces) * len(years))
})

# 数据显示
print(precipitation_data.head())
print(land_use_data.head())

# ---- 计算降水量统计指标 ----

# 计算每年全国的平均降水量
avg_precipitation = precipitation_data.groupby('year')['precipitation_mm'].mean().reset_index()

# 计算降水量的标准差
std_precipitation = precipitation_data.groupby('year')['precipitation_mm'].std().reset_index()

# 计算年均变化率
avg_precipitation['precipitation_change_rate'] = avg_precipitation['precipitation_mm'].pct_change() * 100

# 可视化全国平均降水量和年均变化率
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(x='year', y='precipitation_mm', data=avg_precipitation, ax=ax1, color='b', label='Average Precipitation')
ax1.set_ylabel('Average Precipitation (mm)', fontsize=12)
ax1.set_xlabel('Year', fontsize=12)

# 第二个y轴绘制年均变化率
ax2 = ax1.twinx()
sns.lineplot(x='year', y='precipitation_change_rate', data=avg_precipitation, ax=ax2, color='r', label='Change Rate')
ax2.set_ylabel('Change Rate (%)', fontsize=12)

plt.title('Average Precipitation and Change Rate in China (1990-2020)', fontsize=14)
fig.tight_layout()
plt.show()

# ---- 计算土地利用类型统计指标 ----

# 计算每年全国各类土地利用类型的平均占比
avg_land_use = land_use_data.groupby(['year', 'land_use_type'])['percentage'].mean().reset_index()

# 计算每类土地利用的年均变化率
avg_land_use['change_rate'] = avg_land_use.groupby('land_use_type')['percentage'].pct_change() * 100

# 计算每类土地利用的标准差
std_land_use = land_use_data.groupby(['year', 'land_use_type'])['percentage'].std().reset_index()

# 可视化土地利用类型的变化
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='percentage', hue='land_use_type', data=avg_land_use, marker='o')
plt.title('Land Use Type Proportions (1990-2020)', fontsize=14)
plt.ylabel('Average Proportion (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(title='Land Use Type', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 计算降水量与土地利用的相关性分析 ----

# 将降水量和土地利用数据合并
merged_data = pd.merge(precipitation_data, land_use_data, on=['year', 'province'])

# 按年份计算降水量与不同土地利用类型的相关性
correlations = merged_data.groupby('year').apply(
    lambda x: x[['precipitation_mm', 'percentage']].corr().iloc[0, 1]
).reset_index(name='correlation')

# 可视化降水量与土地利用相关性的变化
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='correlation', data=correlations, marker='o')
plt.title('Correlation between Precipitation and Land Use (1990-2020)', fontsize=14)
plt.ylabel('Correlation', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 设置全局图表样式 ----

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})