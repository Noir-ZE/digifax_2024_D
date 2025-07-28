import pandas as pd
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import os
import time


def run_modeling_and_analysis():
    """
    主函数，执行模块3的全部建模与分析任务，包括
    阶段一的Logistic回归和阶段二的GWR。
    """
    print("模块3: 核心建模与分析")
    start_time_total = time.time()

    # --- 1. 定义常量和路径 ---
    INPUT_DIR = r"D:\Pythonpro\2024_D\result\final_processed_data"
    GPKG_FILE = os.path.join(INPUT_DIR, "analysis_data_final.gpkg")

    OUTPUT_DIR_TABLES = r"D:\Pythonpro\2024_D\result\tables"
    OUTPUT_DIR_DATA = r"D:\Pythonpro\2024_D\result\gwr_results"
    os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)

    LOGISTIC_RESULTS_FILE = os.path.join(OUTPUT_DIR_TABLES, "logistic_regression_summary.csv")
    GWR_RESULTS_FILE = os.path.join(OUTPUT_DIR_DATA, "gwr_results.gpkg")

    # --- 2. 加载并准备数据 ---
    print("\n步骤 1/7: 加载数据...")
    if not os.path.exists(GPKG_FILE):
        print(f"错误: 输入数据文件未找到: {GPKG_FILE}")
        return

    gdf = gpd.read_file(GPKG_FILE)
    print(f"数据加载成功，共有 {len(gdf)} 个格网。")

    # 定义自变量和因变量
    y1_var = 'Y1_heavy_rain_prob'
    y2_var = 'Y2_heavy_rain_intensity'
    x_vars = ['X1_mean_elevation', 'X2_relief', 'X3_mean_slope', 'X5_mean_summer_temp', 'X6_mean_annual_precip']

    # 数据标准化
    print("步骤 2/7: 标准化自变量...")
    scaler = StandardScaler()
    gdf[x_vars] = scaler.fit_transform(gdf[x_vars])
    print("自变量标准化完成。")

    # --- 3. 阶段一：Logistic 回归 ---
    print("\n--- 开始阶段一: Logistic 回归 ---")

    # 准备Logit模型数据
    # 将Y1从[0,100]的概率转换为[0,1]的比例
    y_logit = gdf[y1_var] / 100.0
    X_logit = gdf[x_vars]
    X_logit = sm.add_constant(X_logit)  # statsmodels需要手动添加截距项

    print("步骤 3/7: 拟合Logistic回归模型...")
    # 使用广义线性模型(GLM)中的Binomial族，可以很好地处理比率/概率数据
    logit_model = sm.GLM(y_logit, X_logit, family=sm.families.Binomial())
    logit_results = logit_model.fit()
    print("模型拟合完成。")

    print("步骤 4/7: 保存Logistic回归结果...")
    # 提取结果摘要并保存为CSV
    summary_df = logit_results.summary2().tables[1]
    summary_df.to_csv(LOGISTIC_RESULTS_FILE)
    print(f"Logistic回归结果已保存至: {LOGISTIC_RESULTS_FILE}")
    print("\nLogistic Regression Model Summary:")
    print(summary_df)

    # --- 4. 阶段二：地理加权回归 (GWR) ---
    print("\n--- 开始阶段二: 地理加权回归 (GWR) ---")

    # 准备GWR数据
    # 筛选出Y2（暴雨强度）不为空的行
    gdf_gwr = gdf.dropna(subset=[y2_var]).copy()
    print(f"步骤 5/7: 准备GWR数据... 用于GWR的样本量: {len(gdf_gwr)}")

    # 准备GWR模型的输入变量
    y_gwr = gdf_gwr[y2_var].values.reshape(-1, 1)
    X_gwr = gdf_gwr[x_vars].values
    # 提取坐标
    coords_gwr = list(zip(gdf_gwr.geometry.x, gdf_gwr.geometry.y))

    print("步骤 6/7: 寻找GWR最优带宽... (这可能需要几分钟甚至更长时间)")
    # 使用Sel_BW寻找最优带宽，AICc是常用的标准
    selector = Sel_BW(coords_gwr, y_gwr, X_gwr, spherical=True)  # spherical=True因为我们用的是经纬度
    best_bandwidth = selector.search()
    print(f"最优带宽寻找完成: {best_bandwidth}")

    print("步骤 7/7: 拟合GWR模型并保存结果...")
    gwr_model = GWR(coords_gwr, y_gwr, X_gwr, bw=best_bandwidth, spherical=True)
    gwr_results = gwr_model.fit()
    print("GWR模型拟合完成。")

    # 将GWR结果添加回GeoDataFrame
    gdf_gwr['gwr_pred'] = gwr_results.predy.flatten()
    gdf_gwr['gwr_local_R2'] = gwr_results.localR2
    # 为每个自变量的局部系数创建一个新列
    for i, var in enumerate(x_vars):
        gdf_gwr[f'gwr_coeff_{var}'] = gwr_results.params[:, i]

    # 保存包含GWR结果的GeoPackage文件
    gdf_gwr.to_file(GWR_RESULTS_FILE, driver='GPKG')
    print(f"GWR结果已保存至: {GWR_RESULTS_FILE}")
    print("\nGWR Results Preview (first 5 rows):")
    print(gdf_gwr[['geometry', 'gwr_pred', 'gwr_local_R2'] + [f'gwr_coeff_{var}' for var in x_vars]].head())

    end_time_total = time.time()
    print("-" * 50)
    print("模块3: 核心建模与分析全部完成！")
    print(f"总耗时: {(end_time_total - start_time_total) / 60:.2f} 分钟")
    print("-" * 50)


if __name__ == '__main__':
    run_modeling_and_analysis()
