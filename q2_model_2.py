# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
# import time
# from shapely.geometry import LineString
#
#
# def parse_gmt_file(file_path, encoding='GB2312'):
#     """
#     解析.gmt地理数据文件，并将其转换为GeoDataFrame。
#     """
#     lines = []
#     with open(file_path, 'r', encoding=encoding) as f:
#         segment_points = []
#         for line in f:
#             if line.startswith('>'):
#                 if segment_points:
#                     lines.append(LineString(segment_points))
#                 segment_points = []
#             else:
#                 try:
#                     lon, lat = map(float, line.split())
#                     segment_points.append((lon, lat))
#                 except ValueError:
#                     continue
#         if segment_points:  # 添加最后一个线段
#             lines.append(LineString(segment_points))
#
#     return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
#
#
# def plot_spatial_distribution(gdf, column_name, china_border, output_path, title, cmap='viridis'):
#     """
#     为指定列绘制专业的空间分布图。
#     """
#     print(f"  - Plotting: {title}...")
#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
#
#     # 绘制中国边界作为底图
#     china_border.plot(ax=ax, edgecolor='black', linewidth=0.8)
#
#     # 绘制我们的数据点，使用列的值进行着色
#     gdf.plot(column=column_name, ax=ax, legend=True,
#              markersize=5, cmap=cmap,
#              legend_kwds={'label': column_name, 'orientation': "horizontal", 'pad': 0.05, 'shrink': 0.6})
#
#     # 设置图表美学
#     ax.set_title(title, fontsize=16, weight='bold')
#     ax.set_xlabel("Longitude", fontsize=12)
#     ax.set_ylabel("Latitude", fontsize=12)
#     ax.set_aspect('equal')
#     ax.grid(True, linestyle='--', alpha=0.6)
#
#     # 设置显示范围，聚焦于中国大陆
#     ax.set_xlim(70, 140)
#     ax.set_ylim(15, 55)
#
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)  # 关闭图表以释放内存
#
#
# def plot_correlation_heatmap(df, columns, output_path):
#     """
#     计算并绘制相关系数矩阵的热力图。
#     """
#     print(f"  - Plotting: Correlation Heatmap...")
#     corr = df[columns].corr()
#
#     fig, ax = plt.subplots(figsize=(12, 10))
#
#     # 使用seaborn绘制热力图
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
#                 square=True, linewidths=.5, cbar_kws={"shrink": .8})
#
#     ax.set_title('Correlation Matrix of Variables', fontsize=16, weight='bold')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#
#
# def exploratory_data_analysis():
#     """
#     执行模块2的探索性数据分析任务。
#     """
#     print("模块2: 探索性数据分析")
#     start_time_total = time.time()
#
#     # --- 1. 定义常量和路径 ---
#     INPUT_DIR = r"D:\Pythonpro\2024_D\result\final_processed_data"
#     GPKG_FILE = os.path.join(INPUT_DIR, "analysis_data_simplified.gpkg")
#
#     CHINA_MAP_DIR = r"D:\Cursor\cursorpro\2024_D\china-geospatial-data-GB2312"
#     CHINA_BORDER_FILE = os.path.join(CHINA_MAP_DIR, "CN-border-La.gmt")  # 使用带省界的版本
#
#     OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\figures"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     # --- 2. 加载数据 ---
#     print("\n步骤 1/3: 加载数据...")
#     if not os.path.exists(GPKG_FILE) or not os.path.exists(CHINA_BORDER_FILE):
#         print("错误: 输入文件未找到，请检查路径。")
#         return
#
#     gdf = gpd.read_file(GPKG_FILE)
#     china_border = parse_gmt_file(CHINA_BORDER_FILE)
#     print(f"数据加载成功，共有 {len(gdf)} 个有效格网。")
#
#     # --- 3. 绘制空间分布图 ---
#     print("\n步骤 2/3: 绘制并保存空间分布图...")
#
#     # 定义要可视化的变量及其图表标题和配色方案
#     variables_to_plot = {
#         'Y1_heavy_rain_prob': {'title': 'Spatial Distribution of Heavy Rain Probability (Y1)', 'cmap': 'Blues'},
#         'Y2_heavy_rain_intensity': {'title': 'Spatial Distribution of Heavy Rain Intensity (Y2)', 'cmap': 'Reds'},
#         'X1_mean_elevation': {'title': 'Spatial Distribution of Mean Elevation (X1)', 'cmap': 'terrain'},
#         'X2_relief': {'title': 'Spatial Distribution of Topographic Relief (X2)', 'cmap': 'plasma'},
#         'X3_mean_slope': {'title': 'Spatial Distribution of Mean Slope (X3)', 'cmap': 'magma'},
#         'X5_mean_summer_temp': {'title': 'Spatial Distribution of Mean Summer Temperature (X5)', 'cmap': 'YlOrRd'},
#         'X6_mean_annual_precip': {'title': 'Spatial Distribution of Mean Annual Precipitation (X6)', 'cmap': 'GnBu'}
#     }
#
#     for col, settings in variables_to_plot.items():
#         if col in gdf.columns:
#             # 对于Y2，它包含很多NaN，只绘制有值的点
#             plot_gdf = gdf if col != 'Y2_heavy_rain_intensity' else gdf.dropna(subset=[col])
#             output_file = os.path.join(OUTPUT_DIR, f"spatial_dist_{col}.png")
#             plot_spatial_distribution(plot_gdf, col, china_border, output_file, settings['title'], settings['cmap'])
#
#     print("所有空间分布图已生成。")
#
#     # --- 4. 绘制相关性热力图 ---
#     print("\n步骤 3/3: 绘制并保存相关性热力图...")
#
#     # 只选择数值型变量进行相关性分析
#     # 对于Y2，我们只能在它有值的子集上计算相关性
#     df_for_corr = gdf.drop(columns='geometry').dropna(subset=['Y2_heavy_rain_intensity'])
#
#     # 重命名列以在图表中获得更简洁的标签
#     rename_dict = {
#         'Y1_heavy_rain_prob': 'Y1_Prob',
#         'Y2_heavy_rain_intensity': 'Y2_Intensity',
#         'X1_mean_elevation': 'X1_Elevation',
#         'X2_relief': 'X2_Relief',
#         'X3_mean_slope': 'X3_Slope',
#         'X5_mean_summer_temp': 'X5_Temp',
#         'X6_mean_annual_precip': 'X6_Precip'
#     }
#     df_for_corr.rename(columns=rename_dict, inplace=True)
#
#     output_file_corr = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
#     plot_correlation_heatmap(df_for_corr, list(rename_dict.values()), output_file_corr)
#     print("相关性热力图已生成。")
#
#     end_time_total = time.time()
#     print("-" * 50)
#     print("模块2: 探索性数据分析全部完成！")
#     print(f"总耗时: {(end_time_total - start_time_total):.2f} 秒")
#     print(f"所有图表已保存至: {OUTPUT_DIR}")
#     print("-" * 50)
#
#
# if __name__ == '__main__':
#     exploratory_data_analysis()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from shapely.geometry import LineString


def parse_gmt_file(file_path, encoding='GB2312'):
    """
    解析.gmt地理数据文件（如中国的国界和省界），并将其转换为可供绘图的GeoDataFrame。

    Args:
        file_path (str): .gmt文件的路径。
        encoding (str): 文件编码，根据您的要求设置为'GB2312'。

    Returns:
        gpd.GeoDataFrame: 包含地理边界线的GeoDataFrame。
    """
    lines = []
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            segment_points = []
            for line in f:
                # '>' 符号标志着一个新线段的开始
                if line.strip().startswith('>'):
                    if segment_points:
                        lines.append(LineString(segment_points))
                    segment_points = []
                else:
                    try:
                        lon, lat = map(float, line.split())
                        segment_points.append((lon, lat))
                    except ValueError:
                        # 忽略无法解析的行
                        continue
            if segment_points:  # 添加文件末尾的最后一个线段
                lines.append(LineString(segment_points))
    except Exception as e:
        print(f"错误: 无法解析GMT文件 '{file_path}'. 错误信息: {e}")
        return None

    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def plot_spatial_distribution(gdf, column_name, china_border, output_path, title, cmap='viridis'):
    """
    为指定的数据列绘制专业、美观的空间分布图。
    """
    print(f"  - 正在绘制: {title}...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

    # 1. 绘制中国边界作为底图
    china_border.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1)

    # 2. 绘制我们的数据点，使用列的值进行着色
    # 使用 'cividis' 'viridis' 'plasma' 等连续色带效果较好
    gdf.plot(column=column_name, ax=ax, legend=True,
             markersize=8, cmap=cmap,
             legend_kwds={'label': f"Value of {column_name}", 'orientation': "horizontal", 'pad': 0.05, 'shrink': 0.6},
             zorder=2)

    # 3. 设置图表美学
    ax.set_title(title, fontsize=18, weight='bold', pad=15)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 4. 设置显示范围，聚焦于中国大陆
    ax.set_xlim(72, 136)
    ax.set_ylim(18, 54)

    # 5. 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图表以释放内存


def plot_correlation_heatmap(df, columns, output_path):
    """
    计算并绘制相关系数矩阵的热力图。
    """
    print(f"  - 正在绘制: Correlation Heatmap...")
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用seaborn绘制热力图，'coolwarm'色带很适合展示正负相关
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 10})

    ax.set_title('Correlation Matrix of Variables', fontsize=18, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def exploratory_data_analysis():
    """
    主函数，执行模块2的全部探索性数据分析任务。
    """
    print("模块2: 探索性数据分析 (EDA)")
    start_time_total = time.time()

    # --- 1. 定义常量和路径 ---
    INPUT_DIR = r"D:\Pythonpro\2024_D\result\final_processed_data"
    GPKG_FILE = os.path.join(INPUT_DIR, "analysis_data_final.gpkg")

    CHINA_MAP_DIR = r"D:\Cursor\cursorpro\2024_D\china-geospatial-data-GB2312"
    CHINA_BORDER_FILE = os.path.join(CHINA_MAP_DIR, "CN-border-La.gmt")  # 使用带省界的版本

    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 加载数据 ---
    print("\n步骤 1/3: 加载数据...")
    if not os.path.exists(GPKG_FILE):
        print(f"错误: 输入数据文件未找到: {GPKG_FILE}")
        return
    if not os.path.exists(CHINA_BORDER_FILE):
        print(f"错误: 中国边界文件未找到: {CHINA_BORDER_FILE}")
        return

    gdf = gpd.read_file(GPKG_FILE)
    china_border = parse_gmt_file(CHINA_BORDER_FILE)
    print(f"数据加载成功，共有 {len(gdf)} 个有效格网。")

    # --- 3. 绘制空间分布图 ---
    print("\n步骤 2/3: 绘制并保存各变量的空间分布图...")

    # 定义要可视化的变量及其图表标题和配色方案
    variables_to_plot = {
        'Y1_heavy_rain_prob': {'title': 'Spatial Distribution of Heavy Rain Probability (Y1)', 'cmap': 'Blues'},
        'Y2_heavy_rain_intensity': {'title': 'Spatial Distribution of Heavy Rain Intensity (Y2)', 'cmap': 'Reds'},
        'X1_mean_elevation': {'title': 'Spatial Distribution of Mean Elevation (X1)', 'cmap': 'terrain'},
        'X2_relief': {'title': 'Spatial Distribution of Topographic Relief (X2)', 'cmap': 'plasma'},
        'X3_mean_slope': {'title': 'Spatial Distribution of Mean Slope (X3)', 'cmap': 'magma'},
        'X5_mean_summer_temp': {'title': 'Spatial Distribution of Mean Summer Temperature (X5)', 'cmap': 'YlOrRd'},
        'X6_mean_annual_precip': {'title': 'Spatial Distribution of Mean Annual Precipitation (X6)', 'cmap': 'GnBu'}
    }

    for col, settings in variables_to_plot.items():
        if col in gdf.columns:
            # 对于Y2，它包含很多NaN，只绘制有值的点以获得有意义的视图
            plot_gdf = gdf if 'Y2' not in col else gdf.dropna(subset=[col])
            output_file = os.path.join(OUTPUT_DIR, f"spatial_dist_{col}.png")
            plot_spatial_distribution(plot_gdf, col, china_border, output_file, settings['title'], settings['cmap'])

    print("所有空间分布图已生成。")

    # --- 4. 绘制相关性热力图 ---
    print("\n步骤 3/3: 绘制并保存相关性热力图...")

    # 只选择数值型变量进行相关性分析
    # 对于Y2，我们只能在它有值的子集上计算相关性，这样结果才有意义
    df_for_corr = gdf.drop(columns='geometry').dropna(subset=['Y2_heavy_rain_intensity'])

    # 重命名列以在图表中获得更简洁的标签
    rename_dict = {
        'Y1_heavy_rain_prob': 'Y1_Prob',
        'Y2_heavy_rain_intensity': 'Y2_Intensity',
        'X1_mean_elevation': 'X1_Elevation',
        'X2_relief': 'X2_Relief',
        'X3_mean_slope': 'X3_Slope',
        'X5_mean_summer_temp': 'X5_Temp',
        'X6_mean_annual_precip': 'X6_Precip'
    }
    df_for_corr.rename(columns=rename_dict, inplace=True)

    output_file_corr = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plot_correlation_heatmap(df_for_corr, list(rename_dict.values()), output_file_corr)
    print("相关性热力图已生成。")

    end_time_total = time.time()
    print("-" * 50)
    print("模块2: 探索性数据分析全部完成！")
    print(f"总耗时: {(end_time_total - start_time_total):.2f} 秒")
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("-" * 50)


if __name__ == '__main__':
    exploratory_data_analysis()