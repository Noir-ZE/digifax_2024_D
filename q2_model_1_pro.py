# import pandas as pd
# import geopandas as gpd
# import xarray as xr
# import os
# import time
#
#
# def merge_via_advanced_interpolation():
#     """
#     使用高级插值（Advanced Interpolation）方法，将降水和地形数据
#     精确采样到气温数据的坐标点上，以最高效率和最低内存消耗完成数据融合。
#     """
#     print("模块1: 最终数据预处理与合并 (V-Ultimate: 高级插值版)")
#     start_time_total = time.time()
#
#     # --- 1. 定义常量 ---
#     BASE_DIR = r"D:\Pythonpro\2024_D\result\Q2"
#     TEMP_FILE = os.path.join(BASE_DIR, "processed_temperature_variables.csv")
#     PRECIP_FILE = os.path.join(BASE_DIR, "processed_precipitation_variables.csv")
#     TOPO_FILE = os.path.join(BASE_DIR, "processed_topography_variables.csv")
#
#     FINAL_OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\final_processed_data"
#     GPKG_OUTPUT_FILENAME = os.path.join(FINAL_OUTPUT_DIR, "analysis_data_full.gpkg")
#     CSV_OUTPUT_FILENAME = os.path.join(FINAL_OUTPUT_DIR, "analysis_data_full.csv")
#
#     # --- 2. 准备工作 ---
#     os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
#
#     # --- 3. 加载基准数据和目标坐标 ---
#     print("\n步骤 1/3: 加载基准数据 (气温)...")
#     df_base = pd.read_csv(TEMP_FILE)
#     print(f"基准数据已加载，目标样本量: {len(df_base)}")
#
#     # 【核心修正】创建用于高级索引的坐标DataArray
#     # 我们为这15355个点创建一个共享的维度 'location'
#     locations = pd.Index(range(len(df_base)), name='location')
#     target_lon = xr.DataArray(df_base['longitude'], dims=['location'], coords={'location': locations})
#     target_lat = xr.DataArray(df_base['latitude'], dims=['location'], coords={'location': locations})
#
#     # --- 4. 加载源数据并进行高级插值 ---
#     print("\n步骤 2/3: 对降水和地形数据进行高级插值采样...")
#
#     def sample_at_points(file_path, lon_da, lat_da, lon_col='longitude', lat_col='latitude'):
#         """
#         读取源CSV，构建xarray网格，并使用高级索引在目标点上采样。
#         """
#         print(f"  - 采样文件: {os.path.basename(file_path)}")
#         df_source = pd.read_csv(file_path)
#         ds_source = df_source.set_index([lat_col, lon_col]).to_xarray()
#
#         # 使用高级插值
#         ds_sampled = ds_source.interp(longitude=lon_da, latitude=lat_da, method="linear")
#
#         # 返回采样后的Dataset
#         return ds_sampled
#
#     # 采样降水数据
#     ds_precip_sampled = sample_at_points(PRECIP_FILE, target_lon, target_lat)
#
#     # 采样地形数据
#     ds_topo_sampled = sample_at_points(TOPO_FILE, target_lon, target_lat)
#
#     print("数据采样完成。")
#
#     # --- 5. 合并最终数据 ---
#     print("\n步骤 3/3: 合并采样数据并保存...")
#
#     # 直接将采样到的数据作为新列添加到基准DataFrame中
#     # .values会提取纯numpy数组，确保维度匹配
#     final_df = df_base.copy()
#     for var_name in ds_precip_sampled.data_vars:
#         final_df[var_name] = ds_precip_sampled[var_name].values
#
#     for var_name in ds_topo_sampled.data_vars:
#         final_df[var_name] = ds_topo_sampled[var_name].values
#
#     # 清理不必要的列
#     final_df.drop(columns=[col for col in final_df.columns if 'spatial_ref' in col], inplace=True, errors='ignore')
#
#     # 清洗插值过程中可能产生的NaN
#     all_vars = [col for col in final_df.columns if col.startswith(('X', 'Y'))]
#     final_df.dropna(subset=all_vars, how='any', inplace=True)
#
#     print(f"合并与清洗完成。最终用于建模的完整格网数量: {len(final_df)}")
#
#     # --- 6. 创建GeoDataFrame并保存 ---
#     if len(final_df) > 0:
#         gdf = gpd.GeoDataFrame(
#             final_df,
#             geometry=gpd.points_from_xy(final_df.longitude, final_df.latitude),
#             crs="EPSG:4326"
#         )
#
#         print(f"正在保存到 GeoPackage: {GPKG_OUTPUT_FILENAME}")
#         gdf.to_file(GPKG_OUTPUT_FILENAME, driver='GPKG')
#
#         print(f"正在保存到 CSV: {CSV_OUTPUT_FILENAME}")
#         final_df.to_csv(CSV_OUTPUT_FILENAME, index=False)
#
#         print("\n文件预览:")
#         print(gdf.head())
#     else:
#         print("\n错误: 最终数据集为空。")
#
#     end_time_total = time.time()
#     print("-" * 50)
#     print("模块1: 数据预处理与特征工程全部完成！")
#     print(f"总耗时: {(end_time_total - start_time_total):.2f} 秒")
#     print(f"最终生成的建模数据文件: {GPKG_OUTPUT_FILENAME}")
#     print("-" * 50)
#
#
# if __name__ == '__main__':
#     merge_via_advanced_interpolation()

import pandas as pd
import geopandas as gpd
import os
import time


def final_merge():
    """
    合并三个已经完全对齐的CSV文件，并将最终结果保存为
    GPKG, CSV, 和 XLSX 三种格式。

    V2 版本: 增加了 CSV 和 XLSX 格式的输出。
    """
    print("模块1 - 子任务: 最终合并 (V2)")
    start_time = time.time()

    # --- 1. 定义路径 ---
    BASE_DIR = r"D:\Pythonpro\2024_D\result\Q2"
    TEMP_FILE = os.path.join(BASE_DIR, "processed_temperature_variables.csv")
    PRECIP_FILE = os.path.join(BASE_DIR, "processed_precipitation_variables_aligned.csv")
    TOPO_FILE = os.path.join(BASE_DIR, "processed_topography_variables_aligned.csv")

    FINAL_OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\final_processed_data"
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

    # 定义所有输出文件的路径
    GPKG_OUTPUT_FILENAME = os.path.join(FINAL_OUTPUT_DIR, "analysis_data_final.gpkg")
    CSV_OUTPUT_FILENAME = os.path.join(FINAL_OUTPUT_DIR, "analysis_data_final.csv")
    XLSX_OUTPUT_FILENAME = os.path.join(FINAL_OUTPUT_DIR, "analysis_data_final.xlsx")

    # --- 2. 读取并合并 ---
    print("步骤 1/3: 读取已对齐的数据...")
    try:
        df_temp = pd.read_csv(TEMP_FILE)
        df_precip = pd.read_csv(PRECIP_FILE)
        df_topo = pd.read_csv(TOPO_FILE)
    except FileNotFoundError as e:
        print(f"\n错误: 关键输入文件未找到: {e}")
        print("请确保您已经成功运行了 'process_topography_aligned_v2.py' 和 'process_precipitation_aligned_v2.py'。")
        return

    print("步骤 2/3: 合并数据...")
    # 因为坐标完全一致，可以直接使用merge
    final_df = pd.merge(df_temp, df_precip, on=['longitude', 'latitude'], how='left')
    final_df = pd.merge(final_df, df_topo, on=['longitude', 'latitude'], how='left')

    # --- 3. 清洗并保存 ---
    print("步骤 3/3: 清洗并保存最终数据...")
    print(f"合并前样本量: {len(final_df)}")

    # 清理在插值过程中可能产生的NaN值（例如，基准点落在了DEM数据范围之外）
    final_df.dropna(subset=['X1_mean_elevation', 'X2_relief', 'X3_mean_slope'], inplace=True)
    print(f"清洗后，最终样本量: {len(final_df)}")

    if len(final_df) > 0:
        # 保存为 CSV (使用普通的DataFrame)
        print(f"\n正在将结果保存到 CSV: {CSV_OUTPUT_FILENAME}")
        final_df.to_csv(CSV_OUTPUT_FILENAME, index=False, encoding='utf-8')

        # 保存为 XLSX (使用普通的DataFrame)
        print(f"正在将结果保存到 XLSX: {XLSX_OUTPUT_FILENAME}")
        try:
            # engine='openpyxl' 是写入.xlsx格式所必需的
            final_df.to_excel(XLSX_OUTPUT_FILENAME, index=False, engine='openpyxl')
        except Exception as e:
            print(f"警告: 无法保存为XLSX文件。请确保您已安装 'openpyxl' (`pip install openpyxl`)。错误: {e}")

        # 创建GeoDataFrame并保存为GPKG
        gdf = gpd.GeoDataFrame(
            final_df,
            geometry=gpd.points_from_xy(final_df.longitude, final_df.latitude),
            crs="EPSG:4326"
        )
        print(f"正在将结果保存到 GeoPackage: {GPKG_OUTPUT_FILENAME}")
        gdf.to_file(GPKG_OUTPUT_FILENAME, driver='GPKG')

        print("\n所有文件保存成功！")
        print("文件预览 (来自GeoDataFrame):")
        print(gdf.head())
    else:
        print("\n错误: 最终数据集为空，无法保存文件。")

    end_time = time.time()
    print("-" * 50)
    print("模块1 - 最终合并任务全部完成！")
    print(f"总耗时: {(end_time - start_time):.2f} 秒")
    print(f"文件已输出至目录: {FINAL_OUTPUT_DIR}")
    print("-" * 50)


if __name__ == '__main__':
    final_merge()