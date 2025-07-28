# import xarray as xr
# import rioxarray
# import pandas as pd
# import numpy as np
# import os
# import time
#
#
# def process_land_use_data():
#     """
#     处理中国大陆0.5°土地利用和覆盖变化数据集。
#
#     功能:
#     1. 读取1990年和2019年的林地(forest)和湿地(wetland)数据。
#     2. 将0.5°的原始数据重采样(插值)到0.25°的目标网格。
#     3. 计算 X7 (2019年林地覆盖率) 和 X8 (2019年湿地覆盖率)。
#     4. 计算 X9 (1990-2019年林地覆盖变化率)。
#     5. 将结果合并并保存为 CSV 和 XLSX 文件。
#     """
#     print("开始处理土地利用数据...")
#     start_time_total = time.time()
#
#     # --- 1. 定义常量 ---
#     BASE_DIR = r"D:\Cursor\cursorpro\2024_D\4. 中国大陆0.5°土地利用和覆盖变化数据集(1900-2019年)\raw_data"
#     OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
#     CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_land_use_variables.csv")
#     XLSX_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_land_use_variables.xlsx")
#
#     # 根据 "问题二代码逻辑_new.md" 定义关键年份和类型
#     START_YEAR = 1990
#     END_YEAR = 2019
#     TARGET_RESOLUTION = 0.25
#     LAND_TYPES = ['forest', 'wetland']  # 我们关心的土地类型
#
#     # --- 2. 准备工作 ---
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     def get_data_path(land_type, year):
#         """辅助函数，用于构建文件路径"""
#         path = os.path.join(BASE_DIR, f"{land_type}-{year}.tif")
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"错误: 输入文件未找到，请检查路径: {path}")
#         return path
#
#     # --- 3. 读取并重采样数据 ---
#     print(f"步骤 1/3: 读取 {START_YEAR} 和 {END_YEAR} 年的土地利用数据并重采样至 {TARGET_RESOLUTION}°...")
#
#     # 读取一个文件作为参考，以定义目标网格
#     try:
#         ref_da = rioxarray.open_rasterio(get_data_path('forest', END_YEAR)).squeeze('band', drop=True)
#     except FileNotFoundError as e:
#         print(e)
#         return
#
#     # 定义目标0.25°网格的坐标
#     min_lon, min_lat, max_lon, max_lat = ref_da.rio.bounds()
#     target_lon = np.arange(min_lon, max_lon, TARGET_RESOLUTION)
#     target_lat = np.arange(min_lat, max_lat, TARGET_RESOLUTION)
#
#     # 存储所有重采样后的数据
#     resampled_data = {}
#
#     for ltype in LAND_TYPES:
#         for year in [START_YEAR, END_YEAR]:
#             print(f"  - 处理: {ltype} {year}...")
#             # 读取原始0.5°数据
#             original_da = rioxarray.open_rasterio(get_data_path(ltype, year)).squeeze('band', drop=True)
#             # 使用线性插值方法重采样到0.25°网格
#             resampled_da = original_da.interp(x=target_lon, y=target_lat, method="linear")
#             resampled_data[f"{ltype}_{year}"] = resampled_da
#
#     print("数据读取和重采样完成。")
#
#     # --- 4. 计算变量 X7, X8, X9 ---
#     print("\n步骤 2/3: 计算变量 X7, X8, X9...")
#
#     # X7: 2019年林地覆盖率
#     x7_forest_cover = resampled_data[f'forest_{END_YEAR}']
#     x7_forest_cover.name = "X7_forest_cover_2019"
#     print("  - X7 (2019年林地覆盖率) 计算完成。")
#
#     # X8: 2019年湿地覆盖率
#     x8_wetland_cover = resampled_data[f'wetland_{END_YEAR}']
#     x8_wetland_cover.name = "X8_wetland_cover_2019"
#     print("  - X8 (2019年湿地覆盖率) 计算完成。")
#
#     # X9: 1990-2019年林地覆盖变化率
#     # 公式: (结束年份覆盖率 - 开始年份覆盖率) / 开始年份覆盖率
#     forest_start = resampled_data[f'forest_{START_YEAR}']
#     forest_end = resampled_data[f'forest_{END_YEAR}']
#
#     # 计算变化率，并处理分母为0的情况
#     # np.divide 会在 0/0 时返回 nan，在 x/0 (x>0) 时返回 inf，我们需要将inf也替换为nan
#     with np.errstate(divide='ignore', invalid='ignore'):
#         x9_forest_change_rate = np.divide((forest_end - forest_start), forest_start)
#
#     x9_forest_change_rate = x9_forest_change_rate.where(np.isfinite(x9_forest_change_rate))  # 将 inf 和 -inf 替换为 NaN
#     x9_forest_change_rate.name = "X9_forest_change_rate_1990_2019"
#     print("  - X9 (林地覆盖变化率) 计算完成。")
#
#     # --- 5. 合并并保存结果 ---
#     print("\n步骤 3/3: 合并所有变量并保存...")
#
#     # 合并所有计算出的变量到一个Dataset中
#     result_ds = xr.merge([x7_forest_cover, x8_wetland_cover, x9_forest_change_rate])
#
#     # 转换为DataFrame
#     df_result = result_ds.to_dataframe().reset_index()
#     df_result = df_result.rename(columns={'x': 'longitude', 'y': 'latitude'})
#
#     # 移除所有变量都为空的行
#     df_result.dropna(subset=list(result_ds.data_vars), how='all', inplace=True)
#
#     # 保存为CSV
#     print(f"正在将结果保存到CSV: {CSV_OUTPUT_FILENAME}")
#     df_result.to_csv(CSV_OUTPUT_FILENAME, index=False, encoding='utf-8')
#
#     # 保存为XLSX
#     print(f"正在将结果保存到XLSX: {XLSX_OUTPUT_FILENAME}")
#     try:
#         df_result.to_excel(XLSX_OUTPUT_FILENAME, index=False, engine='openpyxl')
#     except ImportError:
#         print("\n警告: 未找到 'openpyxl' 库，无法保存为XLSX文件。")
#         print("请运行 'pip install openpyxl' 来安装它。\n")
#
#     end_time_total = time.time()
#     print("-" * 50)
#     print("土地利用数据处理完成！")
#     print(f"总耗时: {end_time_total - start_time_total:.2f} 秒")
#     print(f"生成文件 1: {CSV_OUTPUT_FILENAME}")
#     if os.path.exists(XLSX_OUTPUT_FILENAME):
#         print(f"生成文件 2: {XLSX_OUTPUT_FILENAME}")
#     print("文件预览:")
#     print(df_result.head())
#     print("-" * 50)
#
#
# if __name__ == '__main__':
#     process_land_use_data()

import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import os
import time


def process_topography_data_ultimate_v4():
    """
    处理中国1km数字高程模型(DEM)数据 (终极稳定版 V4 - 性能优化)。
    采用先聚合降维再计算的策略，从根本上解决内存溢出问题。
    """
    print("开始处理地形数据 (终极稳定版 V4 - 性能优化)...")
    start_time_total = time.time()

    # --- 1. 定义常量 ---
    INPUT_FILE_PATH = r"D:\Cursor\cursorpro\2024_D\1. 中国数字高程图(1km)\Geo\TIFF\chinadem_geo.tif"
    GRID_REFERENCE_FILE = r"D:\Pythonpro\2024_D\result\Q2\processed_precipitation_variables.csv"

    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
    CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_topography_variables.csv")

    TARGET_RESOLUTION = 0.25

    # --- 2. 准备工作 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_FILE_PATH) or not os.path.exists(GRID_REFERENCE_FILE):
        print(f"错误: 输入文件未找到，请检查路径。")
        return

    # --- 3. 读取数据并定义目标网格 ---
    print("步骤 1/4: 读取DEM数据和参考网格...")
    dem_1km = rioxarray.open_rasterio(INPUT_FILE_PATH, chunks=True).squeeze('band', drop=True)
    dem_1km = dem_1km.where(dem_1km >= 0)

    ref_df = pd.read_csv(GRID_REFERENCE_FILE)
    target_lon = ref_df['longitude'].unique()
    target_lat = ref_df['latitude'].unique()
    print("目标网格定义完成。")

    # --- 4. 聚合降维与计算 ---
    print("步骤 2/4: 聚合DEM数据以匹配目标分辨率...")

    # 计算聚合因子
    coarsen_factor_x = int(TARGET_RESOLUTION / abs(dem_1km.rio.resolution()[0]))
    coarsen_factor_y = int(TARGET_RESOLUTION / abs(dem_1km.rio.resolution()[1]))

    # 使用coarsen进行聚合，大大减少数据量
    dem_coarse = dem_1km.coarsen(
        x=coarsen_factor_x, y=coarsen_factor_y, boundary='trim'
    ).construct(x=("x_coarse", "x_fine"), y=("y_coarse", "y_fine"))
    print("数据聚合完成。")

    print("步骤 3/4: 在聚合后的数据上计算地形变量...")
    # X1: 平均高程
    x1_mean_elevation = dem_coarse.mean(dim=["x_fine", "y_fine"]).compute()
    x1_mean_elevation.name = "X1_mean_elevation"

    # X2: 地形起伏度
    max_elev = dem_coarse.max(dim=["x_fine", "y_fine"])
    min_elev = dem_coarse.min(dim=["x_fine", "y_fine"])
    x2_relief = (max_elev - min_elev).compute()
    x2_relief.name = "X2_relief"

    # X3: 平均坡度 - 在聚合后的数据上计算
    # 首先获取聚合后每个格网的米制尺寸
    coarse_res_deg = TARGET_RESOLUTION
    pixel_size_meters = coarse_res_deg * 111 * 1000

    # 获取聚合后的高程数据作为numpy数组
    dem_coarse_mean_arr = x1_mean_elevation.values

    grad_y, grad_x = np.gradient(dem_coarse_mean_arr, pixel_size_meters, pixel_size_meters)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    slope_deg_arr = np.degrees(slope_rad)

    # 将计算出的坡度数组放回带坐标的DataArray中
    x3_mean_slope = xr.DataArray(slope_deg_arr, coords=x1_mean_elevation.coords, dims=x1_mean_elevation.dims)
    x3_mean_slope.name = "X3_mean_slope"

    print("所有地形变量计算完成。")

    # --- 5. 对齐到最终网格并保存 ---
    print("步骤 4/4: 将结果对齐到最终网格并保存...")

    # 将计算出的粗糙变量插值到精确的目标网格上
    final_x1 = x1_mean_elevation.interp(x_coarse=target_lon, y_coarse=target_lat, method="linear",
                                        kwargs={"fill_value": "extrapolate"})
    final_x2 = x2_relief.interp(x_coarse=target_lon, y_coarse=target_lat, method="linear",
                                kwargs={"fill_value": "extrapolate"})
    final_x3 = x3_mean_slope.interp(x_coarse=target_lon, y_coarse=target_lat, method="linear",
                                    kwargs={"fill_value": "extrapolate"})

    result_ds = xr.merge([final_x1, final_x2, final_x3])
    df_result = result_ds.to_dataframe().reset_index()

    df_result.rename(columns={'x_coarse': 'longitude', 'y_coarse': 'latitude'}, inplace=True)

    print(f"正在将地形数据更新到: {CSV_OUTPUT_FILENAME}")
    df_result.to_csv(CSV_OUTPUT_FILENAME, index=False)
    print("地形数据更新成功！")
    print("文件预览:")
    print(df_result.head())
    print("-" * 50)


if __name__ == '__main__':
    process_topography_data_ultimate_v4()

