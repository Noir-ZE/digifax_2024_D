# import xarray as xr
# import pandas as pd
# import numpy as np
# import os
# import time
#
#
# def process_precipitation_data_v2():
#     """
#     处理中国大陆0.25°逐日降水数据集(1961-2022年) - V2。
#
#     功能:
#     1. 读取NetCDF文件。
#     2. 将所有负值降水量数据修正为0。
#     3. 筛选1990-01-01至2020-12-31的时间范围。
#     4. 计算因变量 Y1 (年均暴雨发生概率) 和 Y2 (年均暴雨强度)。
#     5. 计算自变量 X6 (年均总降水量)。
#     6. 将结果保存为 CSV 和 XLSX 文件。
#     """
#     print("开始处理降水数据 (V2)...")
#
#     # --- 1. 定义常量 ---
#     # !! 请根据您的实际文件路径修改 !!
#     INPUT_FILE_PATH = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
#
#     # !! 请根据您的实际输出路径修改 !!
#     OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
#     CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_precipitation_variables.csv")
#     XLSX_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_precipitation_variables.xlsx")
#
#     START_DATE = '1990-01-01'
#     END_DATE = '2020-12-31'
#     HEAVY_RAIN_THRESHOLD = 50  # 暴雨阈值 (mm/day)
#
#     # --- 2. 准备工作 ---
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     if not os.path.exists(INPUT_FILE_PATH):
#         print(f"错误: 输入文件未找到，请检查路径: {INPUT_FILE_PATH}")
#         return
#
#     # --- 3. 读取数据 ---
#     print(f"正在从 {INPUT_FILE_PATH} 读取数据...")
#     start_time = time.time()
#
#     try:
#         ds = xr.open_dataset(INPUT_FILE_PATH, chunks={'time': 365})
#         if 'pre' not in ds.variables:
#             print(f"错误: 数据集中未找到名为 'pre' 的变量。可用变量为: {list(ds.variables.keys())}")
#             return
#     except Exception as e:
#         print(f"读取NetCDF文件时出错: {e}")
#         return
#
#     # --- 4. 数据清洗与筛选 ---
#     print("数据读取完毕，开始筛选时间范围...")
#     ds_period = ds.sel(time=slice(START_DATE, END_DATE))
#
#     # 【新功能】将所有负值降水量修正为0
#     print("正在将所有负值降水量修正为0...")
#     ds_period['pre'] = ds_period['pre'].clip(min=0)
#
#     total_days = len(ds_period.time)
#     total_years = (pd.to_datetime(END_DATE).year - pd.to_datetime(START_DATE).year) + 1
#     print(f"数据时间范围: {START_DATE} 到 {END_DATE}，共 {total_days} 天 ({total_years} 年)。")
#
#     # --- 5. 计算变量 ---
#     print("开始计算 Y1, Y2, 和 X6...")
#
#     is_heavy_rain = ds_period['pre'] >= HEAVY_RAIN_THRESHOLD
#     heavy_rain_days = is_heavy_rain.sum(dim='time')
#
#     # Y1: 年均暴雨发生概率
#     y1_prob = heavy_rain_days / total_days
#     y1_prob.name = 'Y1_heavy_rain_prob'
#
#     # Y2: 年均暴雨强度
#     heavy_rain_precipitation = ds_period['pre'].where(is_heavy_rain)  # 只保留暴雨日的值
#     total_heavy_rain_precip = heavy_rain_precipitation.sum(dim='time')
#     y2_intensity = np.divide(total_heavy_rain_precip, heavy_rain_days)
#     y2_intensity = xr.DataArray(y2_intensity, coords=heavy_rain_days.coords, name='Y2_heavy_rain_intensity')
#
#     # X6: 年均总降水量
#     total_precip = ds_period['pre'].sum(dim='time')
#     x6_mean_annual_precip = total_precip / total_years
#     x6_mean_annual_precip.name = 'X6_mean_annual_precip'
#
#     # --- 6. 合并结果并保存 ---
#     print("计算完成，正在合并结果...")
#
#     result_ds = xr.merge([y1_prob, y2_intensity, x6_mean_annual_precip])
#     df_result = result_ds.to_dataframe().reset_index()
#     df_result = df_result.rename(columns={'lon': 'longitude', 'lat': 'latitude'})
#
#     # 保存为CSV
#     print(f"正在将结果保存到CSV: {CSV_OUTPUT_FILENAME}")
#     df_result.to_csv(CSV_OUTPUT_FILENAME, index=False, encoding='utf-8')
#
#     # 【新功能】保存为XLSX
#     print(f"正在将结果保存到XLSX: {XLSX_OUTPUT_FILENAME}")
#     try:
#         df_result.to_excel(XLSX_OUTPUT_FILENAME, index=False, engine='openpyxl')
#     except ImportError:
#         print("\n警告: 未找到 'openpyxl' 库，无法保存为XLSX文件。")
#         print("请运行 'pip install openpyxl' 来安装它。\n")
#
#     end_time = time.time()
#     print("-" * 50)
#     print("降水数据处理完成！")
#     print(f"总耗时: {end_time - start_time:.2f} 秒")
#     print(f"生成文件 1: {CSV_OUTPUT_FILENAME}")
#     if os.path.exists(XLSX_OUTPUT_FILENAME):
#         print(f"生成文件 2: {XLSX_OUTPUT_FILENAME}")
#     print("文件预览:")
#     print(df_result.head())
#     print("-" * 50)
#
#
# if __name__ == '__main__':
#     process_precipitation_data_v2()

import pandas as pd
import xarray as xr
import numpy as np  # <-- 【修正】添加了这行关键的导入语句
import os
import time


def process_precipitation_aligned_to_template():
    """
    读取NetCDF降水数据，并将其直接插值到由一个外部文件定义的基准网格上。

    V-Final.2 版本: 修正了 'np' is not defined 的错误。
    """
    print("模块1 - 子任务: 处理降水数据 (V-Final.2: 基准对齐版)")
    start_time_total = time.time()

    # --- 1. 定义常量 ---
    PRECIP_INPUT_FILE = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
    GRID_TEMPLATE_FILE = r"D:\Pythonpro\2024_D\result\Q2\processed_temperature_variables.csv"

    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
    CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_precipitation_variables_aligned.csv")

    TIME_RANGE = slice('1990-01-01', '2019-12-31')
    HEAVY_RAIN_THRESHOLD = 50

    # --- 2. 加载基准网格和源数据 ---
    print("步骤 1/4: 加载基准网格和原始降水数据...")
    df_template = pd.read_csv(GRID_TEMPLATE_FILE)
    print(f"基准网格已确立，目标样本量: {len(df_template)}")

    locations = pd.Index(range(len(df_template)), name='location')
    target_lon = xr.DataArray(df_template['longitude'], dims=['location'], coords={'location': locations})
    target_lat = xr.DataArray(df_template['latitude'], dims=['location'], coords={'location': locations})

    ds_precip_raw = xr.open_dataset(PRECIP_INPUT_FILE).sel(time=TIME_RANGE)

    # --- 3. 插值原始数据到基准网格 ---
    print("步骤 2/4: 插值原始降水数据到基准网格... (这可能需要几分钟)")
    ds_aligned = ds_precip_raw.interp(longitude=target_lon, latitude=target_lat, method="linear")
    print("插值完成。")

    # --- 4. 在对齐后的数据上计算变量 ---
    print("步骤 3/4: 计算Y1, Y2, X6...")

    heavy_rain_days = (ds_aligned['pre'] >= HEAVY_RAIN_THRESHOLD).sum(dim='time')
    total_days = len(ds_aligned['time'])
    y1_prob = (heavy_rain_days / total_days) * 100
    y1_prob.name = 'Y1_heavy_rain_prob'

    heavy_rain_data = ds_aligned['pre'].where(ds_aligned['pre'] >= HEAVY_RAIN_THRESHOLD)
    y2_intensity = heavy_rain_data.mean(dim='time')
    y2_intensity.name = 'Y2_heavy_rain_intensity'

    total_precip = ds_aligned['pre'].sum(dim='time')
    num_years = len(np.unique(ds_aligned['time.year']))
    x6_annual_precip = total_precip / num_years
    x6_annual_precip.name = 'X6_mean_annual_precip'

    # --- 5. 创建并保存对齐后的DataFrame ---
    print("步骤 4/4: 创建并保存对齐后的降水数据...")
    df_precip_aligned = pd.DataFrame({
        'longitude': target_lon.values,
        'latitude': target_lat.values,
        'Y1_heavy_rain_prob': y1_prob.values,
        'Y2_heavy_rain_intensity': y2_intensity.values,
        'X6_mean_annual_precip': x6_annual_precip.values
    })

    df_precip_aligned.to_csv(CSV_OUTPUT_FILENAME, index=False)

    print("-" * 50)
    print("降水数据对齐完成！")
    print(f"文件已保存至: {CSV_OUTPUT_FILENAME}")
    print("文件预览:")
    print(df_precip_aligned.head())
    print("-" * 50)


if __name__ == '__main__':
    process_precipitation_aligned_to_template()