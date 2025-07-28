# import xarray as xr
# import rioxarray
# import pandas as pd
# import numpy as np
# import os
# import time
# from tqdm import tqdm
#
# try:
#     import richdem as rd
# except ImportError:
#     print("错误: 'richdem' 库未找到。请运行 'pip install richdem' 或 'conda install -c conda-forge richdem' 来安装它。")
#     exit()
#
#
# def process_topography_data():
#     """
#     处理中国1km数字高程模型(DEM)数据。
#     """
#     print("开始处理地形数据...")
#     start_time_total = time.time()
#
#     # --- 1. 定义常量 ---
#     INPUT_FILE_PATH = r"D:\Cursor\cursorpro\2024_D\1. 中国数字高程图(1km)\Geo\TIFF\chinadem_geo.tif"
#     OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
#     CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_topography_variables.csv")
#     XLSX_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_topography_variables.xlsx")
#     TARGET_RESOLUTION = 0.25
#     TARGET_PROJECTION_PROJ_STRING = "+proj=aea +lat_0=0 +lon_0=105 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
#
#     # --- 2. 准备工作 ---
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     if not os.path.exists(INPUT_FILE_PATH):
#         print(f"错误: 输入文件未找到，请检查路径: {INPUT_FILE_PATH}")
#         return
#
#     # --- 3. 读取并预处理DEM数据 ---
#     print(f"步骤 1/4: 从 {INPUT_FILE_PATH} 读取DEM数据...")
#     dem_1km = rioxarray.open_rasterio(INPUT_FILE_PATH, chunks={'x': 512, 'y': 512}).squeeze('band', drop=True)
#     dem_1km = dem_1km.where(dem_1km >= 0)
#     dem_1km.name = "elevation"
#     print("DEM数据读取完成。")
#
#     # --- 4. 聚合计算X1(平均高程) 和 X2(地形起伏度) ---
#     print(f"步骤 2/4: 聚合计算 {TARGET_RESOLUTION}° 网格的平均高程(X1)和地形起伏度(X2)...")
#
#     coarsen_factor_x = int(TARGET_RESOLUTION / abs(dem_1km.rio.resolution()[0]))
#     coarsen_factor_y = int(TARGET_RESOLUTION / abs(dem_1km.rio.resolution()[1]))
#
#     coarsened = dem_1km.coarsen(x=coarsen_factor_x, y=coarsen_factor_y, boundary='trim').construct(
#         x=("x_win", "x_fine"), y=("y_win", "y_fine"))
#
#     print("  - 正在计算 X1 (平均高程)...")
#     x1_mean_elevation = coarsened.mean(dim=["x_fine", "y_fine"]).compute()
#     x1_mean_elevation.name = "X1_mean_elevation"
#     df_x1 = x1_mean_elevation.to_dataframe().reset_index()
#     # 【修正】重命名由 .construct() 方法产生的坐标列
#     df_x1.rename(columns={'y_win': 'y', 'x_win': 'x'}, inplace=True)
#
#     print("  - 正在计算 X2 (地形起伏度)...")
#     max_elev = coarsened.max(dim=["x_fine", "y_fine"])
#     min_elev = coarsened.min(dim=["x_fine", "y_fine"])
#     x2_relief = (max_elev - min_elev).compute()
#     x2_relief.name = "X2_relief"
#     df_x2 = x2_relief.to_dataframe().reset_index()
#     # 【修正】重命名由 .construct() 方法产生的坐标列
#     df_x2.rename(columns={'y_win': 'y', 'x_win': 'x'}, inplace=True)
#
#     print("X1 和 X2 计算完成。")
#
#     # --- 5. 计算坡度并聚合得到X3 ---
#     print("步骤 3/4: 计算坡度(X3)...")
#     print(f"  - 正在将DEM重投影到投影坐标系 (Albers Equal Area)...")
#     dem_1km_proj = dem_1km.rio.reproject(TARGET_PROJECTION_PROJ_STRING)
#
#     print("  - 正在使用richdem计算1km分辨率下的坡度...")
#     dem_arr = dem_1km_proj.to_numpy()
#     rd_dem = rd.rdarray(dem_arr, no_data=np.nan)
#     slope_1km_arr = rd.TerrainAttribute(rd_dem, attrib='slope_degrees')
#
#     slope_1km = xr.DataArray(data=slope_1km_arr, coords=dem_1km_proj.coords, dims=dem_1km_proj.dims)
#     slope_1km.rio.write_crs(dem_1km_proj.rio.crs, inplace=True)
#
#     print("  - 正在将坡度图层重投影回WGS84...")
#     slope_1km_wgs84 = slope_1km.rio.reproject_match(dem_1km)
#
#     print(f"  - 正在将坡度聚合到 {TARGET_RESOLUTION}° 网格...")
#     x3_mean_slope = slope_1km_wgs84.coarsen(x=coarsen_factor_x, y=coarsen_factor_y, boundary='trim').mean().compute()
#     x3_mean_slope.name = "X3_mean_slope"
#     df_x3 = x3_mean_slope.to_dataframe().reset_index()
#
#     print("X3 计算完成。")
#
#     # --- 6. 合并并保存结果 ---
#     print("\n步骤 4/4: 合并所有变量并保存...")
#
#     print("  - 正在使用pandas合并X1, X2, X3...")
#     df_result = pd.merge(df_x1, df_x2, on=['y', 'x'], how='outer')
#     df_result = pd.merge(df_result, df_x3, on=['y', 'x'], how='outer')
#
#     df_result = df_result.rename(columns={'x': 'longitude', 'y': 'latitude'})
#     df_result.dropna(subset=["X1_mean_elevation", "X2_relief", "X3_mean_slope"], how='all', inplace=True)
#
#     print(f"正在将结果保存到CSV: {CSV_OUTPUT_FILENAME}")
#     df_result.to_csv(CSV_OUTPUT_FILENAME, index=False, encoding='utf-8')
#
#     print(f"正在将结果保存到XLSX: {XLSX_OUTPUT_FILENAME}")
#     try:
#         df_result.to_excel(XLSX_OUTPUT_FILENAME, index=False, engine='openpyxl')
#     except ImportError:
#         print("\n警告: 未找到 'openpyxl' 库，无法保存为XLSX文件。")
#         print("请运行 'pip install openpyxl' 来安装它。\n")
#
#     end_time_total = time.time()
#     print("-" * 50)
#     print("地形数据处理完成！")
#     print(f"总耗时: {(end_time_total - start_time_total) / 60:.2f} 分钟")
#     print(f"生成文件 1: {CSV_OUTPUT_FILENAME}")
#     if os.path.exists(XLSX_OUTPUT_FILENAME):
#         print(f"生成文件 2: {XLSX_OUTPUT_FILENAME}")
#     print("文件预览:")
#     print(df_result.head())
#     print("-" * 50)
#
#
# if __name__ == '__main__':
#     process_topography_data()

import pandas as pd
import xarray as xr
import rioxarray
import numpy as np  # <-- 【修正】添加了这行关键的导入语句
import os
import time


def process_topography_aligned_to_template():
    """
    读取高分辨率DEM，并将其直接插值到由一个外部文件定义的基准网格上。

    V-Final.2 版本: 修正了 'np' is not defined 的错误。
    """
    print("模块1 - 子任务: 处理地形数据 (V-Final.2: 基准对齐版)")
    start_time_total = time.time()

    # --- 1. 定义常量 ---
    DEM_INPUT_FILE = r"D:\Cursor\cursorpro\2024_D\1. 中国数字高程图(1km)\Geo\TIFF\chinadem_geo.tif"
    GRID_TEMPLATE_FILE = r"D:\Pythonpro\2024_D\result\Q2\processed_temperature_variables.csv"

    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
    CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_topography_variables_aligned.csv")

    # --- 2. 准备工作 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DEM_INPUT_FILE) or not os.path.exists(GRID_TEMPLATE_FILE):
        print("错误: 输入文件或模板文件未找到，请检查路径。")
        return

    # --- 3. 加载基准网格和源数据 ---
    print("步骤 1/3: 加载基准网格和高分辨率DEM...")
    df_template = pd.read_csv(GRID_TEMPLATE_FILE)
    print(f"基准网格已确立，目标样本量: {len(df_template)}")

    locations = pd.Index(range(len(df_template)), name='location')
    target_lon = xr.DataArray(df_template['longitude'], dims=['location'], coords={'location': locations})
    target_lat = xr.DataArray(df_template['latitude'], dims=['location'], coords={'location': locations})

    dem_1km = rioxarray.open_rasterio(DEM_INPUT_FILE, chunks=True).squeeze('band', drop=True)
    dem_1km = dem_1km.where(dem_1km >= 0)

    # --- 4. 在基准点上插值地形变量 ---
    print("步骤 2/3: 在基准点上插值所有地形变量...")

    print("  - 计算 X1: 平均高程...")
    x1_interp = dem_1km.interp(x=target_lon, y=target_lat, method="linear")

    half_res = 0.25 / 2.0
    x2_relief_values = []
    x3_slope_values = []

    print("  - 计算 X2 (地形起伏度) 和 X3 (平均坡度)... (这可能需要一些时间)")
    for i, row in df_template.iterrows():
        lon, lat = row['longitude'], row['latitude']

        subset = dem_1km.sel(x=slice(lon - half_res, lon + half_res), y=slice(lat + half_res, lat - half_res))
        subset_vals = subset.values
        subset_vals = subset_vals[~np.isnan(subset_vals)]

        if subset_vals.size > 1:
            relief = np.ptp(subset_vals)
            x2_relief_values.append(relief)

            grad_y, grad_x = np.gradient(subset.values)
            slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
            mean_slope_deg = np.nanmean(np.degrees(slope_rad))
            x3_slope_values.append(mean_slope_deg)
        else:
            x2_relief_values.append(np.nan)
            x3_slope_values.append(np.nan)

        if (i + 1) % 1000 == 0:
            print(f"    ...已处理 {i + 1}/{len(df_template)} 个点")

    # --- 5. 创建并保存对齐后的DataFrame ---
    print("步骤 3/3: 创建并保存对齐后的地形数据...")
    df_topo_aligned = pd.DataFrame({
        'longitude': df_template['longitude'],
        'latitude': df_template['latitude'],
        'X1_mean_elevation': x1_interp.values,
        'X2_relief': x2_relief_values,
        'X3_mean_slope': x3_slope_values
    })

    df_topo_aligned.to_csv(CSV_OUTPUT_FILENAME, index=False)

    print("-" * 50)
    print("地形数据对齐完成！")
    print(f"文件已保存至: {CSV_OUTPUT_FILENAME}")
    print("文件预览:")
    print(df_topo_aligned.head())
    print("-" * 50)


if __name__ == '__main__':
    process_topography_aligned_to_template()