import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import os
import glob
import time
from tqdm import tqdm
import warnings

# Suppress warnings from pmdarima and other libraries
warnings.filterwarnings("ignore")

try:
    import pmdarima
except ImportError:
    print(
        "错误: 'pmdarima' 库未找到。请运行 'pip install pmdarima' 或 'conda install -c conda-forge pmdarima' 来安装它。")
    exit()


def arima_forecast_on_pixel(series):
    """对单个像素的时间序列应用auto_arima并进行预测。"""
    # 移除序列中的NaN值，因为ARIMA无法处理它们
    series = series[~np.isnan(series)]

    # 如果有效数据点太少或为常数，则无法预测
    if len(series) < 10 or len(np.unique(series)) <= 1:
        return np.array([np.nan, np.nan], dtype=np.float32)

    try:
        model = pmdarima.auto_arima(series, start_p=1, start_q=1, test='adf',
                                    max_p=3, max_q=3, m=1, d=None, seasonal=False,
                                    start_P=0, D=0, trace=False, error_action='ignore',
                                    suppress_warnings=True, stepwise=True)
        forecast = model.predict(n_periods=2)
        return np.asarray(forecast, dtype=np.float32)
    except Exception:
        return np.array([np.nan, np.nan], dtype=np.float32)


def preprocess_temp_files(ds):
    """预处理函数，用于在合并前为每个温度文件添加时间坐标。"""
    filename = os.path.basename(ds.encoding["source"])
    date_str = filename.split('_')[0]
    time_coord = pd.to_datetime(date_str, format='%Y%m%d')
    ds = ds.assign_coords(time=time_coord)
    if 'band' in ds.coords:
        ds = ds.squeeze('band').drop_vars('band')
    return ds


def process_temperature_data():
    """处理中国0.1°近地表气温数据集。"""
    print("开始处理气温数据...")
    start_time_total = time.time()

    BASE_DIR = r"D:\Cursor\cursorpro\2024_D\2. 气温日平均数据"
    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\Q2"
    CSV_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_temperature_variables.csv")
    XLSX_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "processed_temperature_variables.xlsx")
    HISTORICAL_YEARS = range(1990, 2019)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"步骤 1/4: 计算 {HISTORICAL_YEARS.start}-{HISTORICAL_YEARS.stop - 1} 年的夏季平均气温...")
    yearly_summer_means = []

    for year in tqdm(HISTORICAL_YEARS, desc="处理历史年份"):
        summer_files = []
        for month in [6, 7, 8]:
            path_pattern = os.path.join(BASE_DIR, f"{year}_avg", f"{year}{month:02d}*_avg.tif")
            summer_files.extend(glob.glob(path_pattern))

        if not summer_files:
            print(f"\n警告: 未找到 {year} 年的夏季文件，跳过该年。")
            continue

        ds_year_summer = xr.open_mfdataset(summer_files, engine="rasterio", concat_dim="time",
                                           combine="nested", preprocess=preprocess_temp_files)

        mean_summer_temp = ds_year_summer.mean(dim="time", keep_attrs=True)
        yearly_summer_means.append(mean_summer_temp)

    historical_summer_means_ds = xr.concat(yearly_summer_means,
                                           dim=pd.to_datetime([f'{y}-01-01' for y in HISTORICAL_YEARS])).rename(
        {'concat_dim': 'time'})

    data_var_name = list(historical_summer_means_ds.data_vars)[0]
    historical_summer_means_da = historical_summer_means_ds[data_var_name]

    print("历史夏季均温计算完成。")

    print("\n步骤 2/4: 使用ARIMA模型预测2019-2020年夏季均温...")
    print("（此步骤将采用手动循环，以确保稳定性，可能需要很长时间...）")

    # 【最终修正】采用手动循环，放弃apply_ufunc
    # 将xarray数据加载到内存中的numpy数组中
    historical_data_np = historical_summer_means_da.to_numpy()

    # 获取维度信息
    num_years, num_lat, num_lon = historical_data_np.shape

    # 创建一个空的numpy数组来存储预测结果
    forecast_np = np.full((2, num_lat, num_lon), np.nan, dtype=np.float32)

    # 使用tqdm来显示循环进度
    for y_idx in tqdm(range(num_lat), desc="逐行像素预测"):
        for x_idx in range(num_lon):
            # 提取单个像素的时间序列
            pixel_series = historical_data_np[:, y_idx, x_idx]

            # 如果整个序列都是NaN，则跳过计算
            if np.all(np.isnan(pixel_series)):
                continue

            # 进行预测
            forecast_result = arima_forecast_on_pixel(pixel_series)

            # 将结果存入数组
            forecast_np[:, y_idx, x_idx] = forecast_result

    # 将numpy结果转换回xarray.DataArray
    forecast_da = xr.DataArray(
        forecast_np,
        dims=('time', 'y', 'x'),
        coords={'time': pd.to_datetime(['2019-01-01', '2020-01-01']),
                'y': historical_summer_means_da.y,
                'x': historical_summer_means_da.x}
    )

    predicted_2019 = forecast_da.sel(time='2019-01-01')
    predicted_2020 = forecast_da.sel(time='2020-01-01')

    print("预测完成。")

    print("\n步骤 3/4: 合并数据并计算1990-2020年总平均值 (X5)...")

    full_period_means = xr.concat([historical_summer_means_da, predicted_2019, predicted_2020], dim="time")

    x5_01_deg = full_period_means.mean(dim='time', keep_attrs=True)
    x5_01_deg.rio.write_crs(historical_summer_means_da.rio.crs, inplace=True)

    print("开始重采样至0.25°网格...")
    min_lon, min_lat, max_lon, max_lat = x5_01_deg.rio.bounds()
    target_lon = np.arange(np.floor(min_lon * 4) / 4, np.ceil(max_lon * 4) / 4, 0.25)
    target_lat = np.arange(np.floor(min_lat * 4) / 4, np.ceil(max_lat * 4) / 4, 0.25)

    x5_025_deg = x5_01_deg.interp(x=target_lon, y=target_lat, method="linear")
    x5_025_deg.name = "X5_mean_summer_temp"

    print("重采样完成。")

    print("\n步骤 4/4: 整理并保存结果...")

    df_result = x5_025_deg.to_dataframe().reset_index()
    df_result = df_result.rename(columns={'x': 'longitude', 'y': 'latitude'})
    df_result = df_result.dropna(subset=['X5_mean_summer_temp'])

    print(f"正在将结果保存到CSV: {CSV_OUTPUT_FILENAME}")
    df_result.to_csv(CSV_OUTPUT_FILENAME, index=False, encoding='utf-8')

    print(f"正在将结果保存到XLSX: {XLSX_OUTPUT_FILENAME}")
    try:
        df_result.to_excel(XLSX_OUTPUT_FILENAME, index=False, engine='openpyxl')
    except ImportError:
        print("\n警告: 未找到 'openpyxl' 库，无法保存为XLSX文件。")
        print("请运行 'pip install openpyxl' 来安装它。\n")

    end_time_total = time.time()
    print("-" * 50)
    print("气温数据处理完成！")
    print(f"总耗时: {(end_time_total - start_time_total) / 60:.2f} 分钟")
    print(f"生成文件 1: {CSV_OUTPUT_FILENAME}")
    if os.path.exists(XLSX_OUTPUT_FILENAME):
        print(f"生成文件 2: {XLSX_OUTPUT_FILENAME}")
    print("文件预览:")
    print(df_result.head())
    print("-" * 50)


if __name__ == '__main__':
    process_temperature_data()