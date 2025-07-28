import geopandas as gpd
import matplotlib.pyplot as plt
import os
import time
from shapely.geometry import LineString


def parse_gmt_file(file_path, encoding='GB2312'):
    """
    解析.gmt地理数据文件（如中国的国界和省界），并将其转换为可供绘图的GeoDataFrame。
    """
    lines = []
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            segment_points = []
            for line in f:
                if line.strip().startswith('>'):
                    if segment_points:
                        lines.append(LineString(segment_points))
                    segment_points = []
                else:
                    try:
                        lon, lat = map(float, line.split())
                        segment_points.append((lon, lat))
                    except ValueError:
                        continue
            if segment_points:
                lines.append(LineString(segment_points))
    except Exception as e:
        print(f"错误: 无法解析GMT文件 '{file_path}'. 错误信息: {e}")
        return None

    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def plot_gwr_map(gdf, column_name, china_border, output_path, title, cmap='viridis', is_coefficient=False):
    """
    为GWR结果的指定列（系数或诊断值）绘制专业的空间分布图。
    """
    print(f"  - 正在绘制: {title}...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

    # 绘制中国边界作为底图
    china_border.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=1)

    # 为系数图选择一个发散的色带，中心为0
    if is_coefficient:
        # 'RdBu_r' 红-白-蓝, r代表反转，使得正值为红，负值为蓝
        cmap = 'RdBu_r'
        # 找到绝对值的最大值，以使色带对称
        vmax = gdf[column_name].abs().max()
        vmin = -vmax
    else:
        vmin = gdf[column_name].min()
        vmax = gdf[column_name].max()

    # 绘制GWR结果数据点
    gdf.plot(column=column_name, ax=ax, legend=True,
             markersize=10, cmap=cmap, vmin=vmin, vmax=vmax,
             legend_kwds={'label': f"Value of {column_name}", 'orientation': "horizontal", 'pad': 0.05, 'shrink': 0.6},
             zorder=2)

    # 设置图表美学
    ax.set_title(title, fontsize=18, weight='bold', pad=15)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置显示范围
    ax.set_xlim(72, 136)
    ax.set_ylim(18, 54)

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_gwr_results():
    """
    主函数，执行模块4的GWR模型结果可视化任务。
    """
    print("模块4: 模型结果可视化")
    start_time_total = time.time()

    # --- 1. 定义常量和路径 ---
    INPUT_DIR = r"D:\Pythonpro\2024_D\result\gwr_results"
    GPKG_FILE = os.path.join(INPUT_DIR, "gwr_results.gpkg")

    CHINA_MAP_DIR = r"D:\Cursor\cursorpro\2024_D\china-geospatial-data-GB2312"
    CHINA_BORDER_FILE = os.path.join(CHINA_MAP_DIR, "CN-border-La.gmt")

    OUTPUT_DIR = r"D:\Pythonpro\2024_D\result\figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 加载数据 ---
    print("\n步骤 1/3: 加载GWR结果数据...")
    if not os.path.exists(GPKG_FILE):
        print(f"错误: GWR结果文件未找到: {GPKG_FILE}")
        return

    gdf_gwr = gpd.read_file(GPKG_FILE)
    china_border = parse_gmt_file(CHINA_BORDER_FILE)
    print(f"数据加载成功，共有 {len(gdf_gwr)} 个格网的GWR结果。")

    # --- 3. 绘制模型诊断图 ---
    print("\n步骤 2/3: 绘制模型诊断图 (Local R-squared)...")
    plot_gwr_map(
        gdf=gdf_gwr,
        column_name='gwr_local_R2',
        china_border=china_border,
        output_path=os.path.join(OUTPUT_DIR, 'gwr_local_R2_distribution.png'),
        title='Spatial Distribution of GWR Local R-squared',
        cmap='plasma'  # 使用一个暖色调的连续色带
    )
    print("Local R-squared 地图绘制完成。")

    # --- 4. 绘制各变量的局部系数图 ---
    print("\n步骤 3/3: 绘制各变量的局部系数空间分布图...")

    # 自动查找所有系数列
    coeff_columns = [col for col in gdf_gwr.columns if col.startswith('gwr_coeff_')]

    # 为方便阅读，创建变量名的映射
    var_name_map = {
        'gwr_coeff_X1_mean_elevation': 'Mean Elevation (X1)',
        'gwr_coeff_X2_relief': 'Topographic Relief (X2)',
        'gwr_coeff_X3_mean_slope': 'Mean Slope (X3)',
        'gwr_coeff_X5_mean_summer_temp': 'Mean Summer Temp (X5)',
        'gwr_coeff_X6_mean_annual_precip': 'Mean Annual Precip (X6)'
    }

    for col in coeff_columns:
        if col in var_name_map:
            clean_name = var_name_map[col]
            plot_gwr_map(
                gdf=gdf_gwr,
                column_name=col,
                china_border=china_border,
                output_path=os.path.join(OUTPUT_DIR, f'{col}_distribution.png'),
                title=f'GWR Coefficient Distribution for {clean_name}',
                is_coefficient=True  # 告知函数这是一个系数图，以使用红蓝分异色带
            )
        else:
            print(f"警告: 系数列 {col} 没有在 var_name_map 中找到对应的名称，将跳过。")

    print("所有系数地图绘制完成。")

    end_time_total = time.time()
    print("-" * 50)
    print("模块4: 模型结果可视化全部完成！")
    print(f"总耗时: {(end_time_total - start_time_total):.2f} 秒")
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("-" * 50)


if __name__ == '__main__':
    visualize_gwr_results()