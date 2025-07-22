import netCDF4 as nc
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class PrecipitationLandUsePreprocessor:
    """
    统一的数据预处理模块：处理降水量和土地利用数据的时空对齐
    """

    def __init__(self, precip_file, landuse_dir, output_dir):
        """
        初始化预处理器

        Parameters:
        -----------
        precip_file : str
            降水数据NC文件路径
        landuse_dir : str
            土地利用数据目录路径
        output_dir : str
            输出目录路径
        """
        self.precip_file = precip_file
        self.landuse_dir = landuse_dir
        self.output_dir = output_dir

        # 分析时间范围：1990-2019 (30年重叠期)
        self.analysis_years = list(range(1990, 2020))
        self.analysis_period = "1990-2019"

        # 土地利用类型
        self.landuse_types = {
            'cropland': 'Cropland',
            'forest': 'Forest',
            'grass': 'Grassland',
            'shrub': 'Shrubland',
            'wetland': 'Wetland'
        }

        # 目标空间分辨率 (统一到0.5度)
        self.target_resolution = 0.5

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 数据容器
        self.precip_data = None
        self.landuse_data = None
        self.aligned_precip = None
        self.aligned_landuse = None
        self.target_lon = None
        self.target_lat = None

    def load_precipitation_data(self):
        """加载并预处理降水数据"""
        print("=" * 60)
        print("LOADING PRECIPITATION DATA")
        print("=" * 60)

        try:
            # 使用xarray加载netCDF数据
            ds = xr.open_dataset(self.precip_file)

            print(f"Original precipitation data info:")
            print(f"  Time range: {ds.time.dt.year.min().values} - {ds.time.dt.year.max().values}")
            print(f"  Spatial shape: {ds.pre.shape}")
            print(f"  Longitude: {ds.longitude.min().values:.2f}° - {ds.longitude.max().values:.2f}°")
            print(f"  Latitude: {ds.latitude.min().values:.2f}° - {ds.latitude.max().values:.2f}°")
            print(f"  Original resolution: ~{abs(ds.longitude[1] - ds.longitude[0]).values:.3f}°")

            # 筛选时间范围到1990-2019
            ds_filtered = ds.sel(time=slice(f'{self.analysis_years[0]}-01-01',
                                            f'{self.analysis_years[-1]}-12-31'))

            print(f"\nFiltered to analysis period {self.analysis_period}:")
            print(f"  Time points: {len(ds_filtered.time)}")
            print(f"  Years: {ds_filtered.time.dt.year.min().values} - {ds_filtered.time.dt.year.max().values}")

            # 处理缺失值
            precip_array = ds_filtered.pre.values
            precip_array[precip_array < 0] = 0  # 将负值设为0

            # 计算年降水量
            print("\nCalculating annual precipitation...")
            years = ds_filtered.time.dt.year.values
            unique_years = np.unique(years)

            annual_precip = np.zeros((len(unique_years),
                                      len(ds_filtered.latitude),
                                      len(ds_filtered.longitude)))

            for i, year in enumerate(unique_years):
                year_mask = years == year
                annual_sum = np.sum(precip_array[year_mask], axis=0)
                annual_precip[i] = annual_sum
                print(
                    f"  Year {year}: {np.sum(year_mask)} days, sum range: {np.nanmin(annual_sum):.1f} - {np.nanmax(annual_sum):.1f} mm")

            # 存储降水数据
            self.precip_data = {
                'data': annual_precip,  # [year, lat, lon]
                'years': unique_years,
                'longitude': ds_filtered.longitude.values,
                'latitude': ds_filtered.latitude.values,
                'resolution': abs(ds_filtered.longitude[1] - ds_filtered.longitude[0]).values
            }

            ds.close()

            print(f"✅ Precipitation data loaded successfully")
            print(f"   Shape: {annual_precip.shape} (years × lat × lon)")

            return True

        except Exception as e:
            print(f"❌ Error loading precipitation data: {e}")
            return False

    def load_landuse_data(self):
        """加载并预处理土地利用数据"""
        print("\n" + "=" * 60)
        print("LOADING LAND USE DATA")
        print("=" * 60)

        try:
            # 扫描土地利用文件
            tif_files = glob.glob(os.path.join(self.landuse_dir, "*.tif"))
            print(f"Found {len(tif_files)} TIF files")

            # 组织文件
            file_map = {}
            for tif_file in tif_files:
                filename = os.path.basename(tif_file)

                # 解析文件名
                for landuse_type in self.landuse_types.keys():
                    if filename.startswith(landuse_type + '-'):
                        import re
                        year_match = re.search(r'-(\d{4})\.tif$', filename)
                        if year_match:
                            year = int(year_match.group(1))
                            if year in self.analysis_years:
                                if year not in file_map:
                                    file_map[year] = {}
                                file_map[year][landuse_type] = tif_file
                        break

            print(f"Organized files for {len(file_map)} years in analysis period")

            # 加载空间参考信息
            sample_file = None
            for year_files in file_map.values():
                if year_files:
                    sample_file = list(year_files.values())[0]
                    break

            if not sample_file:
                raise ValueError("No valid files found for analysis period")

            # 从样本文件获取空间信息
            with rasterio.open(sample_file) as src:
                bounds = src.bounds
                transform = src.transform
                width, height = src.width, src.height

                print(f"Land use spatial info:")
                print(f"  Dimensions: {width} × {height}")
                print(f"  Bounds: {bounds}")
                print(f"  Resolution: ~{abs(transform[0]):.3f}°")

                # 创建坐标数组
                x_coords = [bounds.left + (i + 0.5) * transform[0] for i in range(width)]
                y_coords = [bounds.top + (j + 0.5) * transform[4] for j in range(height)]

                landuse_lon = np.array(x_coords)
                landuse_lat = np.array(y_coords)

            # 加载所有年份的土地利用数据
            n_years = len(self.analysis_years)
            n_types = len(self.landuse_types)

            landuse_array = np.full((n_years, n_types, height, width), np.nan)

            print(f"\nLoading land use data:")

            for i, year in enumerate(self.analysis_years):
                print(f"  Processing year {year}...")

                if year in file_map:
                    for j, landuse_type in enumerate(self.landuse_types.keys()):
                        if landuse_type in file_map[year]:
                            try:
                                with rasterio.open(file_map[year][landuse_type]) as src:
                                    data = src.read(1)
                                    # 处理NoData和异常值
                                    if src.nodata is not None:
                                        data[data == src.nodata] = np.nan
                                    data = np.clip(data, 0.0, 1.0)  # 确保比例在0-1之间
                                    landuse_array[i, j] = data

                            except Exception as e:
                                print(f"    ❌ Error loading {landuse_type}: {e}")
                        else:
                            print(f"    ⚠️  Missing {landuse_type} data for {year}")
                else:
                    print(f"    ⚠️  No data for year {year}")

            # 存储土地利用数据
            self.landuse_data = {
                'data': landuse_array,  # [year, type, lat, lon]
                'years': np.array(self.analysis_years),
                'types': list(self.landuse_types.keys()),
                'longitude': landuse_lon,
                'latitude': landuse_lat,
                'resolution': abs(transform[0])
            }

            # 数据质量报告
            valid_data_count = np.sum(~np.isnan(landuse_array))
            total_data_count = landuse_array.size

            print(f"✅ Land use data loaded successfully")
            print(f"   Shape: {landuse_array.shape} (years × types × lat × lon)")
            print(
                f"   Valid data: {valid_data_count}/{total_data_count} ({100 * valid_data_count / total_data_count:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Error loading land use data: {e}")
            return False

    def create_common_grid(self):
        """创建统一的空间网格"""
        print("\n" + "=" * 60)
        print("CREATING COMMON SPATIAL GRID")
        print("=" * 60)

        # 获取两个数据集的空间范围
        precip_lon_min, precip_lon_max = self.precip_data['longitude'].min(), self.precip_data['longitude'].max()
        precip_lat_min, precip_lat_max = self.precip_data['latitude'].min(), self.precip_data['latitude'].max()

        landuse_lon_min, landuse_lon_max = self.landuse_data['longitude'].min(), self.landuse_data['longitude'].max()
        landuse_lat_min, landuse_lat_max = self.landuse_data['latitude'].min(), self.landuse_data['latitude'].max()

        print(f"Precipitation grid:")
        print(f"  Longitude: {precip_lon_min:.2f}° - {precip_lon_max:.2f}°")
        print(f"  Latitude: {precip_lat_min:.2f}° - {precip_lat_max:.2f}°")
        print(f"  Resolution: {self.precip_data['resolution']:.3f}°")

        print(f"Land use grid:")
        print(f"  Longitude: {landuse_lon_min:.2f}° - {landuse_lon_max:.2f}°")
        print(f"  Latitude: {landuse_lat_min:.2f}° - {landuse_lat_max:.2f}°")
        print(f"  Resolution: {self.landuse_data['resolution']:.3f}°")

        # 计算重叠区域
        overlap_lon_min = max(precip_lon_min, landuse_lon_min)
        overlap_lon_max = min(precip_lon_max, landuse_lon_max)
        overlap_lat_min = max(precip_lat_min, landuse_lat_min)
        overlap_lat_max = min(precip_lat_max, landuse_lat_max)

        print(f"\nOverlap region:")
        print(f"  Longitude: {overlap_lon_min:.2f}° - {overlap_lon_max:.2f}°")
        print(f"  Latitude: {overlap_lat_min:.2f}° - {overlap_lat_max:.2f}°")

        # 创建目标网格 (使用0.5度分辨率)
        self.target_lon = np.arange(overlap_lon_min, overlap_lon_max + self.target_resolution,
                                    self.target_resolution)
        self.target_lat = np.arange(overlap_lat_max, overlap_lat_min - self.target_resolution,
                                    -self.target_resolution)

        print(f"\nTarget grid (resolution: {self.target_resolution}°):")
        print(
            f"  Longitude points: {len(self.target_lon)} ({self.target_lon.min():.2f}° - {self.target_lon.max():.2f}°)")
        print(
            f"  Latitude points: {len(self.target_lat)} ({self.target_lat.min():.2f}° - {self.target_lat.max():.2f}°)")
        print(
            f"  Total grid cells: {len(self.target_lon)} × {len(self.target_lat)} = {len(self.target_lon) * len(self.target_lat)}")

        return True

    def regrid_precipitation_data(self):
        """将降水数据重采样到目标网格"""
        print("\n" + "=" * 60)
        print("REGRIDDING PRECIPITATION DATA")
        print("=" * 60)

        try:
            n_years = len(self.precip_data['years'])
            n_target_lat = len(self.target_lat)
            n_target_lon = len(self.target_lon)

            # 初始化输出数组
            regridded_precip = np.zeros((n_years, n_target_lat, n_target_lon))

            # 创建源网格坐标
            src_lon_grid, src_lat_grid = np.meshgrid(
                self.precip_data['longitude'],
                self.precip_data['latitude']
            )

            # 创建目标网格坐标
            target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)

            print(f"Regridding from {self.precip_data['data'].shape[1:]} to {(n_target_lat, n_target_lon)}...")

            for year_idx in range(n_years):
                year = self.precip_data['years'][year_idx]

                # 获取当年数据
                src_data = self.precip_data['data'][year_idx]

                # 展平坐标和数据用于插值
                src_points = np.column_stack((
                    src_lon_grid.ravel(),
                    src_lat_grid.ravel()
                ))
                src_values = src_data.ravel()

                # 去除NaN值
                valid_mask = ~np.isnan(src_values)
                src_points = src_points[valid_mask]
                src_values = src_values[valid_mask]

                target_points = np.column_stack((
                    target_lon_grid.ravel(),
                    target_lat_grid.ravel()
                ))

                # 使用线性插值
                interpolated = griddata(
                    src_points, src_values, target_points,
                    method='linear', fill_value=0
                )

                # 重塑为网格
                regridded_precip[year_idx] = interpolated.reshape(n_target_lat, n_target_lon)

                if year_idx % 5 == 0:
                    print(f"  Processed year {year} ({year_idx + 1}/{n_years})")

            self.aligned_precip = regridded_precip

            print(f"✅ Precipitation regridding completed")
            print(f"   Output shape: {regridded_precip.shape}")
            print(f"   Value range: {np.nanmin(regridded_precip):.1f} - {np.nanmax(regridded_precip):.1f} mm/year")

            return True

        except Exception as e:
            print(f"❌ Error regridding precipitation data: {e}")
            return False

    def regrid_landuse_data(self):
        """将土地利用数据重采样到目标网格"""
        print("\n" + "=" * 60)
        print("REGRIDDING LAND USE DATA")
        print("=" * 60)

        try:
            n_years = len(self.analysis_years)
            n_types = len(self.landuse_types)
            n_target_lat = len(self.target_lat)
            n_target_lon = len(self.target_lon)

            # 初始化输出数组
            regridded_landuse = np.zeros((n_years, n_types, n_target_lat, n_target_lon))

            # 创建源网格坐标
            src_lon_grid, src_lat_grid = np.meshgrid(
                self.landuse_data['longitude'],
                self.landuse_data['latitude']
            )

            # 创建目标网格坐标
            target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)

            print(f"Regridding from {self.landuse_data['data'].shape[2:]} to {(n_target_lat, n_target_lon)}...")

            for year_idx in range(n_years):
                for type_idx, landuse_type in enumerate(self.landuse_types.keys()):

                    # 获取当年当类型数据
                    src_data = self.landuse_data['data'][year_idx, type_idx]

                    # 检查是否有有效数据
                    if np.all(np.isnan(src_data)):
                        print(f"  ⚠️  No valid data for {landuse_type} in {self.analysis_years[year_idx]}")
                        continue

                    # 展平坐标和数据
                    src_points = np.column_stack((
                        src_lon_grid.ravel(),
                        src_lat_grid.ravel()
                    ))
                    src_values = src_data.ravel()

                    # 去除NaN值
                    valid_mask = ~np.isnan(src_values)
                    if np.sum(valid_mask) == 0:
                        continue

                    src_points = src_points[valid_mask]
                    src_values = src_values[valid_mask]

                    target_points = np.column_stack((
                        target_lon_grid.ravel(),
                        target_lat_grid.ravel()
                    ))

                    # 使用线性插值
                    interpolated = griddata(
                        src_points, src_values, target_points,
                        method='linear', fill_value=0
                    )

                    # 重塑为网格并确保在0-1范围内
                    regridded_data = interpolated.reshape(n_target_lat, n_target_lon)
                    regridded_data = np.clip(regridded_data, 0.0, 1.0)

                    regridded_landuse[year_idx, type_idx] = regridded_data

                if year_idx % 5 == 0:
                    print(f"  Processed year {self.analysis_years[year_idx]} ({year_idx + 1}/{n_years})")

            self.aligned_landuse = regridded_landuse

            print(f"✅ Land use regridding completed")
            print(f"   Output shape: {regridded_landuse.shape}")

            # 数据质量检查
            for type_idx, landuse_type in enumerate(self.landuse_types.keys()):
                type_data = regridded_landuse[:, type_idx, :, :]
                valid_count = np.sum(~np.isnan(type_data) & (type_data > 0))
                total_count = type_data.size
                print(
                    f"   {landuse_type}: {valid_count}/{total_count} valid cells ({100 * valid_count / total_count:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Error regridding land use data: {e}")
            return False

    def create_validation_plots(self):
        """创建数据对齐验证图"""
        print("\n" + "=" * 60)
        print("CREATING VALIDATION PLOTS")
        print("=" * 60)

        try:
            # 修复：创建3x2的子图布局来容纳1个降水图 + 5个土地利用图
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))

            # 选择一个示例年份 (2000年)
            example_year_idx = 10  # 2000年
            example_year = self.analysis_years[example_year_idx]

            # 1. 降水数据 (第一个子图)
            ax = axes[0, 0]
            im = ax.imshow(self.aligned_precip[example_year_idx],
                           extent=[self.target_lon.min(), self.target_lon.max(),
                                   self.target_lat.min(), self.target_lat.max()],
                           cmap='Blues', aspect='auto', origin='upper')
            ax.set_title(f'Aligned Precipitation ({example_year})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            plt.colorbar(im, ax=ax, label='mm/year', shrink=0.8)

            # 2-6. 土地利用类型
            landuse_positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

            for i, (landuse_type, description) in enumerate(self.landuse_types.items()):
                row, col = landuse_positions[i]
                ax = axes[row, col]

                data = self.aligned_landuse[example_year_idx, i]
                im = ax.imshow(data,
                               extent=[self.target_lon.min(), self.target_lon.max(),
                                       self.target_lat.min(), self.target_lat.max()],
                               cmap='viridis', aspect='auto', vmin=0, vmax=1, origin='upper')
                ax.set_title(f'{description} ({example_year})', fontsize=12, fontweight='bold')
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
                plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8)

            plt.suptitle(f'Data Alignment Validation - {self.analysis_period}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            output_path = os.path.join(self.output_dir,
                                       f'data_alignment_validation_{self.analysis_period.replace("-", "_")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            # 创建数据统计图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 降水量时间序列
            ax = axes[0, 0]
            annual_mean_precip = np.mean(self.aligned_precip, axis=(1, 2))
            ax.plot(self.analysis_years, annual_mean_precip, 'b-o', linewidth=2, markersize=5)
            ax.set_title('Annual Mean Precipitation', fontsize=11, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Precipitation (mm/year)')
            ax.grid(True, alpha=0.3)

            # 土地利用类型时间序列
            landuse_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

            for i, (landuse_type, description) in enumerate(self.landuse_types.items()):
                row, col = landuse_positions[i]
                ax = axes[row, col]

                # 计算年平均比例
                annual_mean_landuse = np.mean(self.aligned_landuse[:, i, :, :], axis=(1, 2))
                ax.plot(self.analysis_years, annual_mean_landuse, 'g-o', linewidth=2, markersize=5)
                ax.set_title(f'{description} Mean Proportion', fontsize=11, fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Proportion')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, None)

            plt.suptitle(f'Time Series Summary - {self.analysis_period}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存时间序列图
            output_path2 = os.path.join(self.output_dir,
                                        f'time_series_summary_{self.analysis_period.replace("-", "_")}.png')
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            print(f"✅ Validation plots saved to:")
            print(f"   - {output_path}")
            print(f"   - {output_path2}")

            return True

        except Exception as e:
            print(f"❌ Error creating validation plots: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export_aligned_data(self):
        """导出对齐后的数据"""
        print("\n" + "=" * 60)
        print("EXPORTING ALIGNED DATA")
        print("=" * 60)

        try:
            # 导出为NetCDF格式以保持空间信息
            output_file = os.path.join(self.output_dir,
                                       f'aligned_precipitation_landuse_data_{self.analysis_period.replace("-", "_")}.nc')

            # 创建xarray数据集
            coords = {
                'year': self.analysis_years,
                'landuse_type': list(self.landuse_types.keys()),
                'latitude': self.target_lat,
                'longitude': self.target_lon
            }

            data_vars = {
                'precipitation': (['year', 'latitude', 'longitude'], self.aligned_precip),
                'landuse': (['year', 'landuse_type', 'latitude', 'longitude'], self.aligned_landuse)
            }

            attrs = {
                'title': 'Aligned Precipitation and Land Use Data for China',
                'time_coverage_start': f'{self.analysis_years[0]}-01-01',
                'time_coverage_end': f'{self.analysis_years[-1]}-12-31',
                'spatial_resolution': f'{self.target_resolution} degrees',
                'landuse_types': ', '.join(self.landuse_types.values()),
                'created_by': 'PrecipitationLandUsePreprocessor',
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)

            # 添加变量属性
            ds.precipitation.attrs = {
                'units': 'mm/year',
                'long_name': 'Annual Precipitation',
                'description': 'Annual total precipitation regridded to common grid'
            }

            ds.landuse.attrs = {
                'units': 'proportion (0-1)',
                'long_name': 'Land Use Type Proportion',
                'description': 'Proportion of each land use type in grid cell'
            }

            # 保存到NetCDF
            ds.to_netcdf(output_file)

            print(f"✅ Aligned data exported to: {output_file}")
            print(f"   File size: {os.path.getsize(output_file) / (1024 ** 2):.1f} MB")

            # 导出数据摘要
            summary_file = os.path.join(self.output_dir,
                                        f'data_alignment_summary_{self.analysis_period.replace("-", "_")}.txt')

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("PRECIPITATION AND LAND USE DATA ALIGNMENT SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Period: {self.analysis_period}\n")
                f.write(f"Target Resolution: {self.target_resolution}°\n\n")

                f.write("SPATIAL COVERAGE:\n")
                f.write(f"  Longitude: {self.target_lon.min():.2f}° - {self.target_lon.max():.2f}°\n")
                f.write(f"  Latitude: {self.target_lat.min():.2f}° - {self.target_lat.max():.2f}°\n")
                f.write(
                    f"  Grid Size: {len(self.target_lon)} × {len(self.target_lat)} = {len(self.target_lon) * len(self.target_lat)} cells\n\n")

                f.write("DATA SUMMARY:\n")
                f.write(f"  Precipitation Data Shape: {self.aligned_precip.shape}\n")
                f.write(
                    f"  Precipitation Range: {np.nanmin(self.aligned_precip):.1f} - {np.nanmax(self.aligned_precip):.1f} mm/year\n")
                f.write(f"  Land Use Data Shape: {self.aligned_landuse.shape}\n\n")

                f.write("LAND USE TYPES:\n")
                for i, (key, description) in enumerate(self.landuse_types.items()):
                    type_data = self.aligned_landuse[:, i, :, :]
                    valid_ratio = np.sum(type_data > 0) / type_data.size
                    f.write(f"  {i + 1}. {description} ({key}): {valid_ratio:.1%} valid cells\n")

                f.write(f"\nREADY FOR CORRELATION ANALYSIS\n")
                f.write(
                    f"Use the aligned_precipitation_landuse_data_{self.analysis_period.replace('-', '_')}.nc file\n")
                f.write(f"for calculating correlations between precipitation and land use types.\n")

            print(f"✅ Summary exported to: {summary_file}")

            return True

        except Exception as e:
            print(f"❌ Error exporting aligned data: {e}")
            return False

    def run_complete_preprocessing(self):
        """运行完整的数据预处理流程"""
        print("=" * 80)
        print("PRECIPITATION AND LAND USE DATA PREPROCESSING")
        print(f"TARGET PERIOD: {self.analysis_period}")
        print("=" * 80)

        try:
            # 1. 加载降水数据
            if not self.load_precipitation_data():
                return False

            # 2. 加载土地利用数据
            if not self.load_landuse_data():
                return False

            # 3. 创建统一网格
            if not self.create_common_grid():
                return False

            # 4. 重采样降水数据
            if not self.regrid_precipitation_data():
                return False

            # 5. 重采样土地利用数据
            if not self.regrid_landuse_data():
                return False

            # 6. 创建验证图
            if not self.create_validation_plots():
                return False

            # 7. 导出对齐后的数据
            if not self.export_aligned_data():
                return False

            print("\n" + "=" * 80)
            print("PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nData is now ready for correlation analysis:")
            print(f"  📊 Aligned precipitation data: {self.aligned_precip.shape}")
            print(f"  📊 Aligned land use data: {self.aligned_landuse.shape}")
            print(f"  🌍 Common grid: {len(self.target_lon)} × {len(self.target_lat)} cells")
            print(f"  📅 Time period: {self.analysis_period} ({len(self.analysis_years)} years)")
            print(f"  📁 Output directory: {self.output_dir}")

            return True

        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 文件路径配置
    precip_file = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
    landuse_dir = r"C:\Users\ZEC\Desktop\raw_data"
    output_dir = r"D:\Pythonpro\2024_D\result\Q1_preprocessing"

    print("Starting unified data preprocessing...")
    print(f"Precipitation file: {precip_file}")
    print(f"Land use directory: {landuse_dir}")
    print(f"Output directory: {output_dir}")

    # 检查输入文件
    if not os.path.exists(precip_file):
        print(f"❌ Precipitation file not found: {precip_file}")
        return

    if not os.path.exists(landuse_dir):
        print(f"❌ Land use directory not found: {landuse_dir}")
        return

    # 创建预处理器并运行
    preprocessor = PrecipitationLandUsePreprocessor(precip_file, landuse_dir, output_dir)

    success = preprocessor.run_complete_preprocessing()

    if success:
        print(f"\n🎉 Data preprocessing completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"📋 Check the summary file for detailed information")
        print(f"🔄 Ready for correlation analysis!")
    else:
        print(f"\n❌ Data preprocessing failed!")
        print(f"📋 Check error messages above for troubleshooting")


if __name__ == "__main__":
    main()
