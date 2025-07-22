import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# 设置英文环境和图表样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PrecipitationLandUseCorrelationAnalyzer:
    """
    降水量与土地利用关联性分析器
    基于公式：ρ_PLk = cov(P_it, L_itk) / (σ_Pt * σ_Ltk)
    """

    def __init__(self, aligned_data_file, output_dir):
        """
        初始化相关性分析器

        Parameters:
        -----------
        aligned_data_file : str
            对齐后的数据NetCDF文件路径
        output_dir : str
            输出目录路径
        """
        self.aligned_data_file = aligned_data_file
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 土地利用类型和颜色映射
        self.landuse_types = {
            'cropland': 'Cropland',
            'forest': 'Forest',
            'grass': 'Grassland',
            'shrub': 'Shrubland',
            'wetland': 'Wetland'
        }

        self.landuse_colors = {
            'cropland': '#FFD700',  # Gold
            'forest': '#228B22',  # Forest Green
            'grass': '#9ACD32',  # Yellow Green
            'shrub': '#8FBC8F',  # Dark Sea Green
            'wetland': '#4682B4'  # Steel Blue
        }

        # 数据容器
        self.dataset = None
        self.precipitation = None  # P_it: [region_i, time_t]
        self.landuse = None  # L_itk: [region_i, time_t, landuse_k]
        self.years = None
        self.longitude = None
        self.latitude = None

        # 相关性分析结果
        self.correlations = {}  # ρ_PLk for each land use type
        self.p_values = {}  # Statistical significance
        self.temporal_correlations = {}  # Time-varying correlations
        self.spatial_correlations = {}  # Spatial correlation patterns

    def load_aligned_data(self):
        """加载对齐后的数据"""
        print("=" * 70)
        print("LOADING ALIGNED PRECIPITATION AND LAND USE DATA")
        print("=" * 70)

        try:
            # 加载NetCDF数据
            self.dataset = xr.open_dataset(self.aligned_data_file)

            print(f"Dataset info:")
            print(f"  Dimensions: {dict(self.dataset.dims)}")
            print(f"  Data variables: {list(self.dataset.data_vars.keys())}")
            print(f"  Coordinates: {list(self.dataset.coords.keys())}")

            # 提取数据
            self.precipitation = self.dataset.precipitation.values  # [year, lat, lon]
            self.landuse = self.dataset.landuse.values  # [year, landuse_type, lat, lon]
            self.years = self.dataset.year.values
            self.longitude = self.dataset.longitude.values
            self.latitude = self.dataset.latitude.values

            print(f"\nData shapes:")
            print(f"  Precipitation: {self.precipitation.shape} (year × lat × lon)")
            print(f"  Land use: {self.landuse.shape} (year × type × lat × lon)")
            print(f"  Years: {len(self.years)} ({self.years[0]} - {self.years[-1]})")
            print(f"  Spatial grid: {len(self.latitude)} × {len(self.longitude)}")

            # 数据质量检查
            precip_valid = np.sum(~np.isnan(self.precipitation))
            precip_total = self.precipitation.size
            landuse_valid = np.sum(~np.isnan(self.landuse))
            landuse_total = self.landuse.size

            print(f"\nData quality:")
            print(f"  Precipitation valid: {precip_valid}/{precip_total} ({100 * precip_valid / precip_total:.1f}%)")
            print(f"  Land use valid: {landuse_valid}/{landuse_total} ({100 * landuse_valid / landuse_total:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Error loading aligned data: {e}")
            return False

    def reshape_data_for_correlation(self):
        """
        重塑数据为相关性分析所需的格式
        P_it: [region_i, time_t]
        L_itk: [region_i, time_t, landuse_k]
        """
        print("\n" + "=" * 70)
        print("RESHAPING DATA FOR CORRELATION ANALYSIS")
        print("=" * 70)

        try:
            n_years = len(self.years)
            n_landuse = len(self.landuse_types)
            n_lat = len(self.latitude)
            n_lon = len(self.longitude)
            n_regions = n_lat * n_lon

            print(f"Reshaping data:")
            print(f"  Original precipitation: {self.precipitation.shape}")
            print(f"  Original land use: {self.landuse.shape}")
            print(f"  Target: {n_regions} regions × {n_years} years")

            # 重塑降水数据: P_it [region_i, time_t]
            self.P_it = self.precipitation.reshape(n_years, n_regions).T  # [region, year]

            # 重塑土地利用数据: L_itk [region_i, time_t, landuse_k]
            self.L_itk = self.landuse.transpose(1, 0, 2, 3).reshape(n_landuse, n_years, n_regions).transpose(2, 1,
                                                                                                             0)  # [region, year, landuse]

            print(f"\nReshaped data:")
            print(f"  P_it shape: {self.P_it.shape} (region × year)")
            print(f"  L_itk shape: {self.L_itk.shape} (region × year × landuse)")

            # 创建有效数据掩码
            self.valid_mask = (~np.isnan(self.P_it)) & (~np.isnan(self.L_itk).any(axis=2))
            valid_regions = np.sum(self.valid_mask.any(axis=1))

            print(f"  Valid regions with data: {valid_regions}/{n_regions}")

            return True

        except Exception as e:
            print(f"❌ Error reshaping data: {e}")
            return False

    def calculate_correlations(self):
        """
        计算降水量与土地利用的相关系数
        ρ_PLk = cov(P_it, L_itk) / (σ_Pt * σ_Ltk)
        """
        print("\n" + "=" * 70)
        print("CALCULATING PRECIPITATION-LAND USE CORRELATIONS")
        print("=" * 70)

        try:
            n_regions, n_years, n_landuse = self.L_itk.shape

            # 初始化结果容器
            self.correlations = {}
            self.p_values = {}

            print(f"Computing correlations for {n_landuse} land use types...")

            for k, (landuse_key, landuse_name) in enumerate(self.landuse_types.items()):
                print(f"\nProcessing {landuse_name} ({landuse_key})...")

                # 提取当前土地利用类型的数据 L_itk
                L_k = self.L_itk[:, :, k]  # [region, year]

                # 计算区域级相关系数
                region_correlations = []
                region_p_values = []

                valid_count = 0
                for i in range(n_regions):
                    # 获取区域i的降水和土地利用时间序列
                    P_i = self.P_it[i, :]  # 区域i的降水时间序列
                    L_ik = L_k[i, :]  # 区域i的k类土地利用时间序列

                    # 检查有效数据
                    valid_data_mask = self.valid_mask[i, :] & (~np.isnan(P_i)) & (~np.isnan(L_ik))

                    if np.sum(valid_data_mask) >= 5:  # 至少需要5个有效数据点
                        P_valid = P_i[valid_data_mask]
                        L_valid = L_ik[valid_data_mask]

                        # 检查数据方差
                        if np.std(P_valid) > 0 and np.std(L_valid) > 0:
                            # 计算Pearson相关系数
                            corr, p_val = pearsonr(P_valid, L_valid)
                            region_correlations.append(corr)
                            region_p_values.append(p_val)
                            valid_count += 1
                        else:
                            region_correlations.append(np.nan)
                            region_p_values.append(np.nan)
                    else:
                        region_correlations.append(np.nan)
                        region_p_values.append(np.nan)

                self.correlations[landuse_key] = np.array(region_correlations)
                self.p_values[landuse_key] = np.array(region_p_values)

                # 统计结果
                valid_correlations = np.array(region_correlations)[~np.isnan(region_correlations)]
                if len(valid_correlations) > 0:
                    print(f"  Valid correlations: {len(valid_correlations)}/{n_regions}")
                    print(f"  Correlation range: {np.min(valid_correlations):.3f} to {np.max(valid_correlations):.3f}")
                    print(f"  Mean correlation: {np.mean(valid_correlations):.3f} ± {np.std(valid_correlations):.3f}")

                    # 显著性统计
                    valid_p_values = np.array(region_p_values)[~np.isnan(region_p_values)]
                    significant_count = np.sum(valid_p_values < 0.05)
                    print(
                        f"  Significant correlations (p<0.05): {significant_count}/{len(valid_p_values)} ({100 * significant_count / len(valid_p_values):.1f}%)")
                else:
                    print(f"  No valid correlations found")

            print(f"\n✅ Correlation analysis completed for all land use types")
            return True

        except Exception as e:
            print(f"❌ Error calculating correlations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_temporal_correlations(self):
        """计算时间变化的相关性"""
        print("\n" + "=" * 70)
        print("CALCULATING TEMPORAL CORRELATION PATTERNS")
        print("=" * 70)

        try:
            n_years = len(self.years)
            window_size = 10  # 10年滑动窗口

            self.temporal_correlations = {}

            for landuse_key, landuse_name in self.landuse_types.items():
                print(f"Processing temporal correlations for {landuse_name}...")

                temporal_corrs = []
                temporal_years = []

                for start_year in range(n_years - window_size + 1):
                    end_year = start_year + window_size
                    window_years = self.years[start_year:end_year]
                    center_year = window_years[window_size // 2]

                    # 提取窗口内的数据
                    P_window = self.P_it[:, start_year:end_year]
                    L_window = self.L_itk[:, start_year:end_year, list(self.landuse_types.keys()).index(landuse_key)]

                    # 计算所有区域的平均相关性
                    window_correlations = []

                    for i in range(self.P_it.shape[0]):
                        P_i = P_window[i, :]
                        L_i = L_window[i, :]

                        valid_mask = (~np.isnan(P_i)) & (~np.isnan(L_i))
                        if np.sum(valid_mask) >= 5:
                            P_valid = P_i[valid_mask]
                            L_valid = L_i[valid_mask]

                            if np.std(P_valid) > 0 and np.std(L_valid) > 0:
                                corr, _ = pearsonr(P_valid, L_valid)
                                window_correlations.append(corr)

                    if len(window_correlations) > 0:
                        mean_corr = np.mean(window_correlations)
                        temporal_corrs.append(mean_corr)
                        temporal_years.append(center_year)

                self.temporal_correlations[landuse_key] = {
                    'years': np.array(temporal_years),
                    'correlations': np.array(temporal_corrs)
                }

                if len(temporal_corrs) > 0:
                    print(f"  Mean temporal correlation: {np.mean(temporal_corrs):.3f}")
                    print(f"  Temporal correlation range: {np.min(temporal_corrs):.3f} to {np.max(temporal_corrs):.3f}")

            print(f"✅ Temporal correlation analysis completed")
            return True

        except Exception as e:
            print(f"❌ Error calculating temporal correlations: {e}")
            return False

    def calculate_national_correlations(self):
        """计算全国平均的相关性"""
        print("\n" + "=" * 70)
        print("CALCULATING NATIONAL AVERAGE CORRELATIONS")
        print("=" * 70)

        try:
            # 计算全国平均降水量和土地利用比例
            self.national_correlations = {}

            for landuse_key, landuse_name in self.landuse_types.items():
                print(f"Processing national correlation for {landuse_name}...")

                # 计算全国年平均值
                national_precip = []
                national_landuse = []

                for year_idx in range(len(self.years)):
                    # 当年全国平均降水量
                    year_precip = self.precipitation[year_idx, :, :].flatten()
                    valid_precip = year_precip[~np.isnan(year_precip)]
                    if len(valid_precip) > 0:
                        national_precip.append(np.mean(valid_precip))
                    else:
                        national_precip.append(np.nan)

                    # 当年全国平均土地利用比例
                    landuse_idx = list(self.landuse_types.keys()).index(landuse_key)
                    year_landuse = self.landuse[year_idx, landuse_idx, :, :].flatten()
                    valid_landuse = year_landuse[~np.isnan(year_landuse)]
                    if len(valid_landuse) > 0:
                        national_landuse.append(np.mean(valid_landuse))
                    else:
                        national_landuse.append(np.nan)

                # 计算全国相关性
                national_precip = np.array(national_precip)
                national_landuse = np.array(national_landuse)

                valid_mask = (~np.isnan(national_precip)) & (~np.isnan(national_landuse))

                if np.sum(valid_mask) >= 5:
                    valid_precip = national_precip[valid_mask]
                    valid_landuse = national_landuse[valid_mask]

                    if np.std(valid_precip) > 0 and np.std(valid_landuse) > 0:
                        corr, p_val = pearsonr(valid_precip, valid_landuse)
                        spearman_corr, spearman_p = spearmanr(valid_precip, valid_landuse)

                        self.national_correlations[landuse_key] = {
                            'pearson_r': corr,
                            'pearson_p': p_val,
                            'spearman_r': spearman_corr,
                            'spearman_p': spearman_p,
                            'years': self.years[valid_mask],
                            'precipitation': valid_precip,
                            'landuse': valid_landuse
                        }

                        print(f"  Pearson r = {corr:.3f} (p = {p_val:.3f})")
                        print(f"  Spearman r = {spearman_corr:.3f} (p = {spearman_p:.3f})")

                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"  Significance: {significance}")
                    else:
                        print(f"  Insufficient variation in data")
                else:
                    print(f"  Insufficient valid data points")

            print(f"✅ National correlation analysis completed")
            return True

        except Exception as e:
            print(f"❌ Error calculating national correlations: {e}")
            return False

    def create_correlation_maps(self):
        """创建相关性空间分布地图"""
        print("\n" + "=" * 70)
        print("CREATING CORRELATION SPATIAL DISTRIBUTION MAPS")
        print("=" * 70)

        try:
            # 重塑相关系数回到空间网格
            n_lat = len(self.latitude)
            n_lon = len(self.longitude)

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()

            for i, (landuse_key, landuse_name) in enumerate(self.landuse_types.items()):
                ax = axes[i]

                # 重塑相关系数到空间网格
                corr_grid = self.correlations[landuse_key].reshape(n_lat, n_lon)
                p_grid = self.p_values[landuse_key].reshape(n_lat, n_lon)

                # 创建显著性掩码
                significant_mask = p_grid < 0.05

                # 绘制相关性地图
                im = ax.imshow(corr_grid,
                               extent=[self.longitude.min(), self.longitude.max(),
                                       self.latitude.min(), self.latitude.max()],
                               cmap='RdBu_r', vmin=-1, vmax=1,
                               aspect='auto', origin='upper')

                # 标记显著性区域
                if np.sum(significant_mask) > 0:
                    lat_indices, lon_indices = np.where(significant_mask)
                    if len(lat_indices) > 0:
                        scatter_lats = self.latitude[lat_indices]
                        scatter_lons = self.longitude[lon_indices]
                        ax.scatter(scatter_lons, scatter_lats,
                                   c='black', s=0.5, alpha=0.7, marker='.')

                ax.set_title(f'{landuse_name}\nPrecipitation-Land Use Correlation',
                             fontsize=11, fontweight='bold')
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')

                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Correlation Coefficient', fontsize=9)

                # 添加统计信息
                valid_corrs = self.correlations[landuse_key][~np.isnan(self.correlations[landuse_key])]
                if len(valid_corrs) > 0:
                    mean_corr = np.mean(valid_corrs)
                    sig_count = np.sum(self.p_values[landuse_key] < 0.05)
                    total_count = np.sum(~np.isnan(self.p_values[landuse_key]))

                    stats_text = f'Mean r: {mean_corr:.3f}\nSig. cells: {sig_count}/{total_count}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # 移除多余的子图
            fig.delaxes(axes[5])

            plt.suptitle('Precipitation-Land Use Correlation Maps\n(Black dots indicate p < 0.05)',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            output_path = os.path.join(self.output_dir, 'correlation_spatial_maps.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            print(f"✅ Correlation maps saved to: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating correlation maps: {e}")
            return False

    def create_correlation_summary_plots(self):
        """创建相关性分析摘要图表"""
        print("\n" + "=" * 70)
        print("CREATING CORRELATION SUMMARY PLOTS")
        print("=" * 70)

        try:
            # 1. 相关系数分布直方图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, (landuse_key, landuse_name) in enumerate(self.landuse_types.items()):
                ax = axes[i]

                valid_corrs = self.correlations[landuse_key][~np.isnan(self.correlations[landuse_key])]

                if len(valid_corrs) > 0:
                    ax.hist(valid_corrs, bins=30, alpha=0.7, color=self.landuse_colors[landuse_key],
                            edgecolor='black', linewidth=0.5)

                    # 添加统计线
                    mean_corr = np.mean(valid_corrs)
                    ax.axvline(mean_corr, color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {mean_corr:.3f}')
                    ax.axvline(0, color='black', linestyle='-', alpha=0.5)

                    ax.set_title(f'{landuse_name}\nCorrelation Distribution', fontweight='bold')
                    ax.set_xlabel('Correlation Coefficient')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Valid Data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14)
                    ax.set_title(f'{landuse_name}\nNo Valid Correlations')

            fig.delaxes(axes[5])
            plt.suptitle('Distribution of Precipitation-Land Use Correlations',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()

            output_path1 = os.path.join(self.output_dir, 'correlation_distributions.png')
            plt.savefig(output_path1, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            # 2. 全国相关性时间序列
            if hasattr(self, 'national_correlations'):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                # 散点图
                for landuse_key, landuse_name in self.landuse_types.items():
                    if landuse_key in self.national_correlations:
                        data = self.national_correlations[landuse_key]
                        ax1.scatter(data['precipitation'], data['landuse'],
                                    label=f"{landuse_name} (r={data['pearson_r']:.3f})",
                                    color=self.landuse_colors[landuse_key], alpha=0.7, s=50)

                ax1.set_xlabel('National Average Precipitation (mm/year)')
                ax1.set_ylabel('National Average Land Use Proportion')
                ax1.set_title('National Precipitation vs Land Use Relationships', fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 时间变化相关性
                if hasattr(self, 'temporal_correlations'):
                    for landuse_key, landuse_name in self.landuse_types.items():
                        if landuse_key in self.temporal_correlations:
                            data = self.temporal_correlations[landuse_key]
                            ax2.plot(data['years'], data['correlations'],
                                     marker='o', linewidth=2, markersize=5,
                                     label=landuse_name, color=self.landuse_colors[landuse_key])

                ax2.set_xlabel('Year')
                ax2.set_ylabel('10-Year Window Correlation')
                ax2.set_title('Temporal Evolution of Correlations', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(0, color='black', linestyle='-', alpha=0.5)

                plt.tight_layout()

                output_path2 = os.path.join(self.output_dir, 'national_correlations.png')
                plt.savefig(output_path2, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

            # 3. 相关性统计摘要
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            summary_data = []
            for landuse_key, landuse_name in self.landuse_types.items():
                valid_corrs = self.correlations[landuse_key][~np.isnan(self.correlations[landuse_key])]
                valid_p_vals = self.p_values[landuse_key][~np.isnan(self.p_values[landuse_key])]

                if len(valid_corrs) > 0:
                    summary_data.append({
                        'Land Use': landuse_name,
                        'Mean Correlation': np.mean(valid_corrs),
                        'Std Correlation': np.std(valid_corrs),
                        'Positive (%)': 100 * np.sum(valid_corrs > 0) / len(valid_corrs),
                        'Significant (%)': 100 * np.sum(valid_p_vals < 0.05) / len(valid_p_vals) if len(
                            valid_p_vals) > 0 else 0
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)

                x_pos = np.arange(len(summary_df))
                bars = ax.bar(x_pos, summary_df['Mean Correlation'],
                              yerr=summary_df['Std Correlation'],
                              color=[self.landuse_colors[key] for key in self.landuse_types.keys()],
                              alpha=0.7, capsize=5)

                ax.set_xlabel('Land Use Type')
                ax.set_ylabel('Mean Correlation Coefficient')
                ax.set_title('Summary of Precipitation-Land Use Correlations', fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(summary_df['Land Use'], rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(0, color='black', linestyle='-', alpha=0.5)

                # 添加显著性百分比标注
                for i, (bar, sig_pct) in enumerate(zip(bars, summary_df['Significant (%)'])):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{sig_pct:.0f}% sig.',
                            ha='center', va='bottom', fontsize=9)

                plt.tight_layout()

                output_path3 = os.path.join(self.output_dir, 'correlation_summary.png')
                plt.savefig(output_path3, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

            print(f"✅ Summary plots created and saved")
            return True

        except Exception as e:
            print(f"❌ Error creating summary plots: {e}")
            return False

    def export_correlation_results(self):
        """导出相关性分析结果"""
        print("\n" + "=" * 70)
        print("EXPORTING CORRELATION ANALYSIS RESULTS")
        print("=" * 70)

        try:
            # 1. 导出区域相关性结果
            correlation_results = []

            n_lat = len(self.latitude)
            n_lon = len(self.longitude)

            for i, lat in enumerate(self.latitude):
                for j, lon in enumerate(self.longitude):
                    region_idx = i * n_lon + j

                    row = {
                        'latitude': lat,
                        'longitude': lon,
                        'region_index': region_idx
                    }

                    for landuse_key in self.landuse_types.keys():
                        if region_idx < len(self.correlations[landuse_key]):
                            corr = self.correlations[landuse_key][region_idx]
                            p_val = self.p_values[landuse_key][region_idx]

                            row[f'{landuse_key}_correlation'] = corr if not np.isnan(corr) else None
                            row[f'{landuse_key}_p_value'] = p_val if not np.isnan(p_val) else None
                            row[f'{landuse_key}_significant'] = p_val < 0.05 if not np.isnan(p_val) else None

                    correlation_results.append(row)

            correlation_df = pd.DataFrame(correlation_results)

            # 2. 导出全国相关性结果
            if hasattr(self, 'national_correlations'):
                national_results = []
                for landuse_key, landuse_name in self.landuse_types.items():
                    if landuse_key in self.national_correlations:
                        data = self.national_correlations[landuse_key]
                        national_results.append({
                            'land_use_type': landuse_name,
                            'land_use_key': landuse_key,
                            'pearson_correlation': data['pearson_r'],
                            'pearson_p_value': data['pearson_p'],
                            'spearman_correlation': data['spearman_r'],
                            'spearman_p_value': data['spearman_p'],
                            'pearson_significant': data['pearson_p'] < 0.05,
                            'spearman_significant': data['spearman_p'] < 0.05,
                            'sample_size': len(data['years'])
                        })

                national_df = pd.DataFrame(national_results)
            else:
                national_df = pd.DataFrame()

            # 3. 导出统计摘要
            summary_results = []
            for landuse_key, landuse_name in self.landuse_types.items():
                valid_corrs = self.correlations[landuse_key][~np.isnan(self.correlations[landuse_key])]
                valid_p_vals = self.p_values[landuse_key][~np.isnan(self.p_values[landuse_key])]

                if len(valid_corrs) > 0:
                    summary_results.append({
                        'land_use_type': landuse_name,
                        'land_use_key': landuse_key,
                        'mean_correlation': np.mean(valid_corrs),
                        'std_correlation': np.std(valid_corrs),
                        'min_correlation': np.min(valid_corrs),
                        'max_correlation': np.max(valid_corrs),
                        'median_correlation': np.median(valid_corrs),
                        'positive_correlations_pct': 100 * np.sum(valid_corrs > 0) / len(valid_corrs),
                        'negative_correlations_pct': 100 * np.sum(valid_corrs < 0) / len(valid_corrs),
                        'significant_correlations_pct': 100 * np.sum(valid_p_vals < 0.05) / len(valid_p_vals) if len(
                            valid_p_vals) > 0 else 0,
                        'total_regions': len(valid_corrs),
                        'significant_regions': np.sum(valid_p_vals < 0.05) if len(valid_p_vals) > 0 else 0
                    })

            summary_df = pd.DataFrame(summary_results)

            # 导出到Excel文件
            excel_file = os.path.join(self.output_dir, 'precipitation_landuse_correlation_results.xlsx')

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                correlation_df.to_excel(writer, sheet_name='Regional_Correlations', index=False)
                if not national_df.empty:
                    national_df.to_excel(writer, sheet_name='National_Correlations', index=False)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

                # 添加时间序列相关性数据
                if hasattr(self, 'temporal_correlations'):
                    temporal_data = []
                    for landuse_key in self.landuse_types.keys():
                        if landuse_key in self.temporal_correlations:
                            data = self.temporal_correlations[landuse_key]
                            for year, corr in zip(data['years'], data['correlations']):
                                temporal_data.append({
                                    'land_use_type': self.landuse_types[landuse_key],
                                    'land_use_key': landuse_key,
                                    'center_year': year,
                                    'correlation': corr
                                })

                    if temporal_data:
                        temporal_df = pd.DataFrame(temporal_data)
                        temporal_df.to_excel(writer, sheet_name='Temporal_Correlations', index=False)

            # 导出CSV文件
            csv_file = os.path.join(self.output_dir, 'regional_correlations.csv')
            correlation_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

            csv_summary_file = os.path.join(self.output_dir, 'correlation_summary.csv')
            summary_df.to_csv(csv_summary_file, index=False, encoding='utf-8-sig')

            print(f"✅ Correlation results exported:")
            print(f"   - Excel file: {excel_file}")
            print(f"   - Regional correlations CSV: {csv_file}")
            print(f"   - Summary CSV: {csv_summary_file}")

            # 打印摘要统计
            print(f"\n" + "=" * 70)
            print("CORRELATION ANALYSIS SUMMARY")
            print("=" * 70)

            for _, row in summary_df.iterrows():
                print(f"\n{row['land_use_type']}:")
                print(f"  Mean correlation: {row['mean_correlation']:.3f} ± {row['std_correlation']:.3f}")
                print(f"  Range: {row['min_correlation']:.3f} to {row['max_correlation']:.3f}")
                print(f"  Positive correlations: {row['positive_correlations_pct']:.1f}%")
                print(f"  Significant correlations: {row['significant_correlations_pct']:.1f}%")
                print(f"  Total regions analyzed: {row['total_regions']}")

            return True

        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return False

    def run_complete_correlation_analysis(self):
        """运行完整的相关性分析"""
        print("=" * 80)
        print("PRECIPITATION-LAND USE CORRELATION ANALYSIS")
        print("=" * 80)

        try:
            # 1. 加载对齐后的数据
            if not self.load_aligned_data():
                return False

            # 2. 重塑数据格式
            if not self.reshape_data_for_correlation():
                return False

            # 3. 计算相关性
            if not self.calculate_correlations():
                return False

            # 4. 计算时间变化相关性
            if not self.calculate_temporal_correlations():
                return False

            # 5. 计算全国相关性
            if not self.calculate_national_correlations():
                return False

            # 6. 创建相关性地图
            if not self.create_correlation_maps():
                return False

            # 7. 创建摘要图表
            if not self.create_correlation_summary_plots():
                return False

            # 8. 导出结果
            if not self.export_correlation_results():
                return False

            print(f"\n" + "=" * 80)
            print("CORRELATION ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\nGenerated outputs in '{self.output_dir}':")
            print("  📊 RESULT FILES:")
            print("    - precipitation_landuse_correlation_results.xlsx")
            print("    - regional_correlations.csv")
            print("    - correlation_summary.csv")
            print("  📈 VISUALIZATION FILES:")
            print("    - correlation_spatial_maps.png")
            print("    - correlation_distributions.png")
            print("    - national_correlations.png")
            print("    - correlation_summary.png")

            return True

        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 文件路径配置
    aligned_data_file = r"D:\Pythonpro\2024_D\result\Q1_preprocessing\aligned_precipitation_landuse_data_1990_2019.nc"
    output_dir = r"D:\Pythonpro\2024_D\result\Q1_correlation_analysis"

    print("Starting precipitation-land use correlation analysis...")
    print(f"Input file: {aligned_data_file}")
    print(f"Output directory: {output_dir}")

    # 检查输入文件
    if not os.path.exists(aligned_data_file):
        print(f"❌ Aligned data file not found: {aligned_data_file}")
        print("Please run the data preprocessing script first.")
        return

    # 创建分析器并运行
    analyzer = PrecipitationLandUseCorrelationAnalyzer(aligned_data_file, output_dir)

    success = analyzer.run_complete_correlation_analysis()

    if success:
        print(f"\n🎉 Correlation analysis completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"📊 Results ready for interpretation and reporting!")
    else:
        print(f"\n❌ Correlation analysis failed!")
        print(f"📋 Check error messages above for troubleshooting")


if __name__ == "__main__":
    main()