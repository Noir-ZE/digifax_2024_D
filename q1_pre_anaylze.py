# import netCDF4 as nc
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import os
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 设置英文环境和图表样式
# plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['xtick.labelsize'] = 9
# plt.rcParams['ytick.labelsize'] = 9
# plt.rcParams['legend.fontsize'] = 9
#
#
# class PrecipitationAnalyzer:
#     def __init__(self, file_path, output_dir):
#         """Initialize precipitation analyzer"""
#         self.file_path = file_path
#         self.output_dir = output_dir
#         self.dataset = None
#         self.precipitation_data = None
#         self.time_data = None
#         self.longitude = None
#         self.latitude = None
#         self.years_range = None
#
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
#
#     def load_data(self):
#         """Load NC file data"""
#         print("Loading data...")
#
#         try:
#             self.dataset = nc.Dataset(self.file_path, 'r')
#
#             # Read coordinate and time data
#             self.longitude = self.dataset.variables['longitude'][:]
#             self.latitude = self.dataset.variables['latitude'][:]
#             self.time_data = self.dataset.variables['time'][:]
#
#             # Convert time format
#             time_units = self.dataset.variables['time'].units
#             self.time_dates = nc.num2date(self.time_data, time_units)
#
#             # Get year range
#             self.years_range = list(range(1961, 2023))  # 1961-2022
#
#             print(f"Data loaded successfully:")
#             print(f"  - Time range: {self.time_dates[0].year} - {self.time_dates[-1].year}")
#             print(f"  - Spatial range: Longitude {self.longitude.min():.2f}° - {self.longitude.max():.2f}°")
#             print(f"  - Spatial range: Latitude {self.latitude.min():.2f}° - {self.latitude.max():.2f}°")
#             print(f"  - Data shape: {self.dataset.variables['pre'].shape}")
#
#             return True
#
#         except Exception as e:
#             print(f"Data loading failed: {e}")
#             return False
#
#     def preprocess_precipitation_data(self):
#         """Preprocess precipitation data, handle missing values"""
#         print("\nPreprocessing precipitation data...")
#
#         # Read data in batches to save memory
#         prec_var = self.dataset.variables['pre']
#
#         # First check data range
#         print("Checking data quality...")
#         sample_data = prec_var[0:100, :, :]  # Read first 100 days as sample
#         print(f"  - Sample data range: {sample_data.min():.3f} - {sample_data.max():.3f}")
#         print(f"  - Negative values count: {np.sum(sample_data < 0)}")
#         print(f"  - -99.9 values count: {np.sum(np.abs(sample_data + 99.9) < 0.1)}")
#
#         # Read complete data and handle missing values
#         print("Reading complete precipitation data (may take some time)...")
#         self.precipitation_data = prec_var[:]
#
#         # Handle missing values: set -99.9 and other negative values to 0
#         print("Processing missing values...")
#         original_negatives = np.sum(self.precipitation_data < 0)
#         self.precipitation_data[self.precipitation_data < 0] = 0.0
#
#         print(f"  - Set {original_negatives} negative values (missing values) to 0")
#         print(
#             f"  - Data range after processing: {self.precipitation_data.min():.3f} - {self.precipitation_data.max():.3f}")
#
#         return True
#
#     def calculate_annual_precipitation(self):
#         """Calculate annual precipitation"""
#         print("\nCalculating annual precipitation...")
#
#         # Convert time to year array
#         years_array = np.array([d.year for d in self.time_dates])
#
#         # Initialize result array
#         annual_precip = np.zeros((len(self.years_range), len(self.latitude), len(self.longitude)))
#
#         # Calculate annual total precipitation by year
#         for i, year in enumerate(self.years_range):
#             year_mask = years_array == year
#             if np.sum(year_mask) > 0:
#                 # Calculate total precipitation for this year
#                 annual_precip[i, :, :] = np.sum(self.precipitation_data[year_mask, :, :], axis=0)
#
#                 if i < 5 or i % 10 == 0:  # Show progress
#                     print(f"  - Processed year {year}, {np.sum(year_mask)} days")
#
#         self.annual_precipitation = annual_precip
#         print(f"Annual precipitation calculation completed, data shape: {annual_precip.shape}")
#         return True
#
#     def analyze_national_precipitation_trends(self):
#         """Analyze national precipitation trends"""
#         print("\nAnalyzing national precipitation trends...")
#
#         # 1. Calculate annual national average precipitation P_t
#         # Average spatial data for each year (ignore 0 values, which may be ocean or no-data areas)
#         national_annual_mean = []
#         national_annual_std = []
#         valid_years = []
#
#         for i, year in enumerate(self.years_range):
#             year_data = self.annual_precipitation[i, :, :]
#
#             # Only consider valid data points (areas with precipitation > 0)
#             valid_data = year_data[year_data > 0]
#
#             if len(valid_data) > 100:  # Ensure sufficient valid data points
#                 # P_t: National average precipitation
#                 mean_precip = np.mean(valid_data)
#                 # σ_Pt: Spatial standard deviation
#                 std_precip = np.std(valid_data)
#
#                 national_annual_mean.append(mean_precip)
#                 national_annual_std.append(std_precip)
#                 valid_years.append(year)
#
#         self.national_mean = np.array(national_annual_mean)
#         self.national_std = np.array(national_annual_std)
#         self.valid_years = np.array(valid_years)
#
#         # 2. Calculate annual change rate r_t
#         self.annual_change_rate = np.zeros(len(self.national_mean) - 1)
#         for i in range(len(self.national_mean) - 1):
#             if self.national_mean[i] != 0:
#                 self.annual_change_rate[i] = (self.national_mean[i + 1] - self.national_mean[i]) / self.national_mean[
#                     i] * 100
#
#         print(f"  - Valid years count: {len(valid_years)}")
#         print(
#             f"  - Average annual precipitation range: {self.national_mean.min():.1f} - {self.national_mean.max():.1f} mm")
#         print(f"  - Average standard deviation range: {self.national_std.min():.1f} - {self.national_std.max():.1f} mm")
#         print(
#             f"  - Annual change rate range: {self.annual_change_rate.min():.2f}% - {self.annual_change_rate.max():.2f}%")
#
#         return True
#
#     def create_individual_plots(self):
#         """Create individual analysis plots"""
#         print("\nGenerating individual analysis plots...")
#
#         # 1. National annual average precipitation trend (P_t)
#         plt.figure(figsize=(12, 8))
#         plt.plot(self.valid_years, self.national_mean, 'b-', linewidth=2, marker='o', markersize=5, alpha=0.8)
#
#         # Add trend line
#         z = np.polyfit(self.valid_years, self.national_mean, 1)
#         p = np.poly1d(z)
#         plt.plot(self.valid_years, p(self.valid_years), "r--", alpha=0.8, linewidth=2,
#                  label=f'Trend line (slope: {z[0]:.2f} mm/year)')
#
#         plt.title('National Annual Average Precipitation Trend (P_t)', fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Annual Precipitation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#
#         # Add statistics text
#         mean_val = np.mean(self.national_mean)
#         std_val = np.std(self.national_mean)
#         plt.text(0.02, 0.98, f'Mean: {mean_val:.1f} mm\nStd: {std_val:.1f} mm\nTrend: {z[0]:.2f} mm/year',
#                  transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig1_National_Average_Precipitation_Trend.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 2. Precipitation spatial standard deviation (σ_Pt)
#         plt.figure(figsize=(12, 8))
#         plt.plot(self.valid_years, self.national_std, 'g-', linewidth=2, marker='s', markersize=5, alpha=0.8)
#
#         # Add trend line
#         z_std = np.polyfit(self.valid_years, self.national_std, 1)
#         p_std = np.poly1d(z_std)
#         plt.plot(self.valid_years, p_std(self.valid_years), "r--", alpha=0.8, linewidth=2,
#                  label=f'Trend line (slope: {z_std[0]:.2f} mm/year)')
#
#         plt.title('Spatial Standard Deviation of Precipitation (σ_Pt)', fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Standard Deviation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#
#         # Add statistics text
#         mean_std = np.mean(self.national_std)
#         std_std = np.std(self.national_std)
#         plt.text(0.02, 0.98, f'Mean: {mean_std:.1f} mm\nStd: {std_std:.1f} mm\nTrend: {z_std[0]:.2f} mm/year',
#                  transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig2_Precipitation_Spatial_Standard_Deviation.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 3. Annual change rate (r_t)
#         plt.figure(figsize=(12, 8))
#         change_years = self.valid_years[1:]  # Change rate has one less year
#         colors = ['red' if x < 0 else 'blue' for x in self.annual_change_rate]
#         bars = plt.bar(change_years, self.annual_change_rate, color=colors, alpha=0.7, width=0.8)
#
#         plt.title('Annual Precipitation Change Rate (r_t)', fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Change Rate (%)', fontsize=12)
#         plt.grid(True, alpha=0.3, axis='y')
#         plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
#
#         # Add legend for colors
#         from matplotlib.patches import Patch
#         legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Increase'),
#                            Patch(facecolor='red', alpha=0.7, label='Decrease')]
#         plt.legend(handles=legend_elements)
#
#         # Add statistics text
#         mean_change = np.mean(self.annual_change_rate)
#         max_increase = np.max(self.annual_change_rate)
#         max_decrease = np.min(self.annual_change_rate)
#         plt.text(0.02, 0.98,
#                  f'Mean: {mean_change:.2f}%\nMax increase: {max_increase:.2f}%\nMax decrease: {max_decrease:.2f}%',
#                  transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig3_Annual_Precipitation_Change_Rate.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 4. Precipitation distribution by decades
#         plt.figure(figsize=(12, 8))
#         decades = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
#         decade_data = []
#         decade_labels = []
#
#         for i, decade_start in enumerate(range(1960, 2030, 10)):
#             decade_mask = (self.valid_years >= decade_start) & (self.valid_years < decade_start + 10)
#             if np.sum(decade_mask) > 0:
#                 decade_data.append(self.national_mean[decade_mask])
#                 decade_labels.append(decades[i])
#
#         box_plot = plt.boxplot(decade_data, labels=decade_labels, patch_artist=True)
#
#         # Color the boxes
#         colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
#         for patch, color in zip(box_plot['boxes'], colors):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.7)
#
#         plt.title('Precipitation Distribution by Decades', fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Decade', fontsize=12)
#         plt.ylabel('Annual Precipitation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3, axis='y')
#         plt.xticks(rotation=45)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig4_Precipitation_Distribution_by_Decades.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 5. Time series decomposition view
#         plt.figure(figsize=(14, 10))
#
#         # Subplot 1: Original time series
#         plt.subplot(3, 1, 1)
#         plt.plot(self.valid_years, self.national_mean, 'b-', linewidth=1.5, alpha=0.8)
#         plt.title('Original Time Series', fontsize=12, fontweight='bold')
#         plt.ylabel('Precipitation (mm)')
#         plt.grid(True, alpha=0.3)
#
#         # Subplot 2: 5-year moving average
#         plt.subplot(3, 1, 2)
#         window_size = 5
#         if len(self.national_mean) >= window_size:
#             moving_avg = pd.Series(self.national_mean).rolling(window=window_size, center=True).mean()
#             plt.plot(self.valid_years, self.national_mean, 'lightblue', alpha=0.6, label='Annual values')
#             plt.plot(self.valid_years, moving_avg, 'red', linewidth=2, label=f'{window_size}-year moving average')
#             plt.title('Trend Analysis', fontsize=12, fontweight='bold')
#             plt.ylabel('Precipitation (mm)')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#
#         # Subplot 3: Detrended series (residuals)
#         plt.subplot(3, 1, 3)
#         trend_line = p(self.valid_years)
#         residuals = self.national_mean - trend_line
#         plt.plot(self.valid_years, residuals, 'g-', linewidth=1.5, alpha=0.8)
#         plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#         plt.title('Detrended Series (Residuals)', fontsize=12, fontweight='bold')
#         plt.xlabel('Year')
#         plt.ylabel('Residual (mm)')
#         plt.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig5_Time_Series_Decomposition.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 6. Correlation analysis
#         plt.figure(figsize=(12, 8))
#
#         # Create correlation plot between precipitation and its standard deviation
#         plt.scatter(self.national_mean, self.national_std, alpha=0.6, s=50, c=self.valid_years,
#                     cmap='viridis', edgecolors='black', linewidth=0.5)
#
#         # Add correlation line
#         correlation = np.corrcoef(self.national_mean, self.national_std)[0, 1]
#         z_corr = np.polyfit(self.national_mean, self.national_std, 1)
#         p_corr = np.poly1d(z_corr)
#         x_corr = np.linspace(self.national_mean.min(), self.national_mean.max(), 100)
#         plt.plot(x_corr, p_corr(x_corr), 'r--', alpha=0.8, linewidth=2,
#                  label=f'Correlation: {correlation:.3f}')
#
#         plt.colorbar(label='Year')
#         plt.title('Correlation: Mean Precipitation vs Spatial Variability', fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('National Average Precipitation (mm)', fontsize=12)
#         plt.ylabel('Spatial Standard Deviation (mm)', fontsize=12)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'Fig6_Precipitation_Mean_vs_Variability_Correlation.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         print("All individual plots generated successfully!")
#         return True
#
#     def export_results(self):
#         """Export analysis results to CSV and XLSX"""
#         print("\nExporting analysis results...")
#
#         # Create results DataFrame
#         results_df = pd.DataFrame({
#             'Year': self.valid_years,
#             'National_Average_Precipitation_mm': self.national_mean,
#             'Spatial_Standard_Deviation_mm': self.national_std
#         })
#
#         # Add change rate (note: length differs by 1)
#         change_rate_full = np.full(len(self.valid_years), np.nan)
#         change_rate_full[1:] = self.annual_change_rate
#         results_df['Annual_Change_Rate_percent'] = change_rate_full
#
#         # Add additional calculated fields
#         results_df['Coefficient_of_Variation'] = (results_df['Spatial_Standard_Deviation_mm'] /
#                                                   results_df['National_Average_Precipitation_mm']) * 100
#
#         # Export to CSV
#         csv_path = os.path.join(self.output_dir, 'precipitation_analysis_results.csv')
#         results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
#
#         # Export to XLSX with formatting
#         xlsx_path = os.path.join(self.output_dir, 'precipitation_analysis_results.xlsx')
#         with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
#             # Main results sheet
#             results_df.to_excel(writer, sheet_name='Results', index=False)
#
#             # Summary statistics sheet
#             summary_stats = pd.DataFrame({
#                 'Statistic': ['Mean', 'Standard Deviation', 'Minimum', 'Maximum', 'Trend (mm/year)', 'First Year',
#                               'Last Year', 'Total Years'],
#                 'National_Average_Precipitation_mm': [
#                     self.national_mean.mean(),
#                     self.national_mean.std(),
#                     self.national_mean.min(),
#                     self.national_mean.max(),
#                     np.polyfit(self.valid_years, self.national_mean, 1)[0],
#                     self.valid_years.min(),
#                     self.valid_years.max(),
#                     len(self.valid_years)
#                 ],
#                 'Spatial_Standard_Deviation_mm': [
#                     self.national_std.mean(),
#                     self.national_std.std(),
#                     self.national_std.min(),
#                     self.national_std.max(),
#                     np.polyfit(self.valid_years, self.national_std, 1)[0],
#                     np.nan,
#                     np.nan,
#                     np.nan
#                 ]
#             })
#             summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
#
#         # Print summary
#         print(f"\nAnalysis results exported to:")
#         print(f"  - CSV: {csv_path}")
#         print(f"  - XLSX: {xlsx_path}")
#
#         # Display first 10 years of results
#         print("\nFirst 10 years of analysis results:")
#         print(results_df.head(10).to_string(index=False, float_format='%.2f'))
#
#         # Display summary statistics
#         print(f"\nSummary Statistics ({self.valid_years[0]}-{self.valid_years[-1]}):")
#         print(f"  - Average annual precipitation: {self.national_mean.mean():.1f} ± {self.national_mean.std():.1f} mm")
#         print(f"  - Precipitation trend: {np.polyfit(self.valid_years, self.national_mean, 1)[0]:.2f} mm/year")
#         print(f"  - Average spatial variability: {self.national_std.mean():.1f} ± {self.national_std.std():.1f} mm")
#         print(
#             f"  - Maximum annual precipitation: {self.national_mean.max():.1f} mm in {self.valid_years[np.argmax(self.national_mean)]}")
#         print(
#             f"  - Minimum annual precipitation: {self.national_mean.min():.1f} mm in {self.valid_years[np.argmin(self.national_mean)]}")
#
#         return results_df
#
#     def run_complete_analysis(self):
#         """Run complete analysis workflow"""
#         print("=" * 70)
#         print("China Mainland Precipitation Spatio-Temporal Evolution Analysis")
#         print("=" * 70)
#
#         # 1. Load data
#         if not self.load_data():
#             return False
#
#         # 2. Preprocess data
#         if not self.preprocess_precipitation_data():
#             return False
#
#         # 3. Calculate annual precipitation
#         if not self.calculate_annual_precipitation():
#             return False
#
#         # 4. Analyze trends
#         if not self.analyze_national_precipitation_trends():
#             return False
#
#         # 5. Generate individual plots
#         if not self.create_individual_plots():
#             return False
#
#         # 6. Export results
#         results = self.export_results()
#
#         print("\n" + "=" * 70)
#         print("Analysis completed successfully!")
#         print("=" * 70)
#         print(f"\nGenerated files in '{self.output_dir}':")
#         print("  Data files:")
#         print("    - precipitation_analysis_results.csv")
#         print("    - precipitation_analysis_results.xlsx")
#         print("  Figure files:")
#         print("    - Fig1_National_Average_Precipitation_Trend.png")
#         print("    - Fig2_Precipitation_Spatial_Standard_Deviation.png")
#         print("    - Fig3_Annual_Precipitation_Change_Rate.png")
#         print("    - Fig4_Precipitation_Distribution_by_Decades.png")
#         print("    - Fig5_Time_Series_Decomposition.png")
#         print("    - Fig6_Precipitation_Mean_vs_Variability_Correlation.png")
#
#         return results
#
#     def __del__(self):
#         """Destructor to close dataset"""
#         if self.dataset:
#             self.dataset.close()
#
#
# # Main function
# def main():
#     file_path = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
#     output_dir = r"D:\Pythonpro\2024_D\result\Q1"
#
#     # Create analyzer instance
#     analyzer = PrecipitationAnalyzer(file_path, output_dir)
#
#     # Run complete analysis
#     results = analyzer.run_complete_analysis()
#
#     if results is not None:
#         print(f"\n✓ Analysis completed successfully!")
#         print(f"✓ All files saved to: {output_dir}")
#     else:
#         print(f"\n✗ Analysis failed. Please check file path and data format.")
#
#
# if __name__ == "__main__":
#     main()


# import netCDF4 as nc
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import os
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 设置英文环境和图表样式
# plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['xtick.labelsize'] = 9
# plt.rcParams['ytick.labelsize'] = 9
# plt.rcParams['legend.fontsize'] = 9
#
#
# class PrecipitationAnalyzer1990_2020:
#     def __init__(self, file_path, output_dir):
#         """Initialize precipitation analyzer for 1990-2020 period"""
#         self.file_path = file_path
#         self.output_dir = output_dir
#         self.dataset = None
#         self.precipitation_data = None
#         self.time_data = None
#         self.longitude = None
#         self.latitude = None
#
#         # Focus on 1990-2020 period (31 years)
#         self.target_years = list(range(1990, 2021))  # 1990-2020
#         self.analysis_period = "1990-2020"
#
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
#
#     def load_data(self):
#         """Load NC file data"""
#         print("Loading precipitation data for 1990-2020 analysis...")
#
#         try:
#             self.dataset = nc.Dataset(self.file_path, 'r')
#
#             # Read coordinate and time data
#             self.longitude = self.dataset.variables['longitude'][:]
#             self.latitude = self.dataset.variables['latitude'][:]
#             self.time_data = self.dataset.variables['time'][:]
#
#             # Convert time format
#             time_units = self.dataset.variables['time'].units
#             self.time_dates = nc.num2date(self.time_data, time_units)
#
#             # Filter data for 1990-2020 period
#             years_array = np.array([d.year for d in self.time_dates])
#             self.target_mask = np.isin(years_array, self.target_years)
#
#             print(f"Data loaded successfully:")
#             print(f"  - Full dataset time range: {self.time_dates[0].year} - {self.time_dates[-1].year}")
#             print(
#                 f"  - Analysis period: {self.target_years[0]} - {self.target_years[-1]} ({len(self.target_years)} years)")
#             print(f"  - Selected time points: {np.sum(self.target_mask)} out of {len(self.time_dates)}")
#             print(f"  - Spatial range: Longitude {self.longitude.min():.2f}° - {self.longitude.max():.2f}°")
#             print(f"  - Spatial range: Latitude {self.latitude.min():.2f}° - {self.latitude.max():.2f}°")
#             print(f"  - Full data shape: {self.dataset.variables['pre'].shape}")
#
#             return True
#
#         except Exception as e:
#             print(f"Data loading failed: {e}")
#             return False
#
#     def preprocess_precipitation_data(self):
#         """Preprocess precipitation data for 1990-2020, handle missing values"""
#         print(f"\nPreprocessing precipitation data for {self.analysis_period}...")
#
#         # Read precipitation data for the target period only
#         prec_var = self.dataset.variables['pre']
#
#         print("Extracting data for 1990-2020 period...")
#         # Extract only the target years to save memory
#         self.precipitation_data = prec_var[self.target_mask, :, :]
#         self.filtered_time_dates = [self.time_dates[i] for i in range(len(self.time_dates)) if self.target_mask[i]]
#
#         print(f"Extracted data shape: {self.precipitation_data.shape}")
#
#         # Check data quality for the target period
#         print("Checking data quality for 1990-2020...")
#         sample_data = self.precipitation_data[0:50, :, :]  # First 50 days as sample
#         print(f"  - Sample data range: {sample_data.min():.3f} - {sample_data.max():.3f}")
#         print(f"  - Negative values count: {np.sum(sample_data < 0)}")
#         print(f"  - -99.9 values count: {np.sum(np.abs(sample_data + 99.9) < 0.1)}")
#
#         # Handle missing values: set -99.9 and other negative values to 0
#         print("Processing missing values...")
#         original_negatives = np.sum(self.precipitation_data < 0)
#         self.precipitation_data[self.precipitation_data < 0] = 0.0
#
#         print(f"  - Set {original_negatives} negative values (missing values) to 0")
#         print(
#             f"  - Data range after processing: {self.precipitation_data.min():.3f} - {self.precipitation_data.max():.3f}")
#
#         return True
#
#     def calculate_annual_precipitation(self):
#         """Calculate annual precipitation for 1990-2020"""
#         print(f"\nCalculating annual precipitation for {self.analysis_period}...")
#
#         # Convert filtered time to year array
#         years_array = np.array([d.year for d in self.filtered_time_dates])
#
#         # Initialize result array for 31 years (1990-2020)
#         annual_precip = np.zeros((len(self.target_years), len(self.latitude), len(self.longitude)))
#
#         # Calculate annual total precipitation by year
#         for i, year in enumerate(self.target_years):
#             year_mask = years_array == year
#             if np.sum(year_mask) > 0:
#                 # Calculate total precipitation for this year
#                 annual_precip[i, :, :] = np.sum(self.precipitation_data[year_mask, :, :], axis=0)
#
#                 if i < 5 or i % 5 == 0:  # Show progress every 5 years
#                     print(f"  - Processed year {year}, {np.sum(year_mask)} days")
#
#         self.annual_precipitation = annual_precip
#         print(f"Annual precipitation calculation completed for {self.analysis_period}")
#         print(f"Result data shape: {annual_precip.shape}")
#         return True
#
#     def analyze_national_precipitation_trends(self):
#         """Analyze national precipitation trends for 1990-2020"""
#         print(f"\nAnalyzing national precipitation trends for {self.analysis_period}...")
#
#         # Calculate annual national statistics
#         national_annual_mean = []
#         national_annual_std = []
#         valid_years = []
#
#         for i, year in enumerate(self.target_years):
#             year_data = self.annual_precipitation[i, :, :]
#
#             # Only consider valid data points (areas with precipitation > 0)
#             # This excludes ocean areas and missing data regions
#             valid_data = year_data[year_data > 0]
#
#             if len(valid_data) > 100:  # Ensure sufficient valid data points
#                 # P_t: National average precipitation
#                 mean_precip = np.mean(valid_data)
#                 # σ_Pt: Spatial standard deviation
#                 std_precip = np.std(valid_data)
#
#                 national_annual_mean.append(mean_precip)
#                 national_annual_std.append(std_precip)
#                 valid_years.append(year)
#
#         self.national_mean = np.array(national_annual_mean)
#         self.national_std = np.array(national_annual_std)
#         self.valid_years = np.array(valid_years)
#
#         # Calculate annual change rate r_t
#         self.annual_change_rate = np.zeros(len(self.national_mean) - 1)
#         for i in range(len(self.national_mean) - 1):
#             if self.national_mean[i] != 0:
#                 self.annual_change_rate[i] = (self.national_mean[i + 1] - self.national_mean[i]) / self.national_mean[
#                     i] * 100
#
#         # Calculate long-term trend
#         if len(self.valid_years) > 2:
#             self.trend_slope, self.trend_intercept = np.polyfit(self.valid_years, self.national_mean, 1)
#             self.std_trend_slope = np.polyfit(self.valid_years, self.national_std, 1)[0]
#         else:
#             self.trend_slope = 0
#             self.std_trend_slope = 0
#
#         print(f"Analysis completed for {self.analysis_period}:")
#         print(f"  - Valid years count: {len(valid_years)}")
#         print(f"  - Average annual precipitation: {self.national_mean.mean():.1f} ± {self.national_mean.std():.1f} mm")
#         print(f"  - Precipitation range: {self.national_mean.min():.1f} - {self.national_mean.max():.1f} mm")
#         print(f"  - Long-term trend: {self.trend_slope:.2f} mm/year")
#         print(f"  - Average spatial variability: {self.national_std.mean():.1f} ± {self.national_std.std():.1f} mm")
#         print(f"  - Spatial variability trend: {self.std_trend_slope:.2f} mm/year")
#         print(
#             f"  - Annual change rate range: {self.annual_change_rate.min():.2f}% - {self.annual_change_rate.max():.2f}%")
#
#         return True
#
#     def create_individual_plots(self):
#         """Create individual analysis plots for 1990-2020"""
#         print(f"\nGenerating analysis plots for {self.analysis_period}...")
#
#         # 1. National annual average precipitation trend (P_t)
#         plt.figure(figsize=(12, 8))
#         plt.plot(self.valid_years, self.national_mean, 'b-', linewidth=2.5, marker='o', markersize=6,
#                  alpha=0.8, markerfacecolor='lightblue', markeredgecolor='blue')
#
#         # Add trend line
#         trend_line = self.trend_slope * self.valid_years + self.trend_intercept
#         plt.plot(self.valid_years, trend_line, "r--", alpha=0.8, linewidth=2.5,
#                  label=f'Linear Trend: {self.trend_slope:.2f} mm/year')
#
#         plt.title(f'National Annual Average Precipitation Trend ({self.analysis_period})',
#                   fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Annual Precipitation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#
#         # Add statistics text
#         mean_val = np.mean(self.national_mean)
#         std_val = np.std(self.national_mean)
#         max_year = self.valid_years[np.argmax(self.national_mean)]
#         min_year = self.valid_years[np.argmin(self.national_mean)]
#
#         stats_text = f'Period: {self.analysis_period}\nMean: {mean_val:.1f} mm\nStd Dev: {std_val:.1f} mm\n'
#         stats_text += f'Trend: {self.trend_slope:.2f} mm/year\nMax: {max_year} ({self.national_mean.max():.1f} mm)\n'
#         stats_text += f'Min: {min_year} ({self.national_mean.min():.1f} mm)'
#
#         plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9), fontsize=9)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir,
#                                  f'Fig1_National_Average_Precipitation_Trend_{self.analysis_period.replace("-", "_")}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 2. Precipitation spatial standard deviation (σ_Pt)
#         plt.figure(figsize=(12, 8))
#         plt.plot(self.valid_years, self.national_std, 'g-', linewidth=2.5, marker='s', markersize=6,
#                  alpha=0.8, markerfacecolor='lightgreen', markeredgecolor='green')
#
#         # Add trend line
#         std_trend_line = self.std_trend_slope * self.valid_years + np.polyfit(self.valid_years, self.national_std, 1)[1]
#         plt.plot(self.valid_years, std_trend_line, "r--", alpha=0.8, linewidth=2.5,
#                  label=f'Linear Trend: {self.std_trend_slope:.2f} mm/year')
#
#         plt.title(f'Spatial Standard Deviation of Precipitation ({self.analysis_period})',
#                   fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Standard Deviation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#
#         # Add statistics text
#         mean_std = np.mean(self.national_std)
#         std_std = np.std(self.national_std)
#         max_std_year = self.valid_years[np.argmax(self.national_std)]
#         min_std_year = self.valid_years[np.argmin(self.national_std)]
#
#         stats_text = f'Period: {self.analysis_period}\nMean: {mean_std:.1f} mm\nStd Dev: {std_std:.1f} mm\n'
#         stats_text += f'Trend: {self.std_trend_slope:.2f} mm/year\nMax: {max_std_year} ({self.national_std.max():.1f} mm)\n'
#         stats_text += f'Min: {min_std_year} ({self.national_std.min():.1f} mm)'
#
#         plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9), fontsize=9)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir,
#                                  f'Fig2_Precipitation_Spatial_Standard_Deviation_{self.analysis_period.replace("-", "_")}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 3. Annual change rate (r_t)
#         plt.figure(figsize=(12, 8))
#         change_years = self.valid_years[1:]  # Change rate has one less year
#         colors = ['crimson' if x < 0 else 'steelblue' for x in self.annual_change_rate]
#         bars = plt.bar(change_years, self.annual_change_rate, color=colors, alpha=0.8, width=0.7, edgecolor='black',
#                        linewidth=0.5)
#
#         plt.title(f'Annual Precipitation Change Rate ({self.analysis_period})',
#                   fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Change Rate (%)', fontsize=12)
#         plt.grid(True, alpha=0.3, axis='y')
#         plt.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
#
#         # Add legend for colors
#         from matplotlib.patches import Patch
#         legend_elements = [Patch(facecolor='steelblue', alpha=0.8, label='Increase'),
#                            Patch(facecolor='crimson', alpha=0.8, label='Decrease')]
#         plt.legend(handles=legend_elements, loc='upper right')
#
#         # Add statistics text
#         mean_change = np.mean(self.annual_change_rate)
#         max_increase = np.max(self.annual_change_rate)
#         max_decrease = np.min(self.annual_change_rate)
#         positive_years = np.sum(self.annual_change_rate > 0)
#         negative_years = np.sum(self.annual_change_rate < 0)
#
#         stats_text = f'Period: {self.analysis_period[:-5]}-{int(self.analysis_period[-4:]) - 1}\nMean: {mean_change:.2f}%\n'
#         stats_text += f'Max increase: {max_increase:.2f}%\nMax decrease: {max_decrease:.2f}%\n'
#         stats_text += f'Increase years: {positive_years}\nDecrease years: {negative_years}'
#
#         plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9), fontsize=9)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir,
#                                  f'Fig3_Annual_Precipitation_Change_Rate_{self.analysis_period.replace("-", "_")}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 4. Precipitation distribution by decades (1990s, 2000s, 2010s, 2020)
#         plt.figure(figsize=(12, 8))
#
#         # Group by decades within 1990-2020
#         decade_data = []
#         decade_labels = []
#
#         # 1990s (1990-1999)
#         decade_1990s = self.national_mean[(self.valid_years >= 1990) & (self.valid_years <= 1999)]
#         if len(decade_1990s) > 0:
#             decade_data.append(decade_1990s)
#             decade_labels.append('1990s\n(1990-1999)')
#
#         # 2000s (2000-2009)
#         decade_2000s = self.national_mean[(self.valid_years >= 2000) & (self.valid_years <= 2009)]
#         if len(decade_2000s) > 0:
#             decade_data.append(decade_2000s)
#             decade_labels.append('2000s\n(2000-2009)')
#
#         # 2010s (2010-2019)
#         decade_2010s = self.national_mean[(self.valid_years >= 2010) & (self.valid_years <= 2019)]
#         if len(decade_2010s) > 0:
#             decade_data.append(decade_2010s)
#             decade_labels.append('2010s\n(2010-2019)')
#
#         # 2020 (single year)
#         if 2020 in self.valid_years:
#             year_2020 = self.national_mean[self.valid_years == 2020]
#             decade_data.append(year_2020)
#             decade_labels.append('2020')
#
#         box_plot = plt.boxplot(decade_data, labels=decade_labels, patch_artist=True, widths=0.6)
#
#         # Color the boxes
#         colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
#         for i, (patch, color) in enumerate(zip(box_plot['boxes'], colors[:len(box_plot['boxes'])])):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.8)
#             patch.set_edgecolor('black')
#
#         plt.title(f'Precipitation Distribution by Decades ({self.analysis_period})',
#                   fontsize=14, fontweight='bold', pad=20)
#         plt.xlabel('Time Period', fontsize=12)
#         plt.ylabel('Annual Precipitation (mm)', fontsize=12)
#         plt.grid(True, alpha=0.3, axis='y')
#
#         # Add sample size annotations
#         for i, data in enumerate(decade_data):
#             plt.text(i + 1, plt.ylim()[1] * 0.95, f'n={len(data)}', ha='center', fontsize=9,
#                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir,
#                                  f'Fig4_Precipitation_Distribution_by_Decades_{self.analysis_period.replace("-", "_")}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         # 5. Detailed trend analysis with confidence intervals
#         plt.figure(figsize=(14, 10))
#
#         # Subplot 1: Original time series with trend and confidence interval
#         plt.subplot(2, 2, 1)
#         plt.plot(self.valid_years, self.national_mean, 'bo-', linewidth=2, markersize=5, alpha=0.8, label='Annual Data')
#         plt.plot(self.valid_years, trend_line, "r-", linewidth=2.5,
#                  label=f'Linear Trend: {self.trend_slope:.2f} mm/year')
#
#         # Add 95% confidence interval for trend (simplified)
#         residuals = self.national_mean - trend_line
#         std_residual = np.std(residuals)
#         ci_upper = trend_line + 1.96 * std_residual
#         ci_lower = trend_line - 1.96 * std_residual
#         plt.fill_between(self.valid_years, ci_lower, ci_upper, alpha=0.2, color='red', label='95% CI')
#
#         plt.title('Precipitation Trend with Confidence Interval', fontsize=11, fontweight='bold')
#         plt.ylabel('Precipitation (mm)')
#         plt.legend(fontsize=8)
#         plt.grid(True, alpha=0.3)
#
#         # Subplot 2: 5-year moving average
#         plt.subplot(2, 2, 2)
#         window_size = 5
#         if len(self.national_mean) >= window_size:
#             moving_avg = pd.Series(self.national_mean).rolling(window=window_size, center=True).mean()
#             plt.plot(self.valid_years, self.national_mean, 'lightblue', alpha=0.6, linewidth=1, label='Annual values')
#             plt.plot(self.valid_years, moving_avg, 'navy', linewidth=3, label=f'{window_size}-year moving average')
#             plt.title('Smoothed Trend Analysis', fontsize=11, fontweight='bold')
#             plt.ylabel('Precipitation (mm)')
#             plt.legend(fontsize=8)
#             plt.grid(True, alpha=0.3)
#
#         # Subplot 3: Detrended series (residuals)
#         plt.subplot(2, 2, 3)
#         plt.plot(self.valid_years, residuals, 'go-', linewidth=2, markersize=4, alpha=0.8)
#         plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
#         plt.axhline(y=np.mean(residuals) + np.std(residuals), color='red', linestyle=':', alpha=0.7, label='+1σ')
#         plt.axhline(y=np.mean(residuals) - np.std(residuals), color='red', linestyle=':', alpha=0.7, label='-1σ')
#         plt.title('Detrended Series (Residuals)', fontsize=11, fontweight='bold')
#         plt.xlabel('Year')
#         plt.ylabel('Residual (mm)')
#         plt.legend(fontsize=8)
#         plt.grid(True, alpha=0.3)
#
#         # Subplot 4: Correlation analysis
#         plt.subplot(2, 2, 4)
#         plt.scatter(self.national_mean, self.national_std, alpha=0.7, s=60, c=self.valid_years,
#                     cmap='viridis', edgecolors='black', linewidth=0.5)
#
#         # Add correlation line
#         correlation = np.corrcoef(self.national_mean, self.national_std)[0, 1]
#         z_corr = np.polyfit(self.national_mean, self.national_std, 1)
#         p_corr = np.poly1d(z_corr)
#         x_corr = np.linspace(self.national_mean.min(), self.national_mean.max(), 100)
#         plt.plot(x_corr, p_corr(x_corr), 'r--', alpha=0.8, linewidth=2)
#
#         plt.colorbar(label='Year', shrink=0.8)
#         plt.title(f'Mean vs Variability\n(r = {correlation:.3f})', fontsize=11, fontweight='bold')
#         plt.xlabel('Average Precipitation (mm)')
#         plt.ylabel('Spatial Std Dev (mm)')
#         plt.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(self.output_dir, f'Fig5_Detailed_Trend_Analysis_{self.analysis_period.replace("-", "_")}.png'),
#             dpi=300, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         print(f"All analysis plots generated successfully for {self.analysis_period}!")
#         return True
#
#     def export_results(self):
#         """Export analysis results to CSV and XLSX for 1990-2020"""
#         print(f"\nExporting analysis results for {self.analysis_period}...")
#
#         # Create results DataFrame
#         results_df = pd.DataFrame({
#             'Year': self.valid_years,
#             'National_Average_Precipitation_mm': np.round(self.national_mean, 2),
#             'Spatial_Standard_Deviation_mm': np.round(self.national_std, 2)
#         })
#
#         # Add change rate (note: length differs by 1)
#         change_rate_full = np.full(len(self.valid_years), np.nan)
#         change_rate_full[1:] = np.round(self.annual_change_rate, 2)
#         results_df['Annual_Change_Rate_percent'] = change_rate_full
#
#         # Add additional calculated fields
#         results_df['Coefficient_of_Variation_percent'] = np.round(
#             (results_df['Spatial_Standard_Deviation_mm'] / results_df['National_Average_Precipitation_mm']) * 100, 2)
#
#         # Export to CSV
#         csv_path = os.path.join(self.output_dir,
#                                 f'precipitation_analysis_results_{self.analysis_period.replace("-", "_")}.csv')
#         results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
#
#         # Export to XLSX with multiple sheets
#         xlsx_path = os.path.join(self.output_dir,
#                                  f'precipitation_analysis_results_{self.analysis_period.replace("-", "_")}.xlsx')
#         with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
#             # Main results sheet
#             results_df.to_excel(writer, sheet_name='Annual_Results', index=False)
#
#             # Summary statistics sheet
#             summary_stats = pd.DataFrame({
#                 'Statistic': [
#                     'Analysis Period',
#                     'Total Years',
#                     'Mean Annual Precipitation (mm)',
#                     'Standard Deviation (mm)',
#                     'Minimum Precipitation (mm)',
#                     'Maximum Precipitation (mm)',
#                     'Linear Trend (mm/year)',
#                     'Trend Significance',
#                     'Wettest Year',
#                     'Driest Year',
#                     'Mean Spatial Variability (mm)',
#                     'Spatial Variability Trend (mm/year)',
#                     'Mean Annual Change Rate (%)',
#                     'Years with Increase',
#                     'Years with Decrease'
#                 ],
#                 'Value': [
#                     self.analysis_period,
#                     len(self.valid_years),
#                     f"{self.national_mean.mean():.1f}",
#                     f"{self.national_mean.std():.1f}",
#                     f"{self.national_mean.min():.1f}",
#                     f"{self.national_mean.max():.1f}",
#                     f"{self.trend_slope:.2f}",
#                     "Positive" if self.trend_slope > 0 else "Negative" if self.trend_slope < 0 else "No trend",
#                     f"{self.valid_years[np.argmax(self.national_mean)]} ({self.national_mean.max():.1f} mm)",
#                     f"{self.valid_years[np.argmin(self.national_mean)]} ({self.national_mean.min():.1f} mm)",
#                     f"{self.national_std.mean():.1f}",
#                     f"{self.std_trend_slope:.2f}",
#                     f"{np.mean(self.annual_change_rate):.2f}",
#                     np.sum(self.annual_change_rate > 0),
#                     np.sum(self.annual_change_rate < 0)
#                 ]
#             })
#             summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
#
#             # Decade comparison
#             decade_comparison = []
#             for decade_start in [1990, 2000, 2010]:
#                 decade_end = min(decade_start + 9, 2020)
#                 decade_mask = (self.valid_years >= decade_start) & (self.valid_years <= decade_end)
#                 if np.sum(decade_mask) > 0:
#                     decade_data = self.national_mean[decade_mask]
#                     decade_comparison.append({
#                         'Decade': f"{decade_start}s ({decade_start}-{decade_end})",
#                         'Years': np.sum(decade_mask),
#                         'Mean_Precipitation_mm': f"{np.mean(decade_data):.1f}",
#                         'Std_Deviation_mm': f"{np.std(decade_data):.1f}",
#                         'Min_Precipitation_mm': f"{np.min(decade_data):.1f}",
#                         'Max_Precipitation_mm': f"{np.max(decade_data):.1f}"
#                     })
#
#             # Add 2020 separately
#             if 2020 in self.valid_years:
#                 data_2020 = self.national_mean[self.valid_years == 2020][0]
#                 decade_comparison.append({
#                     'Decade': "2020",
#                     'Years': 1,
#                     'Mean_Precipitation_mm': f"{data_2020:.1f}",
#                     'Std_Deviation_mm': "N/A",
#                     'Min_Precipitation_mm': f"{data_2020:.1f}",
#                     'Max_Precipitation_mm': f"{data_2020:.1f}"
#                 })
#
#             pd.DataFrame(decade_comparison).to_excel(writer, sheet_name='Decade_Comparison', index=False)
#
#         # Print comprehensive summary
#         print(f"\nAnalysis results exported to:")
#         print(f"  - CSV: {csv_path}")
#         print(f"  - XLSX: {xlsx_path}")
#
#         # Display summary for 1990-2020 period
#         print(f"\n{'=' * 60}")
#         print(f"PRECIPITATION ANALYSIS SUMMARY ({self.analysis_period})")
#         print(f"{'=' * 60}")
#         print(f"Analysis Period: {self.analysis_period} ({len(self.valid_years)} years)")
#         print(
#             f"Spatial Coverage: {self.longitude.min():.2f}° - {self.longitude.max():.2f}°E, {self.latitude.min():.2f}° - {self.latitude.max():.2f}°N")
#         print(f"\nPRECIPITATION STATISTICS:")
#         print(f"  • Average annual precipitation: {self.national_mean.mean():.1f} ± {self.national_mean.std():.1f} mm")
#         print(f"  • Range: {self.national_mean.min():.1f} - {self.national_mean.max():.1f} mm")
#         print(
#             f"  • Wettest year: {self.valid_years[np.argmax(self.national_mean)]} ({self.national_mean.max():.1f} mm)")
#         print(f"  • Driest year: {self.valid_years[np.argmin(self.national_mean)]} ({self.national_mean.min():.1f} mm)")
#
#         print(f"\nTREND ANALYSIS:")
#         print(f"  • Linear trend: {self.trend_slope:.2f} mm/year")
#         trend_total = self.trend_slope * (len(self.valid_years) - 1)
#         print(f"  • Total change over period: {trend_total:.1f} mm")
#         print(
#             f"  • Trend direction: {'Increasing' if self.trend_slope > 0 else 'Decreasing' if self.trend_slope < 0 else 'No clear trend'}")
#
#         print(f"\nSPATIAL VARIABILITY:")
#         print(f"  • Average spatial std deviation: {self.national_std.mean():.1f} ± {self.national_std.std():.1f} mm")
#         print(f"  • Spatial variability trend: {self.std_trend_slope:.2f} mm/year")
#
#         print(f"\nINTER-ANNUAL VARIABILITY:")
#         print(f"  • Average annual change rate: {np.mean(self.annual_change_rate):.2f}%")
#         print(f"  • Maximum annual increase: {np.max(self.annual_change_rate):.2f}%")
#         print(f"  • Maximum annual decrease: {np.min(self.annual_change_rate):.2f}%")
#         print(f"  • Years with precipitation increase: {np.sum(self.annual_change_rate > 0)}")
#         print(f"  • Years with precipitation decrease: {np.sum(self.annual_change_rate < 0)}")
#
#         # Display first 10 years of results
#         print(f"\nFIRST 10 YEARS OF DETAILED RESULTS:")
#         print(results_df.head(10).to_string(index=False, float_format='%.2f'))
#
#         return results_df
#
#     def run_complete_analysis(self):
#         """Run complete analysis workflow for 1990-2020"""
#         print("=" * 80)
#         print("CHINA MAINLAND PRECIPITATION SPATIO-TEMPORAL EVOLUTION ANALYSIS")
#         print(f"FOCUS PERIOD: {self.analysis_period}")
#         print("=" * 80)
#
#         # 1. Load data
#         if not self.load_data():
#             return False
#
#         # 2. Preprocess data for target period
#         if not self.preprocess_precipitation_data():
#             return False
#
#         # 3. Calculate annual precipitation
#         if not self.calculate_annual_precipitation():
#             return False
#
#         # 4. Analyze trends
#         if not self.analyze_national_precipitation_trends():
#             return False
#
#         # 5. Generate individual plots
#         if not self.create_individual_plots():
#             return False
#
#         # 6. Export results
#         results = self.export_results()
#
#         print(f"\n{'=' * 80}")
#         print("ANALYSIS COMPLETED SUCCESSFULLY!")
#         print(f"{'=' * 80}")
#         print(f"\nGenerated files in '{self.output_dir}':")
#         print("  📊 DATA FILES:")
#         print(f"    - precipitation_analysis_results_{self.analysis_period.replace('-', '_')}.csv")
#         print(f"    - precipitation_analysis_results_{self.analysis_period.replace('-', '_')}.xlsx")
#         print("  📈 FIGURE FILES:")
#         print(f"    - Fig1_National_Average_Precipitation_Trend_{self.analysis_period.replace('-', '_')}.png")
#         print(f"    - Fig2_Precipitation_Spatial_Standard_Deviation_{self.analysis_period.replace('-', '_')}.png")
#         print(f"    - Fig3_Annual_Precipitation_Change_Rate_{self.analysis_period.replace('-', '_')}.png")
#         print(f"    - Fig4_Precipitation_Distribution_by_Decades_{self.analysis_period.replace('-', '_')}.png")
#         print(f"    - Fig5_Detailed_Trend_Analysis_{self.analysis_period.replace('-', '_')}.png")
#
#         print(f"\n🎯 Analysis focused on {self.analysis_period} period for consistency with land use data!")
#
#         return results
#
#     def __del__(self):
#         """Destructor to close dataset"""
#         if self.dataset:
#             self.dataset.close()
#
#
# # Main function
# def main():
#     file_path = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
#     output_dir = r"D:\Pythonpro\2024_D\result\Q1"
#
#     # Create analyzer instance for 1990-2020 period
#     analyzer = PrecipitationAnalyzer1990_2020(file_path, output_dir)
#
#     # Run complete analysis
#     results = analyzer.run_complete_analysis()
#
#     if results is not None:
#         print(f"\n✅ Analysis completed successfully for 1990-2020 period!")
#         print(f"✅ All files saved to: {output_dir}")
#         print(f"✅ Analysis period chosen to align with land use data (1900-2019)")
#     else:
#         print(f"\n❌ Analysis failed. Please check file path and data format.")
#
#
# if __name__ == "__main__":
#     main()

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import os
import warnings

warnings.filterwarnings('ignore')

# 设置英文环境和图表样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


class PrecipitationSpatialAnalyzer:
    def __init__(self, file_path, output_dir):
        """Initialize spatial precipitation analyzer for 1990-2020 period"""
        self.file_path = file_path
        self.output_dir = output_dir
        self.dataset = None
        self.precipitation_data = None
        self.longitude = None
        self.latitude = None
        self.target_years = list(range(1990, 2021))  # 1990-2020
        self.analysis_period = "1990-2020"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_and_process_data(self):
        """Load and process precipitation data for spatial analysis"""
        print(f"Loading precipitation data for spatial analysis ({self.analysis_period})...")

        try:
            self.dataset = nc.Dataset(self.file_path, 'r')

            # Read coordinate data
            self.longitude = self.dataset.variables['longitude'][:]
            self.latitude = self.dataset.variables['latitude'][:]

            # Read time data and filter for target years
            time_data = self.dataset.variables['time'][:]
            time_units = self.dataset.variables['time'].units
            time_dates = nc.num2date(time_data, time_units)

            # Create mask for target years
            years_array = np.array([d.year for d in time_dates])
            target_mask = np.isin(years_array, self.target_years)

            print(f"Data overview:")
            print(f"  - Analysis period: {self.target_years[0]} - {self.target_years[-1]}")
            print(f"  - Longitude range: {self.longitude.min():.2f}° - {self.longitude.max():.2f}°E")
            print(f"  - Latitude range: {self.latitude.min():.2f}° - {self.latitude.max():.2f}°N")
            print(
                f"  - Spatial resolution: {abs(self.longitude[1] - self.longitude[0]):.3f}° × {abs(self.latitude[1] - self.latitude[0]):.3f}°")
            print(f"  - Selected time points: {np.sum(target_mask)} out of {len(time_dates)}")

            # Read precipitation data for target period
            print("Loading precipitation data...")
            prec_var = self.dataset.variables['pre']
            precipitation_filtered = prec_var[target_mask, :, :]

            # Handle missing values
            print("Processing missing values...")
            original_negatives = np.sum(precipitation_filtered < 0)
            precipitation_filtered[precipitation_filtered < 0] = 0.0
            print(f"  - Processed {original_negatives} missing values (set to 0)")

            # Calculate annual precipitation for each year
            print("Calculating annual precipitation for each grid point...")
            filtered_time_dates = [time_dates[i] for i in range(len(time_dates)) if target_mask[i]]
            years_filtered = np.array([d.year for d in filtered_time_dates])

            # Initialize array for annual precipitation
            annual_precipitation = np.zeros((len(self.target_years), len(self.latitude), len(self.longitude)))

            # Calculate annual totals for each year
            for i, year in enumerate(self.target_years):
                year_mask = years_filtered == year
                if np.sum(year_mask) > 0:
                    annual_precipitation[i, :, :] = np.sum(precipitation_filtered[year_mask, :, :], axis=0)
                    if i % 5 == 0 or i < 3:
                        print(f"  - Processed year {year}: {np.sum(year_mask)} days")

            # Calculate 31-year average precipitation for each grid point
            print("Calculating 31-year average precipitation...")
            self.mean_precipitation_spatial = np.mean(annual_precipitation, axis=0)

            # Mask out areas with very low precipitation (likely ocean or missing data)
            self.mean_precipitation_spatial[self.mean_precipitation_spatial < 10] = np.nan

            print(f"Spatial analysis data ready:")
            print(
                f"  - Mean precipitation range: {np.nanmin(self.mean_precipitation_spatial):.1f} - {np.nanmax(self.mean_precipitation_spatial):.1f} mm/year")
            print(
                f"  - Valid grid points: {np.sum(~np.isnan(self.mean_precipitation_spatial))} out of {self.mean_precipitation_spatial.size}")

            return True

        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

    def create_spatial_precipitation_map(self):
        """Create spatial distribution map of 31-year average precipitation"""
        print("\nGenerating spatial precipitation distribution map...")

        # Create figure with Cartopy projection
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set extent to cover China with some buffer
        china_extent = [70, 140, 15, 55]  # [west, east, south, north]
        ax.set_extent(china_extent, crs=ccrs.PlateCarree())

        # Add map features
        print("Adding geographic features...")

        # Add country borders (China boundary)
        ax.add_feature(cfeature.BORDERS, linewidth=2, edgecolor='black', alpha=0.8)

        # Add coastlines
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', alpha=0.8)

        # Add land and ocean features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

        # Create custom colormap for precipitation
        colors = ['#8B4513', '#DEB887', '#F0E68C', '#ADFF2F', '#32CD32',
                  '#00FF00', '#00CED1', '#1E90FF', '#0000FF', '#4B0082', '#800080']
        n_bins = 20
        cmap = mcolors.LinearSegmentedColormap.from_list('precipitation', colors, N=n_bins)

        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(self.longitude, self.latitude)

        # Plot precipitation data
        print("Plotting precipitation data...")
        precipitation_plot = ax.contourf(lon_grid, lat_grid, self.mean_precipitation_spatial,
                                         levels=20, cmap=cmap, extend='both',
                                         transform=ccrs.PlateCarree(), alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(precipitation_plot, ax=ax, orientation='horizontal',
                            pad=0.05, shrink=0.8, aspect=40)
        cbar.set_label('Mean Annual Precipitation (mm/year)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        # Set title
        plt.title(f'Spatial Distribution of Mean Annual Precipitation in China ({self.analysis_period})\n' +
                  f'31-Year Average Based on Daily Data',
                  fontsize=14, fontweight='bold', pad=20)

        # Add statistics text box
        valid_data = self.mean_precipitation_spatial[~np.isnan(self.mean_precipitation_spatial)]
        stats_text = f'Statistics for {self.analysis_period}:\n'
        stats_text += f'• Mean: {np.mean(valid_data):.1f} mm/year\n'
        stats_text += f'• Std Dev: {np.std(valid_data):.1f} mm/year\n'
        stats_text += f'• Range: {np.min(valid_data):.1f} - {np.max(valid_data):.1f} mm/year\n'
        stats_text += f'• Valid grid points: {len(valid_data):,}'

        # Add text box in the upper right corner
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                fontsize=9, fontfamily='monospace')

        # Save the figure
        output_path = os.path.join(self.output_dir,
                                   f'Fig_Spatial_Precipitation_Distribution_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()

        print(f"Spatial precipitation map saved: {output_path}")
        return True

    def create_detailed_regional_maps(self):
        """Create detailed regional maps for different parts of China"""
        print("\nGenerating detailed regional precipitation maps...")

        # Define regions
        regions = {
            'Northern_China': {'extent': [110, 130, 35, 50], 'title': 'Northern China'},
            'Southern_China': {'extent': [105, 125, 20, 35], 'title': 'Southern China'},
            'Western_China': {'extent': [75, 105, 25, 45], 'title': 'Western China'},
            'Eastern_China': {'extent': [115, 135, 25, 42], 'title': 'Eastern China'}
        }

        for region_name, region_info in regions.items():
            fig = plt.figure(figsize=(12, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())

            # Set extent for the region
            ax.set_extent(region_info['extent'], crs=ccrs.PlateCarree())

            # Add map features
            ax.add_feature(cfeature.BORDERS, linewidth=2, edgecolor='black', alpha=0.8)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', alpha=0.8)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

            # Create custom colormap
            colors = ['#8B4513', '#DEB887', '#F0E68C', '#ADFF2F', '#32CD32',
                      '#00FF00', '#00CED1', '#1E90FF', '#0000FF', '#4B0082', '#800080']
            cmap = mcolors.LinearSegmentedColormap.from_list('precipitation', colors, N=20)

            # Create meshgrid
            lon_grid, lat_grid = np.meshgrid(self.longitude, self.latitude)

            # Plot precipitation data
            precipitation_plot = ax.contourf(lon_grid, lat_grid, self.mean_precipitation_spatial,
                                             levels=15, cmap=cmap, extend='both',
                                             transform=ccrs.PlateCarree(), alpha=0.8)

            # Add colorbar
            cbar = plt.colorbar(precipitation_plot, ax=ax, orientation='vertical',
                                pad=0.02, shrink=0.8)
            cbar.set_label('Mean Annual Precipitation (mm/year)', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)

            # Add gridlines
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}

            # Set title
            plt.title(f'{region_info["title"]}: Mean Annual Precipitation ({self.analysis_period})',
                      fontsize=13, fontweight='bold', pad=15)

            # Calculate regional statistics
            extent = region_info['extent']
            lon_mask = (self.longitude >= extent[0]) & (self.longitude <= extent[1])
            lat_mask = (self.latitude >= extent[2]) & (self.latitude <= extent[3])

            # Create 2D masks
            lon_indices = np.where(lon_mask)[0]
            lat_indices = np.where(lat_mask)[0]

            if len(lon_indices) > 0 and len(lat_indices) > 0:
                regional_data = self.mean_precipitation_spatial[np.ix_(lat_indices, lon_indices)]
                valid_regional_data = regional_data[~np.isnan(regional_data)]

                if len(valid_regional_data) > 0:
                    stats_text = f'Regional Statistics:\n'
                    stats_text += f'• Mean: {np.mean(valid_regional_data):.1f} mm/year\n'
                    stats_text += f'• Std: {np.std(valid_regional_data):.1f} mm/year\n'
                    stats_text += f'• Range: {np.min(valid_regional_data):.1f} - {np.max(valid_regional_data):.1f} mm/year'

                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                            fontsize=9)

            # Save the figure
            output_path = os.path.join(self.output_dir,
                                       f'Fig_Regional_Precipitation_{region_name}_{self.analysis_period.replace("-", "_")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            plt.close()

            print(f"Regional map saved: {region_name}")

        return True

    def create_precipitation_gradient_analysis(self):
        """Create precipitation gradient analysis maps"""
        print("\nGenerating precipitation gradient analysis...")

        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16),
                                                     subplot_kw={'projection': ccrs.PlateCarree()})

        china_extent = [70, 140, 15, 55]

        # 1. Original precipitation map
        ax1.set_extent(china_extent, crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black', alpha=0.8)
        ax1.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', alpha=0.8)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        lon_grid, lat_grid = np.meshgrid(self.longitude, self.latitude)

        colors = ['#8B4513', '#DEB887', '#F0E68C', '#ADFF2F', '#32CD32',
                  '#00FF00', '#00CED1', '#1E90FF', '#0000FF', '#4B0082', '#800080']
        cmap = mcolors.LinearSegmentedColormap.from_list('precipitation', colors, N=20)

        p1 = ax1.contourf(lon_grid, lat_grid, self.mean_precipitation_spatial,
                          levels=15, cmap=cmap, extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
        ax1.set_title('Mean Annual Precipitation', fontsize=12, fontweight='bold')

        # Add colorbar for subplot 1
        cbar1 = plt.colorbar(p1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar1.set_label('Precipitation (mm/year)', fontsize=10)

        # 2. Precipitation categories
        ax2.set_extent(china_extent, crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black', alpha=0.8)
        ax2.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', alpha=0.8)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        # Create precipitation categories
        precip_categories = np.full_like(self.mean_precipitation_spatial, np.nan)
        valid_mask = ~np.isnan(self.mean_precipitation_spatial)

        precip_categories[valid_mask & (self.mean_precipitation_spatial < 200)] = 1  # Arid
        precip_categories[valid_mask & (self.mean_precipitation_spatial >= 200) & (
                    self.mean_precipitation_spatial < 400)] = 2  # Semi-arid
        precip_categories[valid_mask & (self.mean_precipitation_spatial >= 400) & (
                    self.mean_precipitation_spatial < 800)] = 3  # Semi-humid
        precip_categories[valid_mask & (self.mean_precipitation_spatial >= 800) & (
                    self.mean_precipitation_spatial < 1200)] = 4  # Humid
        precip_categories[valid_mask & (self.mean_precipitation_spatial >= 1200)] = 5  # Very humid

        category_colors = ['#8B4513', '#DEB887', '#32CD32', '#1E90FF', '#4B0082']
        category_cmap = ListedColormap(category_colors)

        p2 = ax2.contourf(lon_grid, lat_grid, precip_categories,
                          levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap=category_cmap,
                          transform=ccrs.PlateCarree(), alpha=0.8)
        ax2.set_title('Precipitation Zones', fontsize=12, fontweight='bold')

        # Add custom legend for categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=category_colors[0], label='Arid (<200 mm)'),
            Patch(facecolor=category_colors[1], label='Semi-arid (200-400 mm)'),
            Patch(facecolor=category_colors[2], label='Semi-humid (400-800 mm)'),
            Patch(facecolor=category_colors[3], label='Humid (800-1200 mm)'),
            Patch(facecolor=category_colors[4], label='Very humid (>1200 mm)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # 3. East-West gradient
        ax3.set_extent(china_extent, crs=ccrs.PlateCarree())
        ax3.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black', alpha=0.8)
        ax3.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', alpha=0.8)
        ax3.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        # Calculate longitude-averaged precipitation
        lon_avg_precip = np.nanmean(self.mean_precipitation_spatial, axis=0)
        lon_gradient = np.tile(lon_avg_precip, (len(self.latitude), 1))
        lon_gradient[np.isnan(self.mean_precipitation_spatial)] = np.nan

        p3 = ax3.contourf(lon_grid, lat_grid, lon_gradient,
                          levels=15, cmap='viridis', extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
        ax3.set_title('East-West Precipitation Gradient', fontsize=12, fontweight='bold')

        cbar3 = plt.colorbar(p3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar3.set_label('Longitude-averaged Precipitation (mm/year)', fontsize=10)

        # 4. North-South gradient
        ax4.set_extent(china_extent, crs=ccrs.PlateCarree())
        ax4.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black', alpha=0.8)
        ax4.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', alpha=0.8)
        ax4.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

        # Calculate latitude-averaged precipitation
        lat_avg_precip = np.nanmean(self.mean_precipitation_spatial, axis=1)
        lat_gradient = np.tile(lat_avg_precip.reshape(-1, 1), (1, len(self.longitude)))
        lat_gradient[np.isnan(self.mean_precipitation_spatial)] = np.nan

        p4 = ax4.contourf(lon_grid, lat_grid, lat_gradient,
                          levels=15, cmap='plasma', extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
        ax4.set_title('North-South Precipitation Gradient', fontsize=12, fontweight='bold')

        cbar4 = plt.colorbar(p4, ax=ax4, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar4.set_label('Latitude-averaged Precipitation (mm/year)', fontsize=10)

        # Add gridlines to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

        plt.suptitle(f'Precipitation Spatial Analysis for China ({self.analysis_period})',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(self.output_dir,
                                   f'Fig_Precipitation_Gradient_Analysis_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()

        print(f"Gradient analysis map saved: {output_path}")
        return True

    def export_spatial_data(self):
        """Export spatial precipitation data"""
        print("\nExporting spatial precipitation data...")

        # Create spatial data arrays
        lon_grid, lat_grid = np.meshgrid(self.longitude, self.latitude)

        # Flatten arrays for export
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        precip_flat = self.mean_precipitation_spatial.flatten()

        # Remove NaN values
        valid_mask = ~np.isnan(precip_flat)

        # Create DataFrame
        spatial_df = pd.DataFrame({
            'Longitude': lon_flat[valid_mask],
            'Latitude': lat_flat[valid_mask],
            'Mean_Annual_Precipitation_mm': np.round(precip_flat[valid_mask], 2)
        })

        # Sort by latitude then longitude
        spatial_df = spatial_df.sort_values(['Latitude', 'Longitude']).reset_index(drop=True)

        # Export to CSV
        csv_path = os.path.join(self.output_dir,
                                f'spatial_precipitation_data_{self.analysis_period.replace("-", "_")}.csv')
        spatial_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Export summary statistics by regions
        summary_stats = []

        # Calculate statistics for different longitude bands (East-West)
        for i, (lon_start, lon_end, region_name) in enumerate(
                [(70, 90, 'Western'), (90, 110, 'Central'), (110, 140, 'Eastern')]):
            mask = (spatial_df['Longitude'] >= lon_start) & (spatial_df['Longitude'] < lon_end)
            if mask.sum() > 0:
                region_data = spatial_df[mask]['Mean_Annual_Precipitation_mm']
                summary_stats.append({
                    'Region_Type': 'Longitude_Band',
                    'Region_Name': f'{region_name}_China',
                    'Longitude_Range': f'{lon_start}-{lon_end}°E',
                    'Grid_Points': len(region_data),
                    'Mean_Precipitation_mm': f"{region_data.mean():.1f}",
                    'Std_Precipitation_mm': f"{region_data.std():.1f}",
                    'Min_Precipitation_mm': f"{region_data.min():.1f}",
                    'Max_Precipitation_mm': f"{region_data.max():.1f}"
                })

        # Calculate statistics for different latitude bands (North-South)
        for i, (lat_start, lat_end, region_name) in enumerate(
                [(15, 30, 'Southern'), (30, 40, 'Central'), (40, 55, 'Northern')]):
            mask = (spatial_df['Latitude'] >= lat_start) & (spatial_df['Latitude'] < lat_end)
            if mask.sum() > 0:
                region_data = spatial_df[mask]['Mean_Annual_Precipitation_mm']
                summary_stats.append({
                    'Region_Type': 'Latitude_Band',
                    'Region_Name': f'{region_name}_China',
                    'Latitude_Range': f'{lat_start}-{lat_end}°N',
                    'Grid_Points': len(region_data),
                    'Mean_Precipitation_mm': f"{region_data.mean():.1f}",
                    'Std_Precipitation_mm': f"{region_data.std():.1f}",
                    'Min_Precipitation_mm': f"{region_data.min():.1f}",
                    'Max_Precipitation_mm': f"{region_data.max():.1f}"
                })

        summary_df = pd.DataFrame(summary_stats)

        # Export to XLSX with multiple sheets
        xlsx_path = os.path.join(self.output_dir,
                                 f'spatial_precipitation_analysis_{self.analysis_period.replace("-", "_")}.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            spatial_df.to_excel(writer, sheet_name='Spatial_Data', index=False)
            summary_df.to_excel(writer, sheet_name='Regional_Summary', index=False)

        print(f"Spatial data exported:")
        print(f"  - CSV: {csv_path}")
        print(f"  - XLSX: {xlsx_path}")
        print(f"  - Total valid grid points: {len(spatial_df):,}")

        return spatial_df

    def run_spatial_analysis(self):
        """Run complete spatial analysis"""
        print("=" * 80)
        print("CHINA PRECIPITATION SPATIAL DISTRIBUTION ANALYSIS")
        print(f"ANALYSIS PERIOD: {self.analysis_period}")
        print("=" * 80)

        # 1. Load and process data
        if not self.load_and_process_data():
            return False

        # 2. Create main spatial map
        if not self.create_spatial_precipitation_map():
            return False

        # 3. Create detailed regional maps
        if not self.create_detailed_regional_maps():
            return False

        # 4. Create gradient analysis
        if not self.create_precipitation_gradient_analysis():
            return False

        # 5. Export spatial data
        spatial_data = self.export_spatial_data()

        print(f"\n{'=' * 80}")
        print("SPATIAL ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"\nGenerated spatial analysis files in '{self.output_dir}':")
        print("  🗺️  SPATIAL MAPS:")
        print(f"    - Fig_Spatial_Precipitation_Distribution_{self.analysis_period.replace('-', '_')}.png")
        print(f"    - Fig_Regional_Precipitation_Northern_China_{self.analysis_period.replace('-', '_')}.png")
        print(f"    - Fig_Regional_Precipitation_Southern_China_{self.analysis_period.replace('-', '_')}.png")
        print(f"    - Fig_Regional_Precipitation_Western_China_{self.analysis_period.replace('-', '_')}.png")
        print(f"    - Fig_Regional_Precipitation_Eastern_China_{self.analysis_period.replace('-', '_')}.png")
        print(f"    - Fig_Precipitation_Gradient_Analysis_{self.analysis_period.replace('-', '_')}.png")
        print("  📊 SPATIAL DATA:")
        print(f"    - spatial_precipitation_data_{self.analysis_period.replace('-', '_')}.csv")
        print(f"    - spatial_precipitation_analysis_{self.analysis_period.replace('-', '_')}.xlsx")

        return spatial_data

    def __del__(self):
        """Destructor to close dataset"""
        if self.dataset:
            self.dataset.close()


# Updated main function to include spatial analysis
def main():
    file_path = r"C:\Users\ZEC\Desktop\CHM_PRE_0.25dg_19612022.nc"
    output_dir = r"D:\Pythonpro\2024_D\result\Q1"

    # Create spatial analyzer instance
    spatial_analyzer = PrecipitationSpatialAnalyzer(file_path, output_dir)

    # Run spatial analysis
    spatial_results = spatial_analyzer.run_spatial_analysis()

    if spatial_results is not None:
        print(f"\n✅ Spatial analysis completed successfully!")
        print(f"✅ China boundary lines included in all maps")
        print(f"✅ 31-year average precipitation (1990-2020) mapped")
        print(f"✅ All files saved to: {output_dir}")
    else:
        print(f"\n❌ Spatial analysis failed. Please check file path and data format.")


if __name__ == "__main__":
    main()