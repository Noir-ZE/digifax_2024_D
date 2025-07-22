import os
import glob
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# 设置英文环境和图表样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


class LandUseAnalyzerFixed:
    def __init__(self, data_dir, output_dir):
        """Initialize land use analyzer for 1990-2019 period"""
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Focus on 1990-2019 period (30 years)
        self.target_years = list(range(1990, 2020))  # 1990-2019
        self.analysis_period = "1990-2019"

        # Land use types
        self.land_use_types = {
            'cropland': 'Cropland',
            'forest': 'Forest',
            'grass': 'Grassland',
            'shrub': 'Shrubland',
            'wetland': 'Wetland'
        }

        self.land_use_colors = {
            'cropland': '#FFD700',  # Gold
            'forest': '#228B22',  # Forest Green
            'grass': '#9ACD32',  # Yellow Green
            'shrub': '#8FBC8F',  # Dark Sea Green
            'wetland': '#4682B4'  # Steel Blue
        }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data containers
        self.spatial_data = {}  # {year: {land_type: 2D_array}}
        self.longitude = None
        self.latitude = None

    def scan_and_organize_files(self):
        """Scan directory and organize files by year and land type"""
        print("Scanning and organizing TIF files...")

        # Find all TIF files
        tif_files = glob.glob(os.path.join(self.data_dir, "*.tif"))
        print(f"Found {len(tif_files)} TIF files")

        # Organize files
        file_map = defaultdict(dict)

        for tif_file in tif_files:
            filename = os.path.basename(tif_file)
            print(f"Processing: {filename}")

            # Extract land use type and year
            found_match = False
            for land_type in self.land_use_types.keys():
                if filename.startswith(land_type + '-'):
                    # Extract year
                    import re
                    year_match = re.search(r'-(\d{4})\.tif$', filename)
                    if year_match:
                        year = int(year_match.group(1))
                        if year in self.target_years:
                            file_map[year][land_type] = tif_file
                            print(f"  -> Mapped to year {year}, type {land_type}")
                            found_match = True
                    break

            if not found_match:
                print(f"  -> Could not parse filename: {filename}")

        print(f"\nFile mapping summary:")
        for year in sorted(file_map.keys()):
            print(f"  Year {year}: {list(file_map[year].keys())}")

        return file_map

    def load_landuse_data(self):
        """Load land use data with detailed debugging"""
        print(f"Loading land use data for {self.analysis_period}...")

        # Scan and organize files
        file_map = self.scan_and_organize_files()

        if not file_map:
            print("❌ No valid files found!")
            return False

        # Load spatial data for each year
        for year in sorted(file_map.keys()):
            print(f"\n--- Loading data for year {year} ---")
            self.spatial_data[year] = {}

            year_files = file_map[year]

            for land_type in self.land_use_types.keys():
                if land_type in year_files:
                    file_path = year_files[land_type]
                    print(f"Loading {land_type} from: {os.path.basename(file_path)}")

                    try:
                        with rasterio.open(file_path) as src:
                            # Get spatial coordinates on first successful file
                            if self.longitude is None or self.latitude is None:
                                print("Extracting spatial coordinates...")

                                # Get bounds and transform
                                bounds = src.bounds
                                transform = src.transform
                                width, height = src.width, src.height

                                # Create coordinate arrays
                                x_coords = [bounds.left + (i + 0.5) * transform[0] for i in range(width)]
                                y_coords = [bounds.top + (j + 0.5) * transform[4] for j in range(height)]

                                self.longitude = np.array(x_coords)
                                self.latitude = np.array(y_coords)

                                print(f"  Grid size: {width} × {height}")
                                print(f"  Longitude: {self.longitude.min():.2f}° to {self.longitude.max():.2f}°")
                                print(f"  Latitude: {self.latitude.min():.2f}° to {self.latitude.max():.2f}°")

                            # Read data
                            data = src.read(1)
                            print(f"  Data shape: {data.shape}")
                            print(f"  Data type: {data.dtype}")
                            print(f"  Data range: {data.min():.6f} to {data.max():.6f}")
                            print(f"  Non-zero pixels: {np.sum(data > 0)}")
                            print(f"  Total pixels: {data.size}")

                            # Handle NoData values
                            if src.nodata is not None:
                                print(f"  NoData value: {src.nodata}")
                                data[data == src.nodata] = 0.0

                            # Clip values to valid range
                            data = np.clip(data, 0.0, 1.0)

                            # Store data
                            self.spatial_data[year][land_type] = data
                            print(f"  ✅ Successfully loaded {land_type} for {year}")

                    except Exception as e:
                        print(f"  ❌ Error loading {land_type} for {year}: {e}")
                        print(f"     File: {file_path}")
                else:
                    print(f"  ⚠️  Missing {land_type} data for {year}")

        loaded_years = len(self.spatial_data)
        print(f"\n📊 Successfully loaded data for {loaded_years} years")

        return loaded_years > 0

    def calculate_national_statistics(self):
        """Calculate L_tk, r_tk, and σ_tk with detailed logging"""
        print(f"\nCalculating national statistics...")

        # Initialize result containers
        self.L_tk = {land_type: [] for land_type in self.land_use_types.keys()}
        self.sigma_tk = {land_type: [] for land_type in self.land_use_types.keys()}
        self.valid_years = sorted(self.spatial_data.keys())

        print(f"Processing {len(self.valid_years)} years: {self.valid_years}")

        # Calculate for each year
        for year in self.valid_years:
            print(f"\n--- Processing year {year} ---")

            for land_type in self.land_use_types.keys():
                if land_type in self.spatial_data[year]:
                    data = self.spatial_data[year][land_type]

                    # Method 1: Calculate statistics using all grid cells
                    # L_tk: National average proportion
                    total_land_area = np.sum(data >= 0)  # All valid grid cells
                    if total_land_area > 0:
                        # Average proportion across all grid cells
                        L_tk_value = np.mean(data)

                        # Spatial standard deviation
                        sigma_tk_value = np.std(data)

                        print(f"  {land_type}: L_tk={L_tk_value:.6f}, σ_tk={sigma_tk_value:.6f}")
                        print(f"    Valid cells: {total_land_area}, Non-zero cells: {np.sum(data > 0)}")
                        print(f"    Data range: {data.min():.6f} - {data.max():.6f}")

                        self.L_tk[land_type].append(L_tk_value)
                        self.sigma_tk[land_type].append(sigma_tk_value)
                    else:
                        print(f"  {land_type}: No valid data")
                        self.L_tk[land_type].append(0.0)
                        self.sigma_tk[land_type].append(0.0)
                else:
                    print(f"  {land_type}: Missing data for {year}")
                    self.L_tk[land_type].append(np.nan)
                    self.sigma_tk[land_type].append(np.nan)

        # Calculate annual change rates r_tk
        print(f"\nCalculating annual change rates...")
        self.r_tk = {land_type: [] for land_type in self.land_use_types.keys()}

        for land_type in self.land_use_types.keys():
            L_values = self.L_tk[land_type]
            print(f"\n{land_type} change rates:")

            for i in range(len(L_values) - 1):
                if (not np.isnan(L_values[i]) and not np.isnan(L_values[i + 1]) and L_values[i] > 0):
                    # r_tk = (L_{t+1} - L_t) / L_t * 100
                    change_rate = (L_values[i + 1] - L_values[i]) / L_values[i] * 100
                    self.r_tk[land_type].append(change_rate)
                    print(f"  {self.valid_years[i]} -> {self.valid_years[i + 1]}: {change_rate:.2f}%")
                else:
                    self.r_tk[land_type].append(np.nan)
                    print(f"  {self.valid_years[i]} -> {self.valid_years[i + 1]}: NaN (invalid data)")

        # Print summary statistics
        print(f"\n{'=' * 60}")
        print(f"SUMMARY STATISTICS FOR {self.analysis_period}")
        print(f"{'=' * 60}")

        for land_type, description in self.land_use_types.items():
            L_values = [x for x in self.L_tk[land_type] if not np.isnan(x)]
            if L_values:
                print(f"\n{description}:")
                print(f"  Average proportion: {np.mean(L_values):.6f} ± {np.std(L_values):.6f}")
                print(f"  Range: {np.min(L_values):.6f} - {np.max(L_values):.6f}")
                print(f"  Valid years: {len(L_values)}")
            else:
                print(f"\n{description}: No valid data")

        return True

    def create_visualization_plots(self):
        """Create comprehensive visualization plots"""
        print(f"\nGenerating visualization plots...")

        # 1. Time series of land use proportions
        plt.figure(figsize=(14, 8))

        for land_type, description in self.land_use_types.items():
            values = self.L_tk[land_type]
            plt.plot(self.valid_years, values, marker='o', linewidth=2.5, markersize=6,
                     label=description, color=self.land_use_colors[land_type], alpha=0.8)

        plt.title(f'National Average Land Use Proportions ({self.analysis_period})',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Proportion', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir,
                                   f'Fig1_Land_Use_Proportions_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # 2. Annual change rates
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        change_years = self.valid_years[1:]  # Change rates have one less year

        for i, (land_type, description) in enumerate(self.land_use_types.items()):
            ax = axes[i]
            change_rates = self.r_tk[land_type]

            # Filter out NaN values for plotting
            valid_indices = [j for j, val in enumerate(change_rates) if not np.isnan(val)]
            if valid_indices:
                valid_years = [change_years[j] for j in valid_indices]
                valid_rates = [change_rates[j] for j in valid_indices]

                colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in valid_rates]
                bars = ax.bar(valid_years, valid_rates, color=colors, alpha=0.7, width=0.8)

                # Add statistics
                mean_rate = np.mean(valid_rates)
                ax.text(0.02, 0.98, f'Mean: {mean_rate:.2f}%', transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax.set_title(f'{description} Annual Change Rate', fontsize=11, fontweight='bold')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Change Rate (%)', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Remove empty subplot
        fig.delaxes(axes[5])

        plt.suptitle(f'Land Use Annual Change Rates ({self.analysis_period})', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir,
                                   f'Fig2_Land_Use_Change_Rates_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # 3. Spatial variability
        plt.figure(figsize=(14, 8))

        for land_type, description in self.land_use_types.items():
            values = self.sigma_tk[land_type]
            plt.plot(self.valid_years, values, marker='s', linewidth=2.5, markersize=6,
                     label=description, color=self.land_use_colors[land_type], alpha=0.8)

        plt.title(f'Spatial Variability of Land Use Types ({self.analysis_period})',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Spatial Standard Deviation', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir,
                                   f'Fig3_Land_Use_Spatial_Variability_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # 4. Stacked area chart
        plt.figure(figsize=(14, 8))

        # Prepare data for stacked plot
        land_use_matrix = []
        for land_type in self.land_use_types.keys():
            land_use_matrix.append(self.L_tk[land_type])

        land_use_matrix = np.array(land_use_matrix)

        # Create stacked area plot
        plt.stackplot(self.valid_years, *land_use_matrix,
                      labels=list(self.land_use_types.values()),
                      colors=[self.land_use_colors[lt] for lt in self.land_use_types.keys()],
                      alpha=0.8)

        plt.title(f'Land Use Composition Over Time ({self.analysis_period})',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Cumulative Proportion', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir,
                                   f'Fig4_Land_Use_Composition_{self.analysis_period.replace("-", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print("All visualization plots generated successfully!")
        return True

    def export_results(self):
        """Export results to CSV and XLSX with validation"""
        print(f"\nExporting analysis results...")

        # Validate data before export
        print("Validating data...")
        for land_type in self.land_use_types.keys():
            valid_count = sum(1 for x in self.L_tk[land_type] if not np.isnan(x))
            print(f"  {land_type}: {valid_count}/{len(self.L_tk[land_type])} valid values")

        # Create results DataFrame
        results_data = {'Year': self.valid_years}

        # Add L_tk values (average proportions)
        for land_type, description in self.land_use_types.items():
            results_data[f'{description}_Average_Proportion'] = self.L_tk[land_type]

        # Add σ_tk values (spatial standard deviations)
        for land_type, description in self.land_use_types.items():
            results_data[f'{description}_Spatial_StdDev'] = self.sigma_tk[land_type]

        results_df = pd.DataFrame(results_data)

        # Create change rates DataFrame
        change_data = {'Year': self.valid_years[1:]}  # One less year for change rates

        for land_type, description in self.land_use_types.items():
            change_data[f'{description}_Change_Rate_Percent'] = self.r_tk[land_type]

        change_df = pd.DataFrame(change_data)

        # Export to CSV
        csv_path = os.path.join(self.output_dir,
                                f'landuse_analysis_results_{self.analysis_period.replace("-", "_")}.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Main results exported to: {csv_path}")

        csv_change_path = os.path.join(self.output_dir,
                                       f'landuse_change_rates_{self.analysis_period.replace("-", "_")}.csv')
        change_df.to_csv(csv_change_path, index=False, encoding='utf-8-sig')
        print(f"Change rates exported to: {csv_change_path}")

        # Export to XLSX
        xlsx_path = os.path.join(self.output_dir,
                                 f'landuse_analysis_results_{self.analysis_period.replace("-", "_")}.xlsx')

        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Main_Results', index=False)
            change_df.to_excel(writer, sheet_name='Change_Rates', index=False)

            # Summary statistics
            summary_stats = []
            for land_type, description in self.land_use_types.items():
                L_values = [x for x in self.L_tk[land_type] if not np.isnan(x)]

                if L_values:
                    # Calculate linear trend
                    years_array = np.array(self.valid_years)
                    L_array = np.array(self.L_tk[land_type])
                    valid_mask = ~np.isnan(L_array)

                    if np.sum(valid_mask) > 2:
                        trend_slope = np.polyfit(years_array[valid_mask], L_array[valid_mask], 1)[0]
                    else:
                        trend_slope = 0

                    summary_stats.append({
                        'Land_Use_Type': description,
                        'Mean_Proportion': f"{np.mean(L_values):.6f}",
                        'Std_Proportion': f"{np.std(L_values):.6f}",
                        'Min_Proportion': f"{np.min(L_values):.6f}",
                        'Max_Proportion': f"{np.max(L_values):.6f}",
                        'Linear_Trend_per_year': f"{trend_slope:.8f}",
                        'Valid_Years': len(L_values)
                    })

            pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Statistics', index=False)

        print(f"XLSX file exported to: {xlsx_path}")

        # Display first few rows to verify
        print(f"\nFirst 5 rows of results:")
        print(results_df.head().to_string(index=False, float_format='%.6f'))

        return results_df, change_df

    def run_complete_analysis(self):
        """Run complete analysis workflow with error handling"""
        print("=" * 80)
        print("CHINA LAND USE AND COVER CHANGE ANALYSIS (FIXED VERSION)")
        print(f"ANALYSIS PERIOD: {self.analysis_period}")
        print("=" * 80)

        try:
            # 1. Load data
            if not self.load_landuse_data():
                print("❌ Failed to load data")
                return False

            # 2. Calculate statistics
            if not self.calculate_national_statistics():
                print("❌ Failed to calculate statistics")
                return False

            # 3. Create plots
            if not self.create_visualization_plots():
                print("❌ Failed to create plots")
                return False

            # 4. Export results
            results_df, change_df = self.export_results()

            print(f"\n{'=' * 80}")
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"{'=' * 80}")
            print(f"\nGenerated files in '{self.output_dir}':")
            print("  📊 DATA FILES:")
            print(f"    - landuse_analysis_results_{self.analysis_period.replace('-', '_')}.csv")
            print(f"    - landuse_change_rates_{self.analysis_period.replace('-', '_')}.csv")
            print(f"    - landuse_analysis_results_{self.analysis_period.replace('-', '_')}.xlsx")
            print("  📈 FIGURE FILES:")
            print(f"    - Fig1_Land_Use_Proportions_{self.analysis_period.replace('-', '_')}.png")
            print(f"    - Fig2_Land_Use_Change_Rates_{self.analysis_period.replace('-', '_')}.png")
            print(f"    - Fig3_Land_Use_Spatial_Variability_{self.analysis_period.replace('-', '_')}.png")
            print(f"    - Fig4_Land_Use_Composition_{self.analysis_period.replace('-', '_')}.png")

            return results_df, change_df

        except Exception as e:
            print(f"❌ Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    data_dir = r"C:\Users\ZEC\Desktop\raw_data"
    output_dir = r"D:\Pythonpro\2024_D\result\Q1"

    print("Starting FIXED land use analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Create analyzer instance
    analyzer = LandUseAnalyzerFixed(data_dir, output_dir)

    # Run analysis
    results = analyzer.run_complete_analysis()

    if results:
        print(f"\n✅ Fixed analysis completed successfully!")
        print(f"✅ Data should now be properly populated in CSV files")
    else:
        print(f"\n❌ Analysis still failed - please check error messages above")


if __name__ == "__main__":
    main()