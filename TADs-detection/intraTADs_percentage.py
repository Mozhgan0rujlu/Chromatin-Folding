#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created for Hi-C intra-TAD analysis 

Hi-C intra-TAD analysis with diagonal removal + bootstrap.
INPUT: *_filtered.txt (intra-TAD), *.txt (all contacts), barcode_ celltype.tsv
OUTPUT: Violin plots + bootstrap distributions (PDF/PNG/SVG) and summary CSV.
CALC: Removes diagonal contacts, computes %intra-TAD per cell, bootstraps (10k iter, n=150) to get mean, SE, 95% CI per cell type.

Author: Mozhgan Orujlu
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,  # Editable text in PDF
    'ps.fonttype': 42
})

# Custom color palette
COLORS = {
    'OPC': '#1f77b4',  # Blue
    'OGC': '#ff7f0e',  # Orange
    'ASC': '#2ca02c',  # Green
    'bootstrap': '#d62728',  # Red
    'original': '#9467bd',   # Purple
    'CI': '#7f7f7f'    # Gray
}

class DiagonalRemovedTADAnalyzer:
    """
    Analyzer for intra-TAD contacts with diagonal contacts removed
    Data format: bin1, bin2, contact_count (tab-separated)
    """
    
    def __init__(self, intra_tads_folder, all_contacts_folder, barcode_file, target_celltypes):
        self.intra_tads_folder = intra_tads_folder
        self.all_contacts_folder = all_contacts_folder
        self.barcode_file = barcode_file
        self.target_celltypes = target_celltypes
        self.cell_data = {}
        self.bootstrap_results = {}
        
    def load_barcode_mapping(self):
        """Load barcode to celltype mapping with diagonal removal filtering"""
        print("=" * 80)
        print("STEP 1: Loading barcode mapping")
        print("=" * 80)
        
        self.barcode_df = pd.read_csv(self.barcode_file, sep='\t')
        original_count = len(self.barcode_df)
        
        # Filter for target cell types
        self.barcode_df = self.barcode_df[
            self.barcode_df['celltype'].isin(self.target_celltypes)
        ]
        
        print(f"  • Total barcodes: {original_count}")
        print(f"  • Target cell types: {', '.join(self.target_celltypes)}")
        print(f"  • Filtered barcodes: {len(self.barcode_df)}")
        
        # Check file availability
        print("\n  Checking file availability...")
        available_cells = []
        
        for barcode in tqdm(self.barcode_df['barcode'], desc="  Progress"):
            intra_file = os.path.join(self.intra_tads_folder, f"{barcode}_filtered.txt")
            all_file = os.path.join(self.all_contacts_folder, f"{barcode}.txt")
            
            if os.path.exists(intra_file) and os.path.exists(all_file):
                available_cells.append(barcode)
        
        self.barcode_df = self.barcode_df[
            self.barcode_df['barcode'].isin(available_cells)
        ]
        
        print(f"\n  • Cells with complete data: {len(self.barcode_df)}")
        
        # Show counts per cell type
        celltype_counts = self.barcode_df['celltype'].value_counts()
        for celltype in self.target_celltypes:
            count = celltype_counts.get(celltype, 0)
            print(f"    - {celltype}: {count} cells")
    
    def calculate_intra_tad_percentage_no_diag(self, barcode):
        """
        Calculate intra-TAD percentage with diagonal contacts removed
        Diagonal contacts: bin1 == bin2
        """
        intra_file = os.path.join(self.intra_tads_folder, f"{barcode}_filtered.txt")
        all_file = os.path.join(self.all_contacts_folder, f"{barcode}.txt")
        
        try:
            # Read files and filter out diagonal contacts
            # Intra-TAD contacts (already filtered by TADs, but still may have diagonal)
            intra_df = pd.read_csv(intra_file, sep='\t', header=None, 
                                  names=['bin1', 'bin2', 'contacts'])
            
            # All contacts
            all_df = pd.read_csv(all_file, sep='\t', header=None,
                                names=['bin1', 'bin2', 'contacts'])
            
            # Remove diagonal contacts (bin1 == bin2)
            intra_no_diag = intra_df[intra_df['bin1'] != intra_df['bin2']]
            all_no_diag = all_df[all_df['bin1'] != all_df['bin2']]
            
            # Calculate total contacts
            total_intra = intra_no_diag['contacts'].sum()
            total_all = all_no_diag['contacts'].sum()
            
            if total_all == 0:
                return barcode, None
            
            # Calculate percentage
            percentage = (total_intra / total_all) * 100
            
            # Store diagonal fraction for quality control
            diag_intra = len(intra_df) - len(intra_no_diag)
            diag_all = len(all_df) - len(all_no_diag)
            
            return barcode, {
                'percentage': percentage,
                'total_intra': total_intra,
                'total_all': total_all,
                'diag_intra': diag_intra,
                'diag_all': diag_all
            }
            
        except Exception as e:
            print(f"    Warning: Error processing {barcode}: {e}")
            return barcode, None
    
    def collect_cell_data(self):
        """Collect intra-TAD percentages for all target cells"""
        print("\n" + "=" * 80)
        print("STEP 2: Calculating intra-TAD percentages (diagonal removed)")
        print("=" * 80)
        
        celltype_groups = self.barcode_df.groupby('celltype')['barcode'].apply(list)
        
        for celltype in self.target_celltypes:
            if celltype not in celltype_groups:
                print(f"\n  ⚠ No cells found for {celltype}")
                continue
            
            barcodes = celltype_groups[celltype]
            percentages = []
            valid_barcodes = []
            qc_stats = {'diag_intra': [], 'diag_all': []}
            
            print(f"\n  Processing {celltype} ({len(barcodes)} cells)...")
            
            for barcode in tqdm(barcodes, desc=f"  {celltype}", leave=True):
                result = self.calculate_intra_tad_percentage_no_diag(barcode)
                
                if result[1] is not None:
                    percentages.append(result[1]['percentage'])
                    valid_barcodes.append(barcode)
                    qc_stats['diag_intra'].append(result[1]['diag_intra'])
                    qc_stats['diag_all'].append(result[1]['diag_all'])
            
            if percentages:
                self.cell_data[celltype] = {
                    'percentages': np.array(percentages),
                    'barcodes': valid_barcodes,
                    'n_cells': len(percentages),
                    'qc_stats': qc_stats
                }
                
                # Print summary statistics
                mean_val = np.mean(percentages)
                std_val = np.std(percentages)
                sem_val = std_val / np.sqrt(len(percentages))
                ci_95 = 1.96 * sem_val
                
                print(f"\n    ✓ {len(percentages)} cells processed")
                print(f"      Mean intra-TAD: {mean_val:.3f}%")
                print(f"      SD: {std_val:.3f}%")
                print(f"      SEM: {sem_val:.3f}%")
                print(f"      95% CI: [{mean_val - ci_95:.3f}%, {mean_val + ci_95:.3f}%]")
                
                # Diagonal removal summary
                if qc_stats['diag_intra']:
                    avg_diag_intra = np.mean(qc_stats['diag_intra'])
                    avg_diag_all = np.mean(qc_stats['diag_all'])
                    print(f"      Avg diagonal contacts removed: "
                          f"{avg_diag_intra:.1f} (intra), {avg_diag_all:.1f} (all)")
            else:
                print(f"\n    ✗ No valid data for {celltype}")
        
        gc.collect()
    
    def bootstrap_analysis(self, data, n_iterations=10000, 
                          bootstrap_sample_size=None, ci_level=0.95):
        """
        Perform bootstrap resampling with optional fixed sample size
        """
        n_original = len(data)
        
        # Set bootstrap sample size
        if bootstrap_sample_size is None:
            sample_size = n_original
        else:
            sample_size = min(bootstrap_sample_size, n_original)
        
        # Generate bootstrap samples (vectorized)
        bootstrap_indices = np.random.randint(
            0, n_original, size=(n_iterations, sample_size)
        )
        
        # Calculate bootstrap means
        bootstrap_means = data[bootstrap_indices].mean(axis=1)
        
        # Calculate statistics
        mean_estimate = np.mean(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, (1 - ci_level) * 50)
        ci_upper = np.percentile(bootstrap_means, 100 - (1 - ci_level) * 50)
        std_error = np.std(bootstrap_means)
        
        return {
            'mean': mean_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': std_error,
            'bootstrap_means': bootstrap_means,
            'original_mean': np.mean(data),
            'original_std': np.std(data),
            'n_cells': n_original,
            'bootstrap_sample_size': sample_size,
            'n_iterations': n_iterations,
            'ci_level': ci_level
        }
    
    def run_bootstrap_analysis(self, n_iterations=10000, 
                              bootstrap_sample_size=150):
        """
        Run bootstrap analysis for all cell types
        """
        print("\n" + "=" * 80)
        print("STEP 3: Bootstrap analysis")
        print("=" * 80)
        
        for celltype in self.target_celltypes:
            if celltype not in self.cell_data:
                continue
            
            print(f"\n  {celltype}:")
            data = self.cell_data[celltype]['percentages']
            
            bootstrap_stats = self.bootstrap_analysis(
                data, 
                n_iterations=n_iterations,
                bootstrap_sample_size=bootstrap_sample_size
            )
            
            self.bootstrap_results[celltype] = bootstrap_stats
            
            # Print results
            print(f"    Original mean: {bootstrap_stats['original_mean']:.3f}%")
            print(f"    Bootstrap mean: {bootstrap_stats['mean']:.3f}%")
            print(f"    95% CI: [{bootstrap_stats['ci_lower']:.3f}%, "
                  f"{bootstrap_stats['ci_upper']:.3f}%]")
            print(f"    Bootstrap SE: {bootstrap_stats['std_error']:.4f}")
    
    def create_publication_figure(self, output_dir='publication_figures'):
        """
        Create publication-ready figure with violin plots and bootstrap distributions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("STEP 4: Creating publication figure")
        print("=" * 80)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Grid specification for subplots
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], 
                              height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        
        # Create three main violin plot axes (top row)
        ax_violin_opc = fig.add_subplot(gs[0, 0])
        ax_violin_ogc = fig.add_subplot(gs[0, 1])
        ax_violin_asc = fig.add_subplot(gs[0, 2])
        
        # Create three bootstrap distribution axes (bottom row)
        ax_boot_opc = fig.add_subplot(gs[1, 0])
        ax_boot_ogc = fig.add_subplot(gs[1, 1])
        ax_boot_asc = fig.add_subplot(gs[1, 2])
        
        violin_axes = [ax_violin_opc, ax_violin_ogc, ax_violin_asc]
        boot_axes = [ax_boot_opc, ax_boot_ogc, ax_boot_asc]
        
        # Calculate global y-axis limits
        all_percentages = []
        for ct in self.target_celltypes:
            if ct in self.cell_data:
                all_percentages.extend(self.cell_data[ct]['percentages'])
        y_min = max(0, np.percentile(all_percentages, 1) - 5)
        y_max = np.percentile(all_percentages, 99) + 5
        
        # Plot for each cell type
        for idx, celltype in enumerate(self.target_celltypes):
            if celltype not in self.cell_data:
                continue
            
            # Get data
            percentages = self.cell_data[celltype]['percentages']
            bootstrap_stats = self.bootstrap_results[celltype]
            bootstrap_means = bootstrap_stats['bootstrap_means']
            
            # ============= VIOLIN PLOT =============
            ax = violin_axes[idx]
            
            # Create violin plot
            violin_parts = ax.violinplot(
                percentages, 
                positions=[0],
                widths=0.7,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            
            # Style the violin
            for pc in violin_parts['bodies']:
                pc.set_facecolor(COLORS[celltype])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
                pc.set_zorder(1)
            
            # Add individual points (jittered)
            jitter = np.random.normal(0, 0.05, size=len(percentages))
            jitter = np.clip(jitter, -0.3, 0.3)
            
            ax.scatter(
                jitter, percentages,
                s=20, alpha=0.4, color='black', 
                edgecolor='none', zorder=2
            )
            
            # Add box plot inside violin
            box_props = {
                'boxprops': {'color': 'black', 'linewidth': 1.2},
                'medianprops': {'color': 'black', 'linewidth': 2},
                'whiskerprops': {'color': 'black', 'linewidth': 1},
                'capprops': {'color': 'black', 'linewidth': 1},
                'flierprops': {'marker': 'o', 'markerfacecolor': 'black',
                              'markersize': 3, 'alpha': 0.5, 'linestyle': 'none'}
            }
            
            ax.boxplot(
                percentages, 
                positions=[0],
                widths=0.2,
                showfliers=False,
                **box_props
            )
            
            # Add bootstrap mean line (RED)
            ax.axhline(
                y=bootstrap_stats['mean'], 
                color=COLORS['bootstrap'], 
                linestyle='-', 
                linewidth=2.5,
                label=f'Bootstrap mean: {bootstrap_stats["mean"]:.2f}%',
                zorder=5
            )
            
            # Add original mean line (PURPLE)
            ax.axhline(
                y=bootstrap_stats['original_mean'], 
                color=COLORS['original'], 
                linestyle='--', 
                linewidth=2,
                label=f'Original mean: {bootstrap_stats["original_mean"]:.2f}%',
                zorder=5
            )
            
            # Add confidence interval shading
            ax.axhspan(
                bootstrap_stats['ci_lower'], 
                bootstrap_stats['ci_upper'],
                alpha=0.15, 
                color=COLORS['CI'],
                label=f'95% CI: [{bootstrap_stats["ci_lower"]:.2f}%, '
                      f'{bootstrap_stats["ci_upper"]:.2f}%]',
                zorder=3
            )
            
            # Format violin plot
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([0])
            ax.set_xticklabels([celltype], fontsize=13, fontweight='bold')
            
            if idx == 0:
                ax.set_ylabel('Intra-TAD Contacts (%)', fontsize=13, fontweight='bold')
            
            ax.set_ylim(y_min, y_max)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='--', axis='y')
            ax.set_axisbelow(True)
            
            # Add sample size
            ax.text(
                0, y_min + 1,
                f'n = {len(percentages)} cells',
                ha='center', va='bottom', 
                fontsize=11, fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.8, edgecolor='none')
            )
            
            # ===== ADD TEXT ANNOTATIONS WITH VALUES TO EACH PLOT =====
            # Add bootstrap mean text annotation
            #ax.text(
             #   0.6, bootstrap_stats['mean'],
              #  f'  Bootstrap: {bootstrap_stats["mean"]:.2f}%',
               # ha='left', va='center',
                #fontsize=9, color=COLORS['bootstrap'],
                #ontweight='bold',
                #bbox=dict(boxstyle='round', facecolor='white', 
                 #        alpha=0.9, edgecolor=COLORS['bootstrap'], linewidth=0.8)
            #)
            
            # Add original mean text annotation
           # ax.text(
            #    0.6, bootstrap_stats['original_mean'],
             #   f'  Original: {bootstrap_stats["original_mean"]:.2f}%',
              #  ha='left', va='center',
               # fontsize=9, color=COLORS['original'],
                #fontweight='bold',
                #bbox=dict(boxstyle='round', facecolor='white', 
                 #        alpha=0.9, edgecolor=COLORS['original'], linewidth=0.8)
            #)
            
            # Add CI text annotation at the bottom of the plot
            ax.text(
                0.6, y_min + 2,
                f'95% CI: [{bootstrap_stats["ci_lower"]:.2f}%, {bootstrap_stats["ci_upper"]:.2f}%]',
                ha='left', va='bottom',
                fontsize=9, color=COLORS['CI'],
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.9, edgecolor=COLORS['CI'], linewidth=0.8)
            )
            
            # Add legend (optional - you can comment this out if you prefer the text annotations)
            ax.legend(loc='upper right', framealpha=0.95, 
                     edgecolor='black', fontsize=8)
            
            # ============= BOOTSTRAP DISTRIBUTION =============
            ax_boot = boot_axes[idx]
            
            # Plot histogram of bootstrap means
            n, bins, patches = ax_boot.hist(
                bootstrap_means, 
                bins=40, 
                density=True,
                alpha=0.7,
                color=COLORS[celltype],
                edgecolor='black',
                linewidth=0.5,
                zorder=1
            )
            
            # Add KDE
            kde = gaussian_kde(bootstrap_means)
            x_range = np.linspace(
                min(bootstrap_means) * 0.995, 
                max(bootstrap_means) * 1.005, 
                200
            )
            kde_values = kde(x_range)
            
            ax_boot.plot(
                x_range, kde_values,
                color='black', 
                linewidth=2,
                linestyle='-',
                label='Density',
                zorder=2
            )
            
            # Add bootstrap mean line
            ax_boot.axvline(
                x=bootstrap_stats['mean'],
                color=COLORS['bootstrap'],
                linestyle='-',
                linewidth=2.5,
                label=f'Bootstrap mean: {bootstrap_stats["mean"]:.2f}%',
                zorder=3
            )
            
            # Add confidence interval shading
            ax_boot.axvspan(
                bootstrap_stats['ci_lower'],
                bootstrap_stats['ci_upper'],
                alpha=0.2,
                color=COLORS['CI'],
                label=f'95% CI',
                zorder=1
            )
            
            # Format bootstrap plot
            ax_boot.set_xlabel('Intra-TAD %', fontsize=11, fontweight='bold')
            
            if idx == 0:
                ax_boot.set_ylabel('Density', fontsize=11, fontweight='bold')
            
            # Add statistics text
            stats_text = (
                f'Mean: {bootstrap_stats["mean"]:.2f}%\n'
                f'SE: {bootstrap_stats["std_error"]:.3f}\n'
                f'95% CI: [{bootstrap_stats["ci_lower"]:.2f}, '
                f'{bootstrap_stats["ci_upper"]:.2f}]'
            )
            
            ax_boot.text(
                0.98, 0.95, stats_text,
                transform=ax_boot.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.9, edgecolor='black', linewidth=0.5)
            )
            
            ax_boot.grid(True, alpha=0.2, linestyle='--')
            ax_boot.set_axisbelow(True)
        
        # Add statistical comparison annotations
        self._add_significance_annotations(violin_axes)
        
        # Add figure title and overall labels
        fig.suptitle(
            'Intra-TAD Contact Distribution (Diagonal Contacts Removed)',
            fontsize=16, fontweight='bold', y=1.02
        )
        
        # Add subtitle
        fig.text(
            0.5, 0.98,
            'Violin plots show distribution; Red = Bootstrap mean, Purple = Original mean',
            ha='center', va='top', fontsize=12, fontstyle='italic'
        )
        
        # Add diagonal removal note
        fig.text(
            0.5, 0.01,
            f'Diagonal contacts (bin1 = bin2) were removed before calculation.\n'
            f'Bootstrap: {self.bootstrap_results[self.target_celltypes[0]]["n_iterations"]:,} iterations, '
            f'sample size = {self.bootstrap_results[self.target_celltypes[0]]["bootstrap_sample_size"]} cells',
            ha='center', va='bottom', fontsize=10, fontstyle='italic'
        )
        
        # Save figure in multiple formats
        plt.tight_layout()
        
        # PDF for publication
        plt.savefig(
            os.path.join(output_dir, 'Figure1_IntraTAD_Violin_Bootstrap.pdf'),
            dpi=300, bbox_inches='tight', format='pdf'
        )
        
        # PNG for quick viewing
        plt.savefig(
            os.path.join(output_dir, 'Figure1_IntraTAD_Violin_Bootstrap.png'),
            dpi=300, bbox_inches='tight', format='png'
        )
        
        # SVG for editable graphics
        plt.savefig(
            os.path.join(output_dir, 'Figure1_IntraTAD_Violin_Bootstrap.svg'),
            bbox_inches='tight', format='svg'
        )
        
        plt.close()
        
        print(f"\n  ✓ Figure saved to: {output_dir}/")
        print(f"    - Figure1_IntraTAD_Violin_Bootstrap.pdf")
        print(f"    - Figure1_IntraTAD_Violin_Bootstrap.png")
        print(f"    - Figure1_IntraTAD_Violin_Bootstrap.svg")
    
    def _add_significance_annotations(self, axes):
        """Add statistical significance annotations between cell types"""
        if len(self.bootstrap_results) < 2:
            return
        
        # Perform pairwise Mann-Whitney U tests
        celltypes = self.target_celltypes
        p_values = {}
        
        for i in range(len(celltypes)):
            for j in range(i + 1, len(celltypes)):
                ct1, ct2 = celltypes[i], celltypes[j]
                
                if ct1 in self.cell_data and ct2 in self.cell_data:
                    data1 = self.cell_data[ct1]['percentages']
                    data2 = self.cell_data[ct2]['percentages']
                    
                    stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    p_values[f"{ct1}_vs_{ct2}"] = p_val
        
        # Add annotations to the console
        print("\n  Statistical significance (Mann-Whitney U test):")
        for comparison, p_val in p_values.items():
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            print(f"    {comparison}: p = {p_val:.6f} {sig}")
    
    def save_results(self, output_dir='publication_figures'):
        """Save all numerical results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary statistics
        summary_data = []
        
        for celltype in self.target_celltypes:
            if celltype in self.bootstrap_results:
                stats_data = self.bootstrap_results[celltype]
                raw_data = self.cell_data[celltype]['percentages']
                
                summary_data.append({
                    'Cell_Type': celltype,
                    'N_Cells': stats_data['n_cells'],
                    'Original_Mean': stats_data['original_mean'],
                    'Original_SD': stats_data['original_std'],
                    'Original_SEM': stats_data['original_std'] / np.sqrt(stats_data['n_cells']),
                    'Bootstrap_Mean': stats_data['mean'],
                    'Bootstrap_SE': stats_data['std_error'],
                    'CI_Lower_95': stats_data['ci_lower'],
                    'CI_Upper_95': stats_data['ci_upper'],
                    'Median': np.median(raw_data),
                    'Q1': np.percentile(raw_data, 25),
                    'Q3': np.percentile(raw_data, 75),
                    'Min': np.min(raw_data),
                    'Max': np.max(raw_data)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(output_dir, 'Table1_IntraTAD_Statistics.csv'),
            index=False, float_format='%.4f'
        )
        
        # Save raw data for reproducibility
        raw_data_list = []
        for celltype in self.target_celltypes:
            if celltype in self.cell_data:
                for barcode, pct in zip(
                    self.cell_data[celltype]['barcodes'],
                    self.cell_data[celltype]['percentages']
                ):
                    raw_data_list.append({
                        'Cell_Type': celltype,
                        'Barcode': barcode,
                        'Intra_TAD_Percentage': pct,
                        'Diagonal_Removed': True
                    })
        
        raw_df = pd.DataFrame(raw_data_list)
        raw_df.to_csv(
            os.path.join(output_dir, 'Supplementary_IntraTAD_RawData.csv'),
            index=False, float_format='%.4f'
        )
        
        print(f"\n  ✓ Results saved to: {output_dir}/")
        print(f"    - Table1_IntraTAD_Statistics.csv")
        print(f"    - Supplementary_IntraTAD_RawData.csv")
    
    def run_complete_analysis(self, n_iterations=10000, 
                             bootstrap_sample_size=150,
                             output_dir='publication_figures'):
        """
        Run complete analysis pipeline
        """
        print("\n" + "=" * 80)
        print("INTRA-TAD ANALYSIS WITH DIAGONAL REMOVAL")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  • Target cell types: {', '.join(self.target_celltypes)}")
        print(f"  • Bootstrap iterations: {n_iterations:,}")
        print(f"  • Bootstrap sample size: {bootstrap_sample_size}")
        print(f"  • Diagonal contacts: REMOVED")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_barcode_mapping()
        
        # Step 2: Calculate percentages with diagonal removal
        self.collect_cell_data()
        
        # Step 3: Run bootstrap
        self.run_bootstrap_analysis(
            n_iterations=n_iterations,
            bootstrap_sample_size=bootstrap_sample_size
        )
        
        # Step 4: Create publication figure
        self.create_publication_figure(output_dir)
        
        # Step 5: Save results
        self.save_results(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 80)
        
        return self.bootstrap_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ==================== CONFIGURATION ====================
    
    # File paths (UPDATE THESE)
    INTRA_TADS_FOLDER = " " #repository of the txt files of intra-TADs contacts 
    ALL_CONTACTS_FOLDER = " " # repository of txt contact files
    BARCODE_FILE = " " # here you should input the repository of a tsv file which has columns of: barcodes and their corresponding cell type
    OUTPUT_DIR = ""
    
    # Analysis parameters
    TARGET_CELLTYPES = ['OPC', 'OGC', 'ASC']
    N_ITERATIONS = 10000
    BOOTSTRAP_SAMPLE_SIZE = 150  # Fixed sample size for fair comparison
    
    # ==================== RUN ANALYSIS ====================
    
    # Initialize analyzer
    analyzer = DiagonalRemovedTADAnalyzer(
        intra_tads_folder=INTRA_TADS_FOLDER,
        all_contacts_folder=ALL_CONTACTS_FOLDER,
        barcode_file=BARCODE_FILE,
        target_celltypes=TARGET_CELLTYPES
    )
    
    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis(
            n_iterations=N_ITERATIONS,
            bootstrap_sample_size=BOOTSTRAP_SAMPLE_SIZE,
            output_dir=OUTPUT_DIR
        )
        
        # Print final comparison
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        
        comparison_df = pd.DataFrame([
            {
                'Cell_Type': ct,
                'N_Cells': results[ct]['n_cells'],
                'Intra-TAD %': f"{results[ct]['mean']:.2f} ± {results[ct]['std_error']:.3f}",
                '95% CI': f"[{results[ct]['ci_lower']:.2f}, {results[ct]['ci_upper']:.2f}]",
                'Original_Mean': f"{results[ct]['original_mean']:.2f}"
            }
            for ct in TARGET_CELLTYPES if ct in results
        ])
        
        print(comparison_df.to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()