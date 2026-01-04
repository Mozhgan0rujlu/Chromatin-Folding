#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 03:15:50 2025

@author: mozhganoroujlu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 02:42:07 2025

@author: mozhganoroujlu
"""

#!/usr/bin/env python3
"""
bulk_pca_projection_bandnorm_by_celltype.py

Adjusted to use Top 10% vs Bottom 10% means for CpG orientation (like Higashi script)
With additional visualization outputs and quantile-normalized score saving.
"""

import os
import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform
from scipy.stats import zscore, pearsonr
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import random

EPS = 1e-5
N_COMPONENTS = 10
WORKERS = 4
EPS_LOG = 1e-3
QUANTILE_RANGE = (-2.5, 2.5)  # New range for quantile normalization

# -------------------- USER PATHS --------------------
txt_dir = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/bandnorm/bandnorm_txt_non_neuron/"          # folder with per-barcode .txt files
cpg_txt = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/chrom1_cpg_ratios.txt"         # CpG ratios text file
celltype_file = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/non_neuron.tsv"  # tab-separated barcode -> celltype
out_dir = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/output_compartments/heatmaps/"   # output folder
quantile_out_dir = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/bandnorm/quantile_scores/"  # folder for quantile normalized scores
chrom = "chr1"                                         # chromosome to process
save_zscore = False                                   # optional: save z-scored compartments
# -----------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
os.makedirs(quantile_out_dir, exist_ok=True)

# -------------------- FUNCTIONS --------------------
def load_cpg_txt(cpg_txt):
    """Load CpG ratios from text file with columns: bin_id, cpg_ratio"""
    df = pd.read_csv(cpg_txt, sep='\t')
    if 'bin_id' not in df.columns or 'cpg_ratio' not in df.columns:
        raise ValueError("CpG text file must contain 'bin_id' and 'cpg_ratio' columns")
    return df

def read_txt_contacts(path):
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['bin1','bin2','c'], engine='c', dtype={'bin1':int,'bin2':int,'c':float})
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', header=None, names=['bin1','bin2','c'], engine='c', dtype={'bin1':int,'bin2':int,'c':float})
    return df

def build_dense_Q(contact_df, num_bins):
    Q = np.zeros((num_bins, num_bins), dtype=float)
    for i,j,v in contact_df[['bin1','bin2','c']].itertuples(index=False):
        if i < 0 or j < 0 or i >= num_bins or j >= num_bins:
            continue
        Q[i,j] += v
        Q[j,i] += v
    np.fill_diagonal(Q, 0.0)
    return Q

def compute_decay(Q):
    n = Q.shape[0]
    decay = np.zeros(n, dtype=float)
    for d in range(n):
        diag = np.diag(Q, d)
        decay[d] = np.nanmean(diag) if diag.size > 0 else 0.0
    decay = np.where(decay <= 0, EPS, decay)
    return decay

def build_E_from_Q(Q, decay):
    n = Q.shape[0]
    E = np.zeros_like(Q, dtype=float)
    for d in range(1, n):
        if decay[d] == 0: continue
        rows = np.arange(0, n-d)
        cols = rows + d
        E[rows, cols] = (Q[rows, cols] + EPS) / (decay[d] + EPS)
        E[cols, rows] = E[rows, cols]
    np.fill_diagonal(E, 1.0)
    return E

def safe_corr_from_E(E, eps_log):
    M = np.log2(E + eps_log)
    row_std = np.nanstd(M, axis=1)
    const_rows = np.where(row_std == 0)[0]
    if const_rows.size > 0:
        M[const_rows, :] += np.random.normal(scale=1e-6, size=(len(const_rows), M.shape[1]))
    C = np.corrcoef(M)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

def compute_bulk_Q(txt_paths, num_bins):
    """Compute bulk contact matrix without min_contacts validation"""
    bulk_Q = np.zeros((num_bins, num_bins), dtype=float)
    used_barcodes = []
    for p in txt_paths:
        try:
            df = read_txt_contacts(p)
            Q = build_dense_Q(df, num_bins)
            bulk_Q += Q
            used_barcodes.append(p)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue
    if len(used_barcodes) == 0:
        raise RuntimeError("No barcodes could be processed")
    return bulk_Q, used_barcodes

def fit_bulk_pca(bulk_Q, n_components, eps_log):
    decay = compute_decay(bulk_Q)
    E_bulk = build_E_from_Q(bulk_Q, decay)
    C_bulk = safe_corr_from_E(E_bulk, eps_log)
    pca = PCA(n_components=n_components)
    pcs_bulk = pca.fit_transform(C_bulk)
    return pca, pcs_bulk

def orient_bulk_by_cpg_extremes(pcs_bulk, cpg_values):
    """
    Use Top 10% vs Bottom 10% means for orientation (like Higashi script)
    Returns: flip = True if compartments need to be reversed
    """
    pc1 = pcs_bulk[:, 0]
    
    # Get top 10% and bottom 10% of PC1 values
    top_10_threshold = np.quantile(pc1, 0.9)
    bottom_10_threshold = np.quantile(pc1, 0.1)
    
    top_10_mask = pc1 > top_10_threshold
    bottom_10_mask = pc1 < bottom_10_threshold
    
    # Calculate mean CpG ratios for extremes
    top_10_cpg_mean = np.nanmean(cpg_values[top_10_mask])
    bottom_10_cpg_mean = np.nanmean(cpg_values[bottom_10_mask])
    
    print(f"Top 10% CpG mean: {top_10_cpg_mean:.4f}, Bottom 10% CpG mean: {bottom_10_cpg_mean:.4f}")
    
    # If top compartments (putative A) have LOWER CpG than bottom compartments (putative B), flip
    flip = top_10_cpg_mean < bottom_10_cpg_mean
    
    if flip:
        print("  → FLIPPING compartments (A compartments should have higher CpG)")
    else:
        print("  → Keeping original orientation")
    
    return flip

def quantile_normalize_scores(scores, valid_idx, num_bins):
    """
    Quantile normalize scores using maximum number of quantiles as 20% of number of bins
    and map to range [-2.5, 2.5]
    """
    n_quantiles = max(2, int(0.2 * num_bins))  # 20% of bins, minimum 2 quantiles
    
    if len(valid_idx) == 0:
        return scores
    
    # Extract valid scores for normalization
    valid_scores = scores[valid_idx]
    
    # Perform quantile normalization
    try:
        # Use uniform distribution as base and then map to desired range
        q_vals = quantile_transform(
            valid_scores.reshape(1, -1), 
            axis=1, 
            output_distribution='uniform',
            n_quantiles=min(n_quantiles, len(valid_scores))
        )[0]
        
        # Map from [0,1] to [-2.5, 2.5]
        # 0 -> -2.5, 0.5 -> 0, 1 -> 2.5
        normalized_scores = q_vals * 5 - 2.5
        
        # Apply normalized scores back to valid positions
        scores_normalized = scores.copy()
        scores_normalized[valid_idx] = normalized_scores
        
        return scores_normalized
        
    except Exception as e:
        print(f"Warning: Quantile normalization failed, using original scores: {e}")
        return scores

def plot_quantile_distributions(celltype_results, quantile_out_dir):
    """Plot distribution of quantile-normalized scores for each cell type"""
    for celltype, results in celltype_results.items():
        if len(results) == 0:
            continue
            
        print(f"Creating quantile distribution plot for {celltype}")
        
        # Load quantile-normalized compartment scores
        all_scores = []
        cell_barcodes = []
        
        for barcode, status, data in results:
            if status == 'ok':
                quantile_file = os.path.join(quantile_out_dir, f"{barcode}_quantile.txt")
                if os.path.exists(quantile_file):
                    df = pd.read_csv(quantile_file, sep='\t', header=None, names=['bin_id', 'score'])
                    all_scores.extend(df['score'].values)
                    cell_barcodes.append(barcode)
        
        if len(all_scores) == 0:
            continue
            
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Quantile-normalized Compartment Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution - {celltype}\n(n={len(cell_barcodes)} cells)')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        cell_scores = []
        for barcode, status, data in results:
            if status == 'ok':
                quantile_file = os.path.join(quantile_out_dir, f"{barcode}_quantile.txt")
                if os.path.exists(quantile_file):
                    df = pd.read_csv(quantile_file, sep='\t', header=None, names=['bin_id', 'score'])
                    cell_scores.append(df['score'].values)
        
        if cell_scores:
            plt.boxplot(cell_scores, labels=[''] * len(cell_scores))
            plt.ylabel('Quantile-normalized Compartment Score')
            plt.title(f'Per-cell Distribution - {celltype}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(quantile_out_dir, f'quantile_distribution_{celltype}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_aggregated_matrices_with_compartments(celltype_groups, bulk_pcas, bulk_flips, cpg_bins, out_dir):
    """First output: Raw contacts heatmap with compartment scores below"""
    for celltype, files in celltype_groups.items():
        print(f"Creating aggregated matrix plot for {celltype}")
        
        # Compute bulk matrix
        bulk_Q, _ = compute_bulk_Q(files, cpg_bins.shape[0])
        
        # Get compartment scores
        pca = bulk_pcas[celltype]
        flip = bulk_flips[celltype]
        decay = compute_decay(bulk_Q)
        E_bulk = build_E_from_Q(bulk_Q, decay)
        C_bulk = safe_corr_from_E(E_bulk, EPS_LOG)
        pcs_bulk = pca.transform(C_bulk)
        pc1 = pcs_bulk[:, 0]
        if flip: pc1 = -pc1
        
        # Apply quantile normalization ONLY for plotting (not saving)
        binfilter = np.any(bulk_Q != 0, axis=1)
        valid_idx = np.where(binfilter)[0]
        pc1_plot = quantile_normalize_scores(pc1, valid_idx, cpg_bins.shape[0])
        
        # Create figure with subplots - make it square
        fig = plt.figure(figsize=(12, 12))  # Square figure
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Top: Contact matrix heatmap (use raw contacts, no log transform)
        ax1 = fig.add_subplot(gs[0])
        # Use raw contacts without log transformation
        im = ax1.imshow(bulk_Q, cmap='Reds', aspect='auto', interpolation='none')
        ax1.set_title(f'Aggregated Contact Matrix - {celltype}')
        ax1.set_ylabel('Bin ID')
        plt.colorbar(im, ax=ax1, label='Raw Contacts')
        
        # Bottom: Compartment scores (using quantile-normalized for plotting only)
        ax2 = fig.add_subplot(gs[1])
        bins = np.arange(len(pc1_plot))
        ax2.plot(bins, pc1_plot, color='blue', linewidth=1)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Bin ID')
        ax2.set_ylabel('Compartment Score')
        ax2.set_title('Bulk Compartment Scores (Quantile-normalized for visualization)')
        ax2.set_ylim(-2.5, 2.5)  # Set y-axis limits to [-2.5, 2.5]
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'aggregated_matrix_{celltype}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_opc_comparison(celltype_results, bulk_pcas, bulk_flips, cpg_bins, out_dir):
    """Plot comparison between OPC bulk and a random OPC single cell before quantile normalization"""
    if 'OPC' not in celltype_results:
        print("OPC cell type not found, skipping OPC comparison plot")
        return
        
    opc_results = celltype_results['OPC']
    if len(opc_results) == 0:
        print("No OPC cells found, skipping OPC comparison plot")
        return
        
    print("Creating OPC bulk vs single cell comparison plot...")
    
    # Get OPC files to compute bulk
    opc_files = []
    for barcode, status, data in opc_results:
        if status == 'ok':
            opc_files.append(os.path.join(txt_dir, f"{barcode}.txt"))
    
    if not opc_files:
        print("No valid OPC files found")
        return
    
    # Compute OPC bulk compartment scores (before quantile normalization)
    bulk_Q, _ = compute_bulk_Q(opc_files, cpg_bins.shape[0])
    pca = bulk_pcas['OPC']
    flip = bulk_flips['OPC']
    decay = compute_decay(bulk_Q)
    E_bulk = build_E_from_Q(bulk_Q, decay)
    C_bulk = safe_corr_from_E(E_bulk, EPS_LOG)
    pcs_bulk = pca.transform(C_bulk)
    bulk_pc1 = pcs_bulk[:, 0]
    if flip: bulk_pc1 = -bulk_pc1
    
    # Pick a random OPC single cell and compute its compartment scores (before quantile normalization)
    random_opc_result = random.choice([r for r in opc_results if r[1] == 'ok'])
    random_barcode = random_opc_result[0]
    
    print(f"Selected random OPC cell: {random_barcode}")
    
    # Process the random cell to get raw compartment scores
    random_cell_path = os.path.join(txt_dir, f"{random_barcode}.txt")
    df = read_txt_contacts(random_cell_path)
    num_bins = cpg_bins.shape[0]
    Q = build_dense_Q(df, num_bins)
    decay_bar = compute_decay(Q)
    E_bar = build_E_from_Q(Q, decay_bar)
    C_bar = safe_corr_from_E(E_bar, EPS_LOG)
    pcs_bar = pca.transform(C_bar)
    single_cell_pc1 = pcs_bar[:, 0]
    if flip: single_cell_pc1 = -single_cell_pc1
    
    # Create comparison plot
    plt.figure(figsize=(14, 8))
    
    bins = np.arange(len(bulk_pc1))
    
    # Plot both scores
    plt.plot(bins, bulk_pc1, color='blue', linewidth=2, label='OPC Bulk', alpha=0.8)
    plt.plot(bins, single_cell_pc1, color='red', linewidth=1.5, label=f'Single OPC Cell ({random_barcode})', alpha=0.8)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Bin ID', fontsize=12)
    plt.ylabel('Compartment Score (Before Quantile Normalization)', fontsize=12)
    plt.title('OPC Bulk vs Single Cell Compartment Scores\n(Before Quantile Normalization)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    bulk_range = f"Bulk range: [{bulk_pc1.min():.3f}, {bulk_pc1.max():.3f}]"
    single_range = f"Single cell range: [{single_cell_pc1.min():.3f}, {single_cell_pc1.max():.3f}]"
    correlation = pearsonr(bulk_pc1, single_cell_pc1)[0]
    
    plt.text(0.02, 0.98, f'{bulk_range}\n{single_range}\nCorrelation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'opc_bulk_vs_single_cell_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OPC comparison plot saved with cell {random_barcode}")

def plot_celltype_correlation_lineplots(celltype_results, out_dir):
    """Second output: Line plots showing correlation between single cells and bulk for each cell type"""
    if not celltype_results:
        return
        
    print("Creating correlation line plots for all cell types...")
    
    # Determine subplot layout
    n_celltypes = len(celltype_results)
    n_cols = min(3, n_celltypes)  # Maximum 3 columns
    n_rows = (n_celltypes + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_celltypes == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for idx, (celltype, results) in enumerate(celltype_results.items()):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        # Load compartment scores for all cells in this cell type
        cell_scores = []
        cell_barcodes = []
        
        for barcode, status, data in results:
            if status == 'ok':
                quantile_file = os.path.join(quantile_out_dir, f"{barcode}_quantile.txt")
                if os.path.exists(quantile_file):
                    df = pd.read_csv(quantile_file, sep='\t', header=None, names=['bin_id', 'score'])
                    cell_scores.append(df['score'].values)
                    cell_barcodes.append(barcode)
        
        if len(cell_scores) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {celltype}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{celltype} (n=0)', fontsize=10)
            continue
            
        cell_scores = np.array(cell_scores)  # n_cells x n_bins
        
        # Compute bulk compartment scores for this cell type
        bulk_scores = np.mean(cell_scores, axis=0)
        
        # Compute correlation between bulk and each single cell
        correlations = []
        for i in range(cell_scores.shape[0]):
            corr = pearsonr(bulk_scores, cell_scores[i])[0]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Sort correlations for better visualization
        sorted_indices = np.argsort(correlations)
        sorted_correlations = correlations[sorted_indices]
        
        # Create smooth line plot
        x_positions = np.arange(len(sorted_correlations))
        
        # Plot with smooth line and markers
        ax.plot(x_positions, sorted_correlations, 
               color='darkblue', linewidth=2, alpha=0.8, marker='o', 
               markersize=3, markerfacecolor='red', markeredgecolor='red')
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Cell Index (sorted)', fontsize=9)
        ax.set_ylabel('Correlation with Bulk', fontsize=9)
        ax.set_title(f'{celltype} (n={len(correlations)})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim(-1.1, 1.1)
        
        # Add some statistics to the plot
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        ax.text(0.02, 0.98, f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(celltype_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_lineplots_all_celltypes.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_cell_correlation_heatmaps(celltype_results, out_dir):
    """Third output: n*n correlation matrices heatmap per cell type"""
    for celltype, results in celltype_results.items():
        if len(results) < 2:  # Need at least 2 cells for correlation
            continue
            
        print(f"Creating single-cell correlation matrix for {celltype}")
        
        # Load quantile-normalized compartment scores
        cell_scores = []
        cell_barcodes = []
        
        for barcode, status, data in results:
            if status == 'ok':
                quantile_file = os.path.join(quantile_out_dir, f"{barcode}.txt")
                if os.path.exists(quantile_file):
                    df = pd.read_csv(quantile_file, sep='\t', header=None, names=['bin_id', 'score'])
                    cell_scores.append(df['score'].values)
                    cell_barcodes.append(barcode)
        
        if len(cell_scores) < 2:
            continue
            
        cell_scores = np.array(cell_scores)  # n_cells x n_bins
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(cell_scores)
        
        # Plot heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, 
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Pearson r'})
        plt.title(f'Single-Cell Compartment Score Correlations - {celltype}\n(n={len(cell_scores)} cells)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'single_cell_correlation_{celltype}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def process_single_barcode(path, out_dir, pca, bulk_flip, cpg_bins):
    """Process single barcode without min_valid_bins validation"""
    barcode = os.path.splitext(os.path.basename(path))[0]
    try:
        print(f"Processing {barcode}...")
        df = read_txt_contacts(path)
        num_bins = cpg_bins.shape[0]
        Q = build_dense_Q(df, num_bins)
        binfilter = np.any(Q != 0, axis=1)
        
        # Remove min_valid_bins check since all cells are pre-validated
        decay_bar = compute_decay(Q)
        E_bar = build_E_from_Q(Q, decay_bar)
        C_bar = safe_corr_from_E(E_bar, EPS_LOG)
        pcs_bar = pca.transform(C_bar)
        pc1 = pcs_bar[:,0]
        if bulk_flip: pc1 = -pc1

        # Get valid indices for normalization
        valid_idx = np.where(binfilter)[0]
        
        # Apply quantile normalization with 20% of bins as max quantiles and range [-2.5, 2.5]
        pc1_normalized = quantile_normalize_scores(pc1, valid_idx, num_bins)

        # Save quantile normalized scores in 2-column format ONLY in quantile_out_dir
        output_data = []
        for bin_id, score in enumerate(pc1_normalized):
            output_data.append([bin_id, score])
        
        output_df = pd.DataFrame(output_data, columns=['bin_id', 'compartment_score'])
        quantile_file = os.path.join(quantile_out_dir, f"{barcode}.txt")
        output_df.to_csv(quantile_file, sep='\t', header=False, index=False, float_format='%.6f')
        
        print(f"Successfully processed {barcode} (range: [{pc1_normalized.min():.3f}, {pc1_normalized.max():.3f}])")
        return (barcode, 'ok', pc1_normalized)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error processing {barcode}: {e}")
        return (barcode, 'error', str(e) + "\n" + tb)

# -------------------- MAIN --------------------
print("Loading CpG bins...")
cpg_bins = load_cpg_txt(cpg_txt)
num_bins = cpg_bins.shape[0]

# Load cell type info
celltype_df = pd.read_csv(celltype_file, sep='\t')
barcode_to_celltype = dict(zip(celltype_df['barcode'], celltype_df['celltype']))

# Collect txt files and group by cell type
txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
celltype_groups = {}
for f in txt_files:
    barcode = os.path.splitext(f)[0]
    celltype = barcode_to_celltype.get(barcode)
    if celltype is None: 
        print(f"Warning: No cell type found for barcode {barcode}, skipping")
        continue
    celltype_groups.setdefault(celltype, []).append(os.path.join(txt_dir, f))

print(f"Found {len(txt_files)} files, grouped into {len(celltype_groups)} cell types.")

# Compute bulk PCA per cell type
bulk_pcas = {}
bulk_flips = {}
for celltype, files in celltype_groups.items():
    print(f"Processing bulk PCA for cell type {celltype} with {len(files)} barcodes...")
    bulk_Q, used = compute_bulk_Q(files, num_bins)
    pca, pcs_bulk = fit_bulk_pca(bulk_Q, N_COMPONENTS, EPS_LOG)
    flip = orient_bulk_by_cpg_extremes(pcs_bulk, cpg_bins['cpg_ratio'].values)
    bulk_pcas[celltype] = pca
    bulk_flips[celltype] = flip
    print(f"Cell type {celltype}: bulk PCA done, flip={flip}")

# First output: Plot aggregated matrices with compartment scores
plot_aggregated_matrices_with_compartments(celltype_groups, bulk_pcas, bulk_flips, cpg_bins, out_dir)

# Process all barcodes sequentially (no parallel processing)
print("Processing single barcodes sequentially...")
all_txt_paths = [os.path.join(txt_dir, f) for f in txt_files]
results = []
celltype_results = {}

processed_count = 0
total_count = len(all_txt_paths)

for path in all_txt_paths:
    barcode = os.path.splitext(os.path.basename(path))[0]
    celltype = barcode_to_celltype.get(barcode)
    if celltype is None: 
        continue
    
    print(f"Processing {barcode} ({processed_count + 1}/{total_count})...")
    
    res = process_single_barcode(path, out_dir, bulk_pcas[celltype], bulk_flips[celltype], cpg_bins)
    results.append(res)
    
    # Group results by cell type
    if celltype not in celltype_results:
        celltype_results[celltype] = []
    celltype_results[celltype].append(res)
    
    processed_count += 1

print(f"Processed {processed_count} barcodes")

# New plot: OPC bulk vs single cell comparison
plot_opc_comparison(celltype_results, bulk_pcas, bulk_flips, cpg_bins, out_dir)

# Plot quantile distribution for each cell type
plot_quantile_distributions(celltype_results, quantile_out_dir)

# Second output: Plot correlation line plots
plot_celltype_correlation_lineplots(celltype_results, out_dir)

# Third output: Plot single-cell correlation matrices
plot_single_cell_correlation_heatmaps(celltype_results, out_dir)

print(f"All done. Outputs in {out_dir}")
print(f"Quantile normalized scores in {quantile_out_dir}")

# Print summary
success_count = sum(1 for r in results if r[1] == 'ok')
error_count = sum(1 for r in results if r[1] == 'error')
print(f"Summary: {success_count} successful, {error_count} errors")

# Print quantile normalization info
print(f"Quantile normalization: range [{QUANTILE_RANGE[0]}, {QUANTILE_RANGE[1]}], max quantiles = 20% of bins")