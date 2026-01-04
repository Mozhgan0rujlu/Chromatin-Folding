#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive visualization for both MoC and AMI similarity matrices.
Enhanced version with detailed plots for both metrics.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import sklearn.manifold
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Input and output directories
input_dir = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/bandnorm/TADs_non_neuron/"
output_dir = "/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/similarity_results"

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load similarity matrices
print("Loading similarity matrices...")
moc_df = pd.read_excel(os.path.join(output_dir, "simi_moc.xlsx"), index_col=0)
ami_df = pd.read_excel(os.path.join(output_dir, "simi_ami.xlsx"), index_col=0)

simi_moc = moc_df.values
simi_ami = ami_df.values
barcodes = moc_df.index.tolist()

n_cells = simi_moc.shape[0]
print(f"Loaded matrices for {n_cells} cells")

def create_comprehensive_figure(similarity_matrix, matrix_name, color_scheme):
    """
    Create comprehensive figure for a similarity matrix
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Flattened matrix (off-diagonal elements)
    matrix_flat = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    # 1. Similarity Matrix Heatmap
    plt.subplot(2, 3, 1)
    im = plt.imshow(similarity_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'{matrix_name} Similarity Matrix\n{n_cells} cells', fontsize=12, fontweight='bold')
    plt.xlabel('Cell Index')
    plt.ylabel('Cell Index')
    
    # 2. Distribution
    plt.subplot(2, 3, 2)
    plt.hist(matrix_flat, bins=50, color=color_scheme['hist'], edgecolor='black', alpha=0.7)
    plt.title(f'{matrix_name} Distribution (off-diagonal)')
    plt.xlabel(f'{matrix_name} Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add distribution statistics
    dist_stats = f'Mean: {np.mean(matrix_flat):.3f}\nStd: {np.std(matrix_flat):.3f}'
    plt.text(0.95, 0.95, dist_stats, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. MDS - Main plot
    plt.subplot(2, 3, (3, 6))
    
    try:
        mds = sklearn.manifold.MDS(
            n_components=2, 
            dissimilarity='precomputed', 
            random_state=42,
            max_iter=1000,
            eps=1e-9,
            normalized_stress='auto'
        )
        embedding = mds.fit_transform(1 - similarity_matrix)
        
        # Calculate optimal point size
        point_size = max(3, 2000 / n_cells)
        
        # Create scatter plot with color based on density
        if len(embedding) > 10:
            xy = embedding.T
            z = gaussian_kde(xy)(xy)
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=z, cmap=color_scheme['cmap'], s=point_size, alpha=0.7)
            plt.colorbar(scatter, label='Point Density')
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], 
                       c=color_scheme['points'], s=point_size*2, alpha=0.8)
        
        plt.title(f'{matrix_name} + MDS\n{n_cells} cells | Stress: {mds.stress_:.4f}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('MDS Component 1')
        plt.ylabel('MDS Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'{matrix_name} Statistics:\nMean: {np.mean(matrix_flat):.3f}\nStd: {np.std(matrix_flat):.3f}\nMin: {np.min(matrix_flat):.3f}\nMax: {np.max(matrix_flat):.3f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    except Exception as e:
        plt.text(0.5, 0.5, f'MDS failed:\n{str(e)}', ha='center', va='center', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.3))
        plt.title(f'{matrix_name} MDS - Failed')
    
    plt.tight_layout()
    return fig

# Create MoC comprehensive figure
print("Creating MoC comprehensive figure...")
moc_colors = {
    'hist': 'lightblue',
    'cmap': 'Blues',
    'points': 'blue'
}
moc_fig = create_comprehensive_figure(simi_moc, "MoC", moc_colors)
moc_fig.savefig(os.path.join(output_dir, "moc_comprehensive_analysis.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(moc_fig)

# Create AMI comprehensive figure
print("Creating AMI comprehensive figure...")
ami_colors = {
    'hist': 'lightcoral',
    'cmap': 'Reds',
    'points': 'red'
}
ami_fig = create_comprehensive_figure(simi_ami, "AMI", ami_colors)
ami_fig.savefig(os.path.join(output_dir, "ami_comprehensive_analysis.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(ami_fig)

# Create combined comparison figure
print("Creating combined comparison figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Compute MDS for both matrices
try:
    mds = sklearn.manifold.MDS(
        n_components=2, 
        dissimilarity='precomputed', 
        random_state=42,
        max_iter=1000,
        eps=1e-9
    )
    
    embedding_moc = mds.fit_transform(1 - simi_moc)
    embedding_ami = mds.fit_transform(1 - simi_ami)
    
    point_size = max(3, 1500 / n_cells)
    
    # Row 1: MoC visualizations
    # MoC Heatmap
    im1 = axes[0,0].imshow(simi_moc, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(im1, ax=axes[0,0])
    axes[0,0].set_title('MoC Similarity Matrix')
    axes[0,0].set_xlabel('Cell Index')
    axes[0,0].set_ylabel('Cell Index')
    
    # MoC Distribution
    moc_flat = simi_moc[np.triu_indices_from(simi_moc, k=1)]
    axes[0,1].hist(moc_flat, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    axes[0,1].set_title('MoC Distribution')
    axes[0,1].set_xlabel('MoC Value')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # MoC MDS with density coloring
    if len(embedding_moc) > 10:
        xy_moc = embedding_moc.T
        z_moc = gaussian_kde(xy_moc)(xy_moc)
        scatter1 = axes[0,2].scatter(embedding_moc[:, 0], embedding_moc[:, 1], 
                                    c=z_moc, cmap='Blues', s=point_size, alpha=0.7)
        plt.colorbar(scatter1, ax=axes[0,2], label='Point Density')
    else:
        scatter1 = axes[0,2].scatter(embedding_moc[:, 0], embedding_moc[:, 1], 
                                    c='blue', s=point_size, alpha=0.7)
    axes[0,2].set_title(f'MoC MDS ({n_cells} cells)')
    axes[0,2].set_xlabel('Component 1')
    axes[0,2].set_ylabel('Component 2')
    axes[0,2].grid(True, alpha=0.3)
    
    # Row 2: AMI visualizations
    # AMI Heatmap
    im2 = axes[1,0].imshow(simi_ami, cmap='Reds', aspect='auto', interpolation='nearest')
    plt.colorbar(im2, ax=axes[1,0])
    axes[1,0].set_title('AMI Similarity Matrix')
    axes[1,0].set_xlabel('Cell Index')
    axes[1,0].set_ylabel('Cell Index')
    
    # AMI Distribution
    ami_flat = simi_ami[np.triu_indices_from(simi_ami, k=1)]
    axes[1,1].hist(ami_flat, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1,1].set_title('AMI Distribution')
    axes[1,1].set_xlabel('AMI Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    # AMI MDS with density coloring
    if len(embedding_ami) > 10:
        xy_ami = embedding_ami.T
        z_ami = gaussian_kde(xy_ami)(xy_ami)
        scatter2 = axes[1,2].scatter(embedding_ami[:, 0], embedding_ami[:, 1], 
                                    c=z_ami, cmap='Reds', s=point_size, alpha=0.7)
        plt.colorbar(scatter2, ax=axes[1,2], label='Point Density')
    else:
        scatter2 = axes[1,2].scatter(embedding_ami[:, 0], embedding_ami[:, 1], 
                                    c='red', s=point_size, alpha=0.7)
    axes[1,2].set_title(f'AMI MDS ({n_cells} cells)')
    axes[1,2].set_xlabel('Component 1')
    axes[1,2].set_ylabel('Component 2')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_moc_ami_comparison.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
except Exception as e:
    print(f"Combined figure MDS failed: {e}")

# Create summary statistics
print("Creating summary statistics...")
summary_stats = {
    'Metric': ['MoC', 'AMI'],
    'Mean': [np.mean(simi_moc[np.triu_indices_from(simi_moc, k=1)]), 
             np.mean(simi_ami[np.triu_indices_from(simi_ami, k=1)])],
    'Std': [np.std(simi_moc[np.triu_indices_from(simi_moc, k=1)]), 
            np.std(simi_ami[np.triu_indices_from(simi_ami, k=1)])],
    'Min': [np.min(simi_moc[np.triu_indices_from(simi_moc, k=1)]), 
            np.min(simi_ami[np.triu_indices_from(simi_ami, k=1)])],
    'Max': [np.max(simi_moc[np.triu_indices_from(simi_moc, k=1)]), 
            np.max(simi_ami[np.triu_indices_from(simi_ami, k=1)])],
    'Median': [np.median(simi_moc[np.triu_indices_from(simi_moc, k=1)]), 
               np.median(simi_ami[np.triu_indices_from(simi_ami, k=1)])]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_excel(os.path.join(output_dir, "similarity_summary_statistics.xlsx"), index=False)

# Export AMI MDS coordinates to TSV file
print("Exporting AMI MDS coordinates to TSV...")
try:
    # Compute AMI MDS with the same parameters as used in the figures
    mds_ami = sklearn.manifold.MDS(
        n_components=2, 
        dissimilarity='precomputed', 
        random_state=42,
        max_iter=1000,
        eps=1e-9
    )
    embedding_ami = mds_ami.fit_transform(1 - simi_ami)
    
    # Remove '_tads' suffix from barcodes
    cleaned_barcodes = [barcode.replace('_tads', '') for barcode in barcodes]
    
    # Create DataFrame with required columns
    ami_mds_df = pd.DataFrame({
        'barcode': cleaned_barcodes,
        'mds_component_1': embedding_ami[:, 0],
        'mds_component_2': embedding_ami[:, 1]
    })
    
    # Save as TSV
    ami_mds_df.to_csv(os.path.join(output_dir, "ami_mds_coordinates.tsv"), 
                     sep='\t', index=False)
    
    print(f"AMI MDS coordinates exported for {len(ami_mds_df)} cells")
    
except Exception as e:
    print(f"Error exporting AMI MDS coordinates: {e}")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"Processed: {n_cells} cells")
print(f"Files created:")
print(f"  - moc_comprehensive_analysis.png: MoC detailed analysis")
print(f"  - ami_comprehensive_analysis.png: AMI detailed analysis")
print(f"  - combined_moc_ami_comparison.png: Side-by-side comparison with density coloring")
print(f"  - similarity_summary_statistics.xlsx: Summary statistics")
print(f"  - ami_mds_coordinates.tsv: AMI MDS coordinates for further analysis")
print(f"Results saved to: {output_dir}")
print("="*60)