#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 16:48:21 2025

@author: mozhganoroujlu
"""


"""this script compute compartment scores from raw cool files and .cpg ratio file fol matching the align of compartments"""

import cooler
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import os
import h5py
import time

# Paths and parameters
cool_dir = "cool_accurate_single_cell"
cpg_path = "mm10_cpg_ratio_500k.hdf"
resolution = 500000  # 500 kb resolution
chrom = "chr1"  # Target chromosome
output_file = "compartments.h5"  # Output HDF5 file
output_format = "hdf5"  # Options: "hdf5" or "csv"
update_interval = 30  # Update progress every 5 seconds

# Load CpG data manually from HDF5
with h5py.File(cpg_path, 'r') as f:
    chrom_data = f['chrom'][:]  # Byte strings
    start_data = f['start'][:]
    end_data = f['end'][:]
    cpg_ratio_data = f['cpg_ratio'][:]

# Convert byte strings to regular strings
chrom_data = [x.decode('utf-8') for x in chrom_data]

# Create DataFrame and filter for chr1
cpg_df = pd.DataFrame({
    'chrom': chrom_data,
    'start': start_data,
    'end': end_data,
    'cpg_ratio': cpg_ratio_data
})
cpg_chr1 = cpg_df[cpg_df['chrom'] == chrom].copy()

# Validate chr1 data
if len(cpg_chr1) == 0:
    raise ValueError(f"No data found for {chrom} in CpG file. Available chromosomes: {np.unique(chrom_data)}")
print(f"Loaded {len(cpg_chr1)} bins for {chrom} from CpG data.")

# Function to get bin filter
def get_binfilter(cool):
    matrix = cool.matrix(balance=False, sparse=True).fetch(chrom).toarray()
    if np.count_nonzero(matrix) == 0:
        raise ValueError("Hi-C matrix is completely empty")
    return np.all(matrix == 0, axis=1) == False

# Adapted compbulk for chr1
def compbulk(cool, cpg_df):
    # Fetch Hi-C matrix and bin table
    Q = cool.matrix(balance=False, sparse=True).fetch(chrom).toarray()
    Q = Q - np.diag(np.diag(Q))  # Remove diagonal
    bins_df = cool.bins()[:]  # Get bin table
    chr1_bins = bins_df[bins_df['chrom'] == chrom].copy()
    
    # Align CpG data with Cooler bins
    cpg_aligned = pd.merge(
        chr1_bins[['chrom', 'start', 'end']],
        cpg_df[['chrom', 'start', 'end', 'cpg_ratio']],
        on=['chrom', 'start', 'end'],
        how='left'  # Changed to 'left' to keep all Cooler bins
    )
    if len(cpg_aligned) == 0:
        raise ValueError(f"No matching bins between Cooler and CpG data for {chrom}")
    
    # Apply bin filter
    binfilter = get_binfilter(cool)
    if not np.any(binfilter):
        raise ValueError(f"No valid bins for {chrom} in {cool.filename}")
    
    # Filter Hi-C matrix and CpG data
    Q = Q[binfilter][:, binfilter]
    cpg_valid = cpg_aligned['cpg_ratio'].values[binfilter]
    cpg_valid = np.where(np.isnan(cpg_valid), 0, cpg_valid)  # Fill missing CpG with 0
    
    # Distance normalization
    decay = np.array([np.mean(np.diag(Q, i)) for i in range(Q.shape[0])])
    E = np.zeros(Q.shape)
    row, col = np.diag_indices(E.shape[0])
    E[row, col] = 1
    for i in range(1, E.shape[0]):
        E[row[:-i], col[i:]] = (Q[row[:-i], col[i:]] + 1e-5) / (decay[i] + 1e-5)
    E = E + E.T
    C = np.corrcoef(np.log2(E + 0.001))
    
    # PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(C)
    
    # Ensure lengths match
    if len(cpg_valid) != len(pc):
        raise ValueError(f"Mismatch in lengths: cpg_valid ({len(cpg_valid)}) vs pc ({len(pc)})")
    
    r = []
    for i in range(2):
        labels, groups = pd.qcut(pc[:,i], 50, labels=False, retbins=True, duplicates='drop')
        sad = np.array([[E[np.ix_(labels==i, labels==j)].sum() for i in range(50)] for j in range(50)])
        count = np.array([[(labels==i).sum()*(labels==j).sum() for i in range(50)] for j in range(50)])
        sad = sad / count
        r.append((sad[:10, :10].sum() + sad[-10:, -10:].sum()) / (sad[:10, -10:].sum() + sad[-10:, :10].sum()))
    
    i = 0 if r[0] > r[1] else 1
    if stats.pearsonr(cpg_valid, pc[:,i])[0] > 0:
        pc = pc[:,i]
        model = pca.components_[i]
    else:
        pc = -pc[:,i]
        model = -pca.components_[i]
    
    # Map scores back to all bins
    full_pc = np.zeros(len(chr1_bins))  # Initialize with zeros for all bins
    full_pc[binfilter] = pc  # Assign PCA scores to valid bins
    return full_pc, cpg_aligned[['chrom', 'start', 'end']].copy()

# Process all cool files
cool_files = [f for f in os.listdir(cool_dir) if f.endswith(".cool")]
barcodes = [f.replace(".cool", "") for f in cool_files]
results = []
completed_count = 0
total_barcodes = len(barcodes)
last_update_time = time.time()

for cool_file, barcode in zip(cool_files, barcodes):
    try:
        cool_path = os.path.join(cool_dir, cool_file)
        cool = cooler.Cooler(cool_path)
        
        # Check if normalization weights exist
        bins_df = cool.bins()[:]
        if 'weight' not in bins_df.columns:
            try:
                cooler.balance_cooler(cool, store=True, max_iters=1000)
            except Exception as e:
                raise ValueError(f"Normalization failed: {str(e)}")
        
        # Compute compartment scores
        pc, bin_coords = compbulk(cool, cpg_chr1)
        
        # Create DataFrame for this barcode
        result_df = pd.DataFrame({
            'barcode': barcode,
            'chrom': bin_coords['chrom'],
            'start': bin_coords['start'],
            'end': bin_coords['end'],
            'compartment_score': pc
        })
        results.append(result_df)
        
        # Update completed count
        completed_count += 1
        
        # Update progress line every 5 seconds
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            print(f"Computed scores for {completed_count}/{total_barcodes} barcodes", end='\r')
            last_update_time = current_time
    except Exception as e:
        print(f"\nError processing {barcode}: {str(e)}")
        continue

# Final progress update and move to a new line
print(f"\nComputed scores for {completed_count}/{total_barcodes} barcodes")

# Concatenate results and save to file
if results:
    final_df = pd.concat(results, ignore_index=True)
    if output_format == "hdf5":
        final_df.to_hdf(output_file, key='data', mode='w', complevel=5)
        print(f"Saved compartment scores to {output_file}")
    elif output_format == "csv":
        final_df.to_csv(output_file.replace('.h5', '.csv'), index=False)
        print(f"Saved compartment scores to {output_file.replace('.h5', '.csv')}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
else:
    print("No results to save.")
