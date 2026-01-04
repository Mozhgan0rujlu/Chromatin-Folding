#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 21:48:16 2025

@author: mozhganoroujlu
"""

import pysam
import pandas as pd
import numpy as np
import h5py
import os

def calculate_cpg_ratio(fasta_path, chrom_size_path, hdf_output_path, resolution=500000):
    """
    Calculate CpG ratio for genomic bins at a specified resolution.
    
    Args:
        fasta_path (str): Path to the genome FASTA file.
        chrom_size_path (str): Path to the chromosome sizes file.
        hdf_output_path (str): Path to save the output HDF5 file.
        resolution (int): Bin size in base pairs (default: 100,000).
    """
    # Validate input files
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not os.path.exists(chrom_size_path):
        raise FileNotFoundError(f"Chromosome sizes file not found: {chrom_size_path}")

    # Read chromosome sizes
    chrom_sizes = pd.read_csv(chrom_size_path, sep='\t', header=None, names=['chrom', 'length'])
    
    # Initialize lists to store results
    data = {'chrom': [], 'start': [], 'end': [], 'cpg_ratio': []}
    
    # Open FASTA file
    with pysam.Fastafile(fasta_path) as fasta:
        for _, row in chrom_sizes.iterrows():
            chrom = row['chrom']
            chrom_length = row['length']
            print(f"Processing {chrom}...")
            
            # Get sequence for the chromosome
            try:
                sequence = fasta.fetch(chrom).upper()
            except ValueError as e:
                print(f"Warning: Could not fetch sequence for {chrom}. Skipping. Error: {e}")
                continue
            
            # Iterate over bins
            for start in range(0, chrom_length, resolution):
                end = min(start + resolution, chrom_length)
                bin_sequence = sequence[start:end]
                
                # Count CpG dinucleotides
                cpg_count = bin_sequence.count('CG')
                bin_length = len(bin_sequence)
                
                # Calculate CpG ratio (number of CpG sites divided by bin length)
                cpg_ratio = cpg_count / bin_length if bin_length > 0 else 0
                
                # Store results
                data['chrom'].append(chrom)
                data['start'].append(start)
                data['end'].append(end)
                data['cpg_ratio'].append(cpg_ratio)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to HDF5
    with h5py.File(hdf_output_path, 'w') as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].values)
    
    print(f"CpG ratios saved to {hdf_output_path}")

def main():
    # Hardcoded file paths
    fasta_path = "mm10.fa"
    hdf_output_path = "mm10_cpg_ratio_500k.hdf"
    chrom_size_path = "chrom_sizes.txt"
    resolution = 500000  # Default resolution of 100 kb

    calculate_cpg_ratio(
        fasta_path=fasta_path,
        chrom_size_path=chrom_size_path,
        hdf_output_path=hdf_output_path,
        resolution=resolution
    )

if __name__ == "__main__":
    main()
