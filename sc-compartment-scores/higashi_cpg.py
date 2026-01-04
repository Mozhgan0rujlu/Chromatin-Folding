#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:47:15 2025

@author: mozhganoroujlu
"""

import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def cal_cpg(bin_str):
    """
    Calculate CpG ratio with proper handling of 'N' nucleotides
    Returns: cpg_ratio, N_count, NN_count
    """
    cg_count = bin_str.count("CG")
    # Each N leads to 2 2-mer with "N": "-N" and "N-"
    N_count = bin_str.count("N")
    # Count consecutive NN pairs to correct for overcounting
    NN_count = len(re.findall(r'(N)(?=\1)', bin_str))
    
    total_count = len(bin_str) - 1
    total_count = total_count - N_count * 2 + NN_count
    
    # Edge case corrections
    if bin_str[0] == 'N':
        total_count += 1
    if bin_str[-1] == 'N':
        total_count += 1
    
    if total_count > 0:
        rate = cg_count / total_count
        if rate < 0:
            print(cg_count, len(bin_str), N_count, NN_count)
            print(bin_str)
            raise EOFError
    else:
        rate = 0.0
    
    return rate, N_count, NN_count

def calculate_and_plot_chrom1_cpg_ratio(fasta_path, chrom_size_path, resolution=500000, output_txt_path=None):
    """
    Calculate CpG ratio for chromosome 1 with proper N handling and plot directly
    
    Args:
        fasta_path (str): Path to the genome FASTA file.
        chrom_size_path (str): Path to the chromosome sizes file.
        resolution (int): Bin size in base pairs (default: 500,000).
        output_txt_path (str): Path to save the CpG ratios as text file.
    """
    # Validate input files
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not os.path.exists(chrom_size_path):
        raise FileNotFoundError(f"Chromosome sizes file not found: {chrom_size_path}")

    # Set default output path if not provided
    if output_txt_path is None:
        output_txt_path = "chrom1_cpg_ratios.txt"

    # Read chromosome sizes
    chrom_sizes = pd.read_csv(chrom_size_path, sep='\t', header=None, names=['chrom', 'length'])
    
    # Filter for chromosome 1 only
    chrom1_sizes = chrom_sizes[chrom_sizes['chrom'].astype(str).str.upper().isin(['CHR1', '1'])]
    
    if chrom1_sizes.empty:
        print("Available chromosomes:", chrom_sizes['chrom'].unique())
        raise ValueError("Chromosome 1 not found in chromosome sizes file")
    
    # Initialize lists to store results for chromosome 1
    chrom1_cpg_ratios = []
    bin_starts = []
    total_N_counts = []
    total_NN_counts = []
    
    # Open FASTA file
    with pysam.Fastafile(fasta_path) as fasta:
        for _, row in chrom1_sizes.iterrows():
            chrom = row['chrom']
            chrom_length = row['length']
            print(f"Processing {chrom}...")
            
            # Get sequence for chromosome 1
            try:
                sequence = fasta.fetch(chrom).upper()
            except ValueError as e:
                print(f"Error: Could not fetch sequence for {chrom}. Error: {e}")
                continue
            
            # Iterate over bins
            bin_number = 0
            for start in range(0, chrom_length, resolution):
                end = min(start + resolution, chrom_length)
                bin_sequence = sequence[start:end]
                
                # Calculate CpG ratio with proper N handling
                cpg_ratio, N_count, NN_count = cal_cpg(bin_sequence)
                
                # Store results for plotting and saving
                chrom1_cpg_ratios.append(cpg_ratio)
                bin_starts.append(start)
                total_N_counts.append(N_count)
                total_NN_counts.append(NN_count)
                bin_number += 1
    
    # Save CpG ratios to text file with numeric bin IDs
    with open(output_txt_path, 'w') as f:
        # Write header
        f.write("bin_id\tcpg_ratio\n")
        # Write data with numeric bin IDs (0, 1, 2, ...)
        for bin_id, cpg_ratio in enumerate(chrom1_cpg_ratios):
            f.write(f"{bin_id}\t{cpg_ratio:.6f}\n")
    
    print(f"CpG ratios saved to: {output_txt_path}")
    
    # Calculate statistics
    mean_ratio = np.mean(chrom1_cpg_ratios)
    median_ratio = np.median(chrom1_cpg_ratios)
    total_N = np.sum(total_N_counts)
    total_NN = np.sum(total_NN_counts)
    total_bases = len(chrom1_cpg_ratios) * resolution
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    
    # Plot CpG ratio vs bins for chromosome 1
    bins = np.arange(len(chrom1_cpg_ratios))
    plt.plot(bins, chrom1_cpg_ratios, linewidth=1, alpha=0.8, color='blue')
    
    # Customize the plot
    plt.xlabel(f'Genomic Bins on Chromosome 1 ({resolution//1000}kb each)')
    plt.ylabel('CpG Ratio')
    plt.title(f'CpG Ratio Across Chromosome 1 (mm10, {resolution//1000}kb resolution)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot with N nucleotide information
    plt.axhline(y=mean_ratio, color='r', linestyle='--', alpha=0.8, 
                label=f'Mean: {mean_ratio:.4f}')
    plt.axhline(y=median_ratio, color='g', linestyle='--', alpha=0.8, 
                label=f'Median: {median_ratio:.4f}')
    
    # Add N nucleotide information to legend
    plt.axhline(y=mean_ratio, color='white', linestyle='-', alpha=0,  # invisible line for spacing
                label=f'N percentage: {(total_N/total_bases*100):.2f}%')
    
    plt.legend()
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print detailed statistics
    print(f"\nChromosome 1 Statistics:")
    print(f"Number of bins: {len(chrom1_cpg_ratios)}")
    print(f"Total bases analyzed: {total_bases:,}")
    print(f"Mean CpG ratio: {mean_ratio:.6f}")
    print(f"Median CpG ratio: {median_ratio:.6f}")
    print(f"Standard deviation: {np.std(chrom1_cpg_ratios):.6f}")
    print(f"Min CpG ratio: {np.min(chrom1_cpg_ratios):.6f}")
    print(f"Max CpG ratio: {np.max(chrom1_cpg_ratios):.6f}")
    print(f"\nNucleotide Statistics:")
    print(f"Total N nucleotides detected: {total_N:,}")
    print(f"Total consecutive NN pairs: {total_NN:,}")
    print(f"Percentage of N nucleotides: {(total_N/total_bases*100):.4f}%")
    print(f"Average N per bin: {np.mean(total_N_counts):.1f}")
    print(f"Max N in single bin: {np.max(total_N_counts)}")
    
    return chrom1_cpg_ratios, bin_starts, total_N_counts, total_NN_counts

def main():
    # Hardcoded file paths
    fasta_path = "mm10.fa"
    chrom_size_path = "chrom_sizes.txt"
    output_txt_path = "chrom1_cpg_ratios.txt"
    resolution = 500000  # 500 kb resolution

    # Calculate and plot directly
    cpg_ratios, bin_starts, N_counts, NN_counts = calculate_and_plot_chrom1_cpg_ratio(
        fasta_path=fasta_path,
        chrom_size_path=chrom_size_path,
        resolution=resolution,
        output_txt_path=output_txt_path
    )

if __name__ == "__main__":
    main()
