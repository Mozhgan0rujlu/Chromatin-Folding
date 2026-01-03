


"""
This script converts single-cell Hi-C .pairs files into .cool contact matrices
for chromosome 1 and resolution 500kb. It parses chromosome sizes from file
headers, bins intra-chromosomal contacts, enforces matrix symmetry, and writes
one .cool file per input .pairs file using the cooler library.
"""

import pandas as pd
import numpy as np
import cooler
import os

# STEP 1: Parse chromosome sizes from header
def parse_chrom_sizes(filepath):
    chrom_sizes = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#chromsize:'):
                parts = line.strip().split()
                chrom = parts[1]
                size = int(parts[2])
                chrom_sizes[chrom] = size
            elif line.startswith('#columns:'):
                columns = line.strip().replace('#columns: ', '').split()
    return chrom_sizes, columns

# STEP 2: Load and filter raw Hi-C data from .pairs file
def load_and_filter_hic_data(filepath):
    chrom_sizes, columns = parse_chrom_sizes(filepath)
    df = pd.read_csv(
        filepath,
        sep='\t',
        comment='#',
        names=columns,
        header=None
    )

    df = df.dropna(subset=['pos1', 'pos2', 'strand1', 'strand2'])
    df = df.drop_duplicates(subset='readID')
    return df[['chrom1', 'pos1', 'chrom2', 'pos2']], chrom_sizes

# STEP 3: Bin intra-chromosomal contacts
def bin_contacts(df, chrom, bin_size):
    df = df[(df['chrom1'] == chrom) & (df['chrom2'] == chrom)].copy()
    df['bin1'] = df['pos1'] // bin_size
    df['bin2'] = df['pos2'] // bin_size
    contacts = df.groupby(['bin1', 'bin2']).size().reset_index(name='count')
    return contacts

# STEP 4: Make contact matrix symmetric
def make_symmetric(contacts_df):
    flipped = contacts_df.rename(columns={'bin1': 'bin2', 'bin2': 'bin1'})
    all_contacts = pd.concat([contacts_df, flipped], ignore_index=True)
    all_contacts['bin_min'] = all_contacts[['bin1', 'bin2']].min(axis=1)
    all_contacts['bin_max'] = all_contacts[['bin1', 'bin2']].max(axis=1)
    symmetric_contacts = (
        all_contacts.groupby(['bin_min', 'bin_max'], as_index=False)['count']
        .sum()
        .rename(columns={'bin_min': 'bin1', 'bin_max': 'bin2'})
    )
    return symmetric_contacts

# STEP 5: Build bins for .cool
def build_bins_df(chrom, max_pos, bin_size):
    bin_starts = np.arange(0, max_pos + bin_size, bin_size)
    bins_df = pd.DataFrame({
        'chrom': chrom,
        'start': bin_starts[:-1],
        'end': bin_starts[1:]
    })
    bins_df['end'] = bins_df['end'].clip(upper=max_pos)
    return bins_df

# STEP 6: Write .cool file
def write_cool_file(bins_df, pixels_df, out_path, bin_size):
    cooler.create_cooler(
        out_path,
        bins=bins_df,
        pixels=pixels_df,
        ordered=True,
        dtypes={'count': 'int32'},
        metadata={'bin-size': int(bin_size)}
    )

# STEP 7: Process a single .pairs file to create a .cool file
def process_pairs_file(filepath, chrom, bin_size, output_dir):
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Load and filter data
    df, chrom_sizes = load_and_filter_hic_data(filepath)

    # Check if chromosome exists
    if chrom not in chrom_sizes:
        raise ValueError(f"Chromosome {chrom} not found in file header for {filepath}.")
    max_pos = chrom_sizes[chrom]

    # Process contacts
    contacts_df = bin_contacts(df, chrom, bin_size)
    if contacts_df.empty:
        print(f"Warning: No contacts found in {filepath}. Skipping.")
        return

    contacts_df = make_symmetric(contacts_df)

    # Build bins, respecting chromosome size
    bins_df = build_bins_df(chrom, max_pos, bin_size)
    num_bins = len(bins_df)

    # Prepare pixels
    pixels_df = contacts_df.rename(columns={'bin1': 'bin1_id', 'bin2': 'bin2_id'})
    pixels_df = pixels_df[(pixels_df['bin1_id'] < num_bins) & (pixels_df['bin2_id'] < num_bins)]

    # Save .cool file
    cool_output = os.path.join(output_dir, f"{filename}.cool")
    write_cool_file(bins_df, pixels_df, cool_output, bin_size)
    print(f"Created .cool file: {cool_output}")

# STEP 8: Main pipeline for processing all .pairs files in a directory
def run_single_cell_pipeline(input_dir, chrom, bin_size, output_dir):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all .pairs files
    pairs_files = [f for f in os.listdir(input_dir) if f.endswith('.pairs')]
    if not pairs_files:
        raise ValueError(f"No .pairs files found in {input_dir}.")
    if len(pairs_files) != 6223:
        print(f"Warning: Found {len(pairs_files)} .pairs files instead of 6223.")

    # Process each .pairs file
    for pairs_file in pairs_files:
        filepath = os.path.join(input_dir, pairs_file)
        print(f"Processing {pairs_file}...")
        process_pairs_file(filepath, chrom, bin_size, output_dir)

    print(f"Processed {len(pairs_files)} .pairs files.")

# Example usage
if __name__ == "__main__":
    run_single_cell_pipeline(
        input_dir="pairs_single_cell",      # Input directory with .pairs files
        chrom="chr1",                       # Target chromosome
        bin_size=500000,                    # Bin size (500 kb)
        output_dir="accurate_cool_single_cell" # Output directory for .cool files
    )
