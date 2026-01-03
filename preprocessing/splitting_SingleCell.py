

"""
Created on Fri Aug 22 22:35:17 2025

@author: mozhganoroujlu
"""

"""
This script splits a Hi-C .pairs.gz file into separate single-cell
.pairs files based on barcode information, creating one output file per barcode
while preserving the original file header.
"""

import os
import pandas as pd
from time import perf_counter as pc
import gzip

# -----------------------------
# SET YOUR INPUT FILES HERE:
pairs_file = "merged_716_462_chr1.pairs.gz"  # Input .pairs.gz file
outdir = "single_cell_pairs"                # Output directory
max_open_files = 1000                       # Limit on simultaneously open files
# -----------------------------

def run():
    start_time = pc()
    print("Splitting contacts by barcode...")
    split_contacts_by_barcode(pairs_file, outdir, max_open_files)
    end_time = pc()
    print('Finished. Time used (secs):', end_time - start_time)

def split_contacts_by_barcode(pairs_file, outdir, max_open_files):
    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Dictionary to store output file paths for each barcode
    outf_dict = {}
    open_files = {}

    # Read barcodes from the pairs file
    barcodes = set()
    with oppf(pairs_file, 'rt') as infile:
        for dline in infile:
            if dline.startswith("#"):
                continue
            fields = dline.strip().split("\t")
            if fields[-1] != fields[-2]:
                continue
            barcode = fields[-1]
            barcodes.add(barcode)
    
    # Create output file paths for each barcode
    for barcode in barcodes:
        outf_dict[barcode] = os.path.join(outdir, f"{barcode}.pairs")

    # Write header to all output files
    print("Writing headers...")
    header_lines = []
    with oppf(pairs_file, 'rt') as infile:
        for dline in infile:
            if dline.startswith("#"):
                header_lines.append(dline)
            else:
                break
    header = "".join(header_lines)
    for ofile in outf_dict.values():
        with open(ofile, "w") as p:
            p.write(header)
    print("Finished writing headers.")

    # Split contacts by barcode
    with oppf(pairs_file, 'rt') as infile:
        for dline in infile:
            if dline.startswith("#"):
                continue
            fields = dline.strip().split("\t")
            if fields[-1] != fields[-2]:
                continue
            barcode = fields[-1]
            fields[-2:] = [barcode, barcode]
            wline = '\t'.join(fields)
            if barcode not in open_files:
                open_files[barcode] = open(outf_dict[barcode], "a")
            open_files[barcode].write(wline + "\n")

            if len(open_files) >= max_open_files:
                for f in open_files.values():
                    f.close()
                open_files.clear()

    # Close remaining files
    for f in open_files.values():
        f.close()

def oppf(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

# Run the script
run()
