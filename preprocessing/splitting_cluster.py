#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mozhganoroujlu
"""

"""
This script splits a Hi-C  .pairs.gz file into 20 separate cluster-specific
.pairs files using a barcode-to-cluster mapping table. Each output file preserves
the original header and contains only contacts belonging to a single cluster.
"""

import os
import numpy as np
import pandas as pd
from time import perf_counter as pc
import gzip

# -----------------------------
# SET YOUR INPUT FILES HERE:
pairs_file = "merged_716_462_chr1.pairs.gz"  # Input .pairs.gz file
cluster_file = "output.tsv"                   # Two columns: barcode, cluster
outdir = "output_umap_folder"                             # Output directory
max_open_files = 1000                                # Limit on simultaneously open files
# -----------------------------

def run():
    start_time = pc()
    print("Splitting contacts by cluster...")
    split_contacts(pairs_file, outdir, cluster_file, max_open_files)
    end_time = pc()
    print('Finished. Time used (secs):', end_time - start_time)

def split_contacts(pairs_file, outPrefix, clusterf, max_open_files):
    # Read cluster table with only barcode and cluster columns
    clust_dat = pd.read_csv(clusterf, sep="\t", names=["barcode", "cluster"], header=0)
    clust_dat_dict = pd.Series(clust_dat["cluster"].values, index=clust_dat["barcode"]).to_dict()
    outf_dict = dict()
    open_files = {}

    os.makedirs(outPrefix, exist_ok=True)

    for cls_id in pd.unique(clust_dat["cluster"]):
        outf_dict[cls_id] = os.path.join(outPrefix, f"{cls_id}.pairs")

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

    # Split contacts
    with oppf(pairs_file, 'rt') as infile:
        for dline in infile:
            if dline.startswith("#"):
                continue
            fields = dline.strip().split("\t")
            if fields[-1] != fields[-2]:
                continue
            barcode = fields[-1]
            if barcode in clust_dat_dict:
                fields[-2:] = [barcode, barcode]
                wline = '\t'.join(fields)
                cls = clust_dat_dict[barcode]
                if cls not in open_files:
                    open_files[cls] = open(outf_dict[cls], "a")
                open_files[cls].write(wline + "\n")

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


run()
