#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICE normalization for .cool single cell files  (chr1 only)

Each .cool file → ICE-normalized contact list (bin1, bin2, contact)
Output: ice_txt/barcode.txt (no header)
"""

import os
import h5py
import numpy as np


# -------------------- ICE NORMALIZATION --------------------

class HiCNorm:
    """Hi-C ICE normalization (iterative correction)."""
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float32)
        self.bias = np.ones(self.matrix.shape[0], dtype=np.float32)

    def iterative_correction(self, max_iter=100):
        mat = np.copy(self.matrix)
        x_idx, y_idx = np.nonzero(mat)
        mat_sum = np.sum(mat)

        for _ in range(max_iter):
            bias = np.sum(mat, axis=1)
            bias_mean = np.mean(bias[bias > 0])
            bias = bias / bias_mean
            bias[bias == 0] = 1

            mat[x_idx, y_idx] = mat[x_idx, y_idx] / (bias[x_idx] * bias[y_idx])
            new_sum = np.sum(mat)
            mat *= (mat_sum / new_sum)
            self.bias *= bias * np.sqrt(new_sum / mat_sum)

        norm_matrix = np.copy(self.matrix)
        norm_matrix[x_idx, y_idx] = norm_matrix[x_idx, y_idx] / (
            self.bias[x_idx] * self.bias[y_idx]
        )
        return norm_matrix


# -------------------- LOAD CONTACT MATRIX --------------------

def load_contact_matrix(cool_file_path):
    """Load full contact matrix from a .cool file (chr1 only)."""
    with h5py.File(cool_file_path, 'r') as f:
        bins = f['bins']
        n_bins = len(bins['chrom'][:])
        pixels = f['pixels']
        bin1 = pixels['bin1_id'][:]
        bin2 = pixels['bin2_id'][:]
        counts = pixels['count'][:]

        matrix = np.zeros((n_bins, n_bins), dtype=np.float32)
        matrix[bin1, bin2] = counts
        matrix = matrix + matrix.T - np.diag(matrix.diagonal())
        return matrix


# -------------------- MAIN LOOP --------------------

def normalize_all_cool_files(input_dir, output_root):
    """ICE-normalize all .cool files and save results as txt (no header)."""
    output_dir = os.path.join(output_root, "ice_txt")
    os.makedirs(output_dir, exist_ok=True)

    cool_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".cool")]
    total = len(cool_files)
    print(f"Found {total} .cool files in {input_dir}")

    processed = set(os.path.splitext(f)[0] for f in os.listdir(output_dir))
    cool_files = [f for f in cool_files if os.path.splitext(os.path.basename(f))[0] not in processed]
    print(f"{len(cool_files)} files left to process")

    for idx, file_path in enumerate(cool_files, start=1):
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n[{idx}/{len(cool_files)}] Processing {sample_name}...")

        try:
            mat = load_contact_matrix(file_path)
            ice = HiCNorm(mat)
            ice_norm = ice.iterative_correction(max_iter=100)

            bin1, bin2 = np.nonzero(ice_norm)
            contacts = ice_norm[bin1, bin2]
            data = np.column_stack((bin1, bin2, contacts))
            out_path = os.path.join(output_dir, f"{sample_name}.txt")

            np.savetxt(out_path, data, fmt="%.0f\t%.0f\t%.10f", delimiter="\t")
            print(f"✔ Saved: {out_path}")

            del mat, ice_norm, ice

        except Exception as e:
            print(f"❌ Error processing {sample_name}: {e}")

    print("\n✅ ICE normalization complete!")


# -------------------- RUN --------------------

if __name__ == "__main__":
    normalize_all_cool_files(
        input_dir="/cool_accurate_single_cell",
        output_root="/normalized_contacts"
    )
