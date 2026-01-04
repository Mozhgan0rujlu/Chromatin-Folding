
higashi_sc_compartment_scores.py

This script computes compartment (A/B) scores from single-cell Hi-C data using **Higashi** framework
processed with bandnorm normalization. It performs the following steps:

1. Loads per-cell Hi-C contact matrices (.txt) and CpG ratios per bin.
2. Builds dense contact matrices and computes decay-normalized correlation matrices.
3. Performs PCA on bulk (cell-type aggregated) contact matrices to define compartments.
4. Orients compartments using Top 10% vs Bottom 10% CpG ratios.
5. Projects each single cell onto the bulk PCA, applies quantile normalization, and saves scores.
6. Generates multiple visualizations:
   - Aggregated contact matrices with bulk compartment scores.
   - OPC bulk vs single-cell comparison.
   - Quantile-normalized compartment score distributions per cell type.
   - Correlation line plots of single cells vs bulk per cell type.
   - Single-cell correlation heatmaps per cell type.


