
"""
Created on Tue Oct  7 16:00:17 2025

@author: mozhganoroujlu
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
import pandas as pd
from scKTLD import callTLD, metrics_similarity

# Specify your input and output directories
dir_input = "/raw_txt/"
output_tads_dir = "/scKTLD_TADs"

# Create output_tads directory if it doesn't exist
os.makedirs(output_tads_dir, exist_ok=True)

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Determine the number of bins (maximum bin ID + 1, assuming 0-based or 1-based IDs)
def get_max_bin_id(directory):
    max_bin_id = 0
    for cell in os.listdir(directory):
        path = os.path.join(directory, cell)
        if path.endswith(".txt"):  # Ensure only .txt files are processed
            data = np.loadtxt(path, dtype=float)  # Read edge-list data
            max_bin_id = max(max_bin_id, int(np.max(data[:, 0])), int(np.max(data[:, 1])))
    return max_bin_id + 1  # Add 1 to account for 0-based or 1-based indexing

num_bins = get_max_bin_id(dir_input)
num_cells = len([f for f in os.listdir(dir_input) if f.endswith(".txt")])  # Count .txt files
print(f"Number of bins: {num_bins}, Number of cells: {num_cells}")

# Initialize arrays for clustering and domain lists
arr_cluster = np.zeros((num_cells, num_bins))  # Cluster assignments for each cell
list_domain = []  # List of domain boundaries

# Process each cell's data and save TAD boundaries
m = 0
for cell in os.listdir(dir_input):
    if cell.endswith(".txt"):  # Process only .txt files
        path_input = os.path.join(dir_input, cell)
        # Read edge-list data
        edge_data = np.loadtxt(path_input, dtype=float)
        
        # Convert edge-list to adjacency matrix
        graph_adj = np.zeros((num_bins, num_bins))
        for row in edge_data:
            i, j, contact = int(row[0]), int(row[1]), row[2]
            graph_adj[i, j] = contact
            graph_adj[j, i] = contact  # Ensure symmetry (Hi-C matrices are typically symmetric)
        
        # Call TAD-like domains
        boundary_spec = callTLD(graph_adj, dimension=16)
        cluster_temp = np.repeat(np.arange(0, len(boundary_spec)-1), np.diff(boundary_spec))
        cluster_temp = np.append(cluster_temp, cluster_temp[-1])  # Pad to match num_bins
        if len(cluster_temp) < num_bins:
            cluster_temp = np.pad(cluster_temp, (0, num_bins - len(cluster_temp)), mode='edge')
        arr_cluster[m, :] = cluster_temp[:num_bins]  # Truncate if too long
        list_domain.append(np.vstack((boundary_spec[:-1], boundary_spec[1:])).transpose())
        
        # Save TAD boundaries to a .txt file
        output_file = os.path.join(output_tads_dir, f"{os.path.splitext(cell)[0]}_tads.txt")
        np.savetxt(output_file, list_domain[-1], fmt='%d', header='start_bin\tend_bin', comments='')
        print(f"Saved TAD boundaries: {output_file}")
        m += 1

# Calculate similarity matrices
simi_moc = np.zeros((num_cells, num_cells))
for i in range(num_cells):
    for j in range(i + 1):
        simi_moc[i, j] = metrics_similarity.moc(list_domain[i], list_domain[j])
simi_moc = simi_moc + simi_moc.T - np.diag(np.diag(simi_moc))  # Make symmetric

simi_ami = np.zeros((num_cells, num_cells))
for i in range(num_cells):
    for j in range(i + 1):
        simi_ami[i, j] = metrics_similarity.ami(arr_cluster[i], arr_cluster[j])
simi_ami = simi_ami + simi_ami.T - np.diag(np.diag(simi_ami))  # Make symmetric

# Save similarity matrices to Excel files in output_tads_dir
try:
    simi_moc_file = os.path.join(output_tads_dir, "simi_moc.xlsx")
    pd.DataFrame(simi_moc).to_excel(simi_moc_file, index=False, header=False)
    print(f"Saved similarity matrix: {simi_moc_file}")
except Exception as e:
    print(f"Error saving simi_moc.xlsx: {e}")

try:
    simi_ami_file = os.path.join(output_tads_dir, "simi_ami.xlsx")
    pd.DataFrame(simi_ami).to_excel(simi_ami_file, index=False, header=False)
    print(f"Saved similarity matrix: {simi_ami_file}")
except Exception as e:
    print(f"Error saving simi_ami.xlsx: {e}")

# MDS embedding and visualization
fig = plt.figure(figsize=(13.0/2.54, 5.0/2.54), constrained_layout=False, dpi=150)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'  # Changed to avoid font warnings
plt.rcParams['font.size'] = 8
plt.rcParams['axes.xmargin'] = 0.01
plt.rcParams['savefig.pad_inches'] = 0.01
plt.rcParams['savefig.bbox'] = 'tight'

# MoC + MDS
mds = sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed')
embedding = mds.fit_transform(1 - simi_moc)
plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
plt.title("MoC + MDS")
plt.scatter(embedding[:, 0], embedding[:, 1], c='dodgerblue', s=8)
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")

# AMI + MDS
mds = sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed')
embedding = mds.fit_transform(1 - simi_ami)
plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
plt.title("AMI + MDS")
plt.scatter(embedding[:, 0], embedding[:, 1], c='darkorange', s=8)
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")

plt.tight_layout()
plt.show()
