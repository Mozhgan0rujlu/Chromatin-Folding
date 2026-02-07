"""
this script creats a 3columns hi-c contact .txt file from a dense cool file(for a cell type)
columns are[bin id1, bin id2, contacts between bin id1 and bin id2]
"""
import cooler
import pandas as pd

# Load the .cool file
cool_file = '/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/paper_pipline/cool_accurate_single_cell/AAACGAAAGACCGCAA.cool'
clr = cooler.Cooler(cool_file)  # Load without resolution path

# Check the resolution of the file
resolution = clr.binsize
print(f"File resolution: {resolution} bp")
if resolution != 500000:
    print(f"Warning: File resolution is {resolution} bp, not 500000 bp. Adjust the script if needed.")

# Select chromosome 1 interactions
chrom = 'chr1'
try:
    matrix = clr.matrix(balance=False).fetch(chrom)
except ValueError:
    print(f"Chromosome 'chr1' not found. Available chromosomes: {clr.chromnames}")
    exit()

# Get bin information
bins = clr.bins().fetch(chrom)
bin_count = len(bins)

# Create lists for output
bin1_list = []
bin2_list = []
contacts_list = []

# Iterate through the matrix to get non-zero contacts
for i in range(bin_count):
    for j in range(i, bin_count):  # Upper triangle including diagonal
        contacts = matrix[i, j]
        if contacts > 0:  # Only include non-zero contacts
            bin1_list.append(i + 1)  # 1-based bin index
            bin2_list.append(j + 1)  # 1-based bin index
            contacts_list.append(contacts)

# Create a DataFrame
df = pd.DataFrame({
    'bin1': bin1_list,
    'bin2': bin2_list,
    'contacts': contacts_list
})

# Save to a tab-separated .txt file
output_file = '/Users/mozhganoroujlu/Desktop/AAACGAAAGACCGCAA.txt'
df.to_csv(output_file, sep='\t', index=False, header=False)  # No header for format like "1 1 34"

print(f"Output saved to {output_file}")