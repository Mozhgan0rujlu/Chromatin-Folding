
"""
Created on Wed Jul 16 20:00:42 2025

@author: mozhganoroujlu
"""
import gzip
from tqdm import tqdm

def filter_hic_by_barcode1_and_chr1(pairs_gz_path, barcodes_gz_path, output_gz_path):
    """
    Filter a .sc.pairs.gz file to extract rows where:
    1. barcode1 is in the barcode list from a .txt.gz file.
    2. chrom1 equals 'chr1'.
    The filtered lines are saved in a new .gz file.
    Displays progress of unique barcodes matched during filtering.

    Parameters:
    - pairs_gz_path (str): Path to the input .sc.pairs.gz file
    - barcodes_gz_path (str): Path to the input .txt.gz file with barcodes
    - output_gz_path (str): Path to the output .gz file

    Raises:
    FileNotFoundError: If input files do not exist.
    gzip.BadGzipFile: If the input files are not valid gzip files.
    """
    try:
        # Load all barcodes from the .txt.gz file into a set for fast lookup
        barcode_set = set()
        with gzip.open(barcodes_gz_path, 'rt', encoding='utf-8') as barcode_file:
            for line in barcode_file:
                barcode = line.strip()
                if barcode:
                    barcode_set.add(barcode)
        total_barcodes = len(barcode_set)
        print(f"Loaded {total_barcodes} barcodes from {barcodes_gz_path}.")

        # Track matched barcodes
        matched_barcodes = set()

        # Open the input .gz file and output .gz file
        with gzip.open(pairs_gz_path, 'rt', encoding='utf-8') as infile, \
             gzip.open(output_gz_path, 'wt', encoding='utf-8') as outfile:
            
            filtered_count = 0
            processed_count = 0
            
            # Use tqdm for progress bar
            for line in tqdm(infile, desc="Processing pairs file", unit=" lines"):
                processed_count += 1
                
                if line.startswith("#") or line.strip() == "":
                    # Write comment/header lines directly
                    outfile.write(line)
                    continue

                fields = line.strip().split('\t')  # Tab-separated based on provided format
                if len(fields) < 10:
                    continue  # Skip malformed lines

                chrom1 = fields[1]
                barcode1 = fields[8]

                # Check both conditions: barcode1 match and chrom1 == 'chr1'
                if barcode1 in barcode_set and chrom1 == "chr1":
                    outfile.write(line)
                    filtered_count += 1
                    matched_barcodes.add(barcode1)
                    # Update progress on unique barcodes matched
                    if len(matched_barcodes) % 100 == 0:  # Update every 100 unique barcodes
                        print(f"Matched {len(matched_barcodes)}/{total_barcodes} unique barcodes so far.")

            print(f"Processed {processed_count} lines, filtered {filtered_count} lines.")
            print(f"Matched {len(matched_barcodes)}/{total_barcodes} unique barcodes in total.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except gzip.BadGzipFile as e:
        raise gzip.BadGzipFile(f"File is not a valid gzip file: {str(e)}")

# Usage
filter_hic_by_barcode1_and_chr1(
    pairs_gz_path="GSM8020244_LC462_mm10.sc.pairs.gz",
    barcodes_gz_path="GSM8020244_LC462_valid_bc.txt.gz",
    output_gz_path="462_filtered_chr1_barcodes.pairs.gz"
)
