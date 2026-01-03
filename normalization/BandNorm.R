

# This script performs bandNorm normalization of single-chromosome Hi-C contact
# matrices (chr1) across multiple cells. It computes mean interaction depth per
# genomic distance (band) and outputs normalized contact values for each cell in
# a simplified three-column format: binA, binB, BandNorm.


library(data.table)
library(dplyr)

# Modified bandnorm function to handle single chromosome and custom output format
bandnorm_modified <- function(path, save_path) {
  if (!dir.exists(path)) {
    stop("Input path for data doesn't exist!")
  }
  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
  }
  
  # Get paths and names for all cells
  paths <- list.files(path, pattern = "\\.txt$", recursive = TRUE, full.names = TRUE)
  names <- basename(paths)
  
  # Load all data
  load_cell <- function(i) {
    return(fread(paths[i]) %>% 
             rename(chrom = V1, binA = V2, chrom2 = V3, binB = V4, count = V5) %>%
             filter(chrom == "chr1" & chrom2 == "chr1") %>%  # Ensure intra-chr1
             mutate(diag = abs(binB - binA), cell = names[i]))
  }
  hic_df <- rbindlist(lapply(seq_along(paths), load_cell))
  
  # Calculate band depth and mean band depth
  band_info <- hic_df %>% group_by(chrom, diag, cell) %>% summarise(band_depth = sum(count))
  alpha_j <- band_info %>% group_by(chrom, diag) %>% summarise(depth = mean(band_depth))
  
  # Normalize
  hic_df <- hic_df %>% 
    left_join(alpha_j, by = c("chrom", "diag")) %>% 
    left_join(band_info, by = c("chrom", "diag", "cell")) %>% 
    mutate(BandNorm = count / band_depth * depth) %>% 
    select(-c(band_depth, depth, count, chrom, chrom2))  # Drop unnecessary columns
  
  # Save each normalized cell as 3-column .txt: binA, binB, BandNorm
  for (cell_name in unique(hic_df$cell)) {
    cell_data <- hic_df[cell == cell_name, .(binA, binB, BandNorm)]
    write.table(cell_data, file = file.path(save_path, cell_name), 
                col.names = FALSE, row.names = FALSE, quote = FALSE, sep = "\t")
  }
  
  return(hic_df)
}

# Usage: Set your paths
raw_txt_dir <- "/raw_txt"  # contact matrices in .txt format
norm_dir <- "/normalized_txt"  # Where to save normalized .txt files

# Run the function
bandnorm_result <- bandnorm_modified(path = raw_txt_dir, save_path = norm_dir)
