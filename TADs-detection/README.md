
TADs_detection.py identifies TAD-like domains in single-cell Hi-C matrices using scKTLD.
For each cell, it converts normalized edge-list contacts into a symmetric
adjacency matrix, detects domain boundaries, saves the TAD intervals, computes
similarity matrices (MoC and AMI) across cells, and visualizes TADs similarities in low dimesnsional space using MDS embeddings.


scKTLD pipeline uses 3columns .txt files as an input for contacts. as we worked with .cool files before, 3columns_contacts converts .cool file into .txt file.
