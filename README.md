# Chromatin Folding and Its Impact on Cell Fate

## Overview

This repository contains the code used in the project **“Chromatin Folding and Its Impact on Cell Fate”**, which investigates how 3D genome organization influences energy-state of non-neuron cells and cell differentiation from stem cells to mature cell types.

Using single-cell droplet Hi-C data from mouse (**mm10 genome**), we analyze chromatin folding patterns and leverage them to create an **energy landscape** and predict cell differentiation trajectories. The repository includes scripts for preprocessing, normalization, TAD detection, compartment score calculation, and chromatin folding simulations.

---

## Data

- **Technology:** Droplet Hi-C  
- **Organism:** Mouse (*mm10*)  
- **Paper:** Droplet Hi-C enables scalable, single-cell profiling of chromatin architecture in heterogeneous tissues ([DOI: 10.1038/s41587-024-02447-1](https://www.nature.com/articles/s41587-024-02447-1))
- **Data source:** GEO  
- **GEO accession number:** [GSE253407](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253407) 

Raw data are not included in this repository. Please download them directly from GEO and place them in the appropriate data directory.

---

## Analysis Modules

The repository is organized around the main analytical steps of the project.

### Preprocessing
- Valid barcode filtering  
- Conversion to contact matrices  
- Generation of `.cool` files 
- Separation of single-cell or cell-type contacts from aggregated `.pairs.gz` files  

### Single Cell Hi-C Normalization
- Band-wise normalization to correct genomic distance biases  
- Implemented using [**BandNorm**](https://github.com/keleslab/BandNorm)

### Single Cell Hi-C TAD Detection
- Identification of topologically associating domains (TADs)  
- Performed using [**scKTLD**](https://github.com/lhqxinghun/scKTLD)

### Single Cell Hi-C Compartment Analysis
- Computation of A/B compartment scores  
- Implemented using the [**Higashi**](https://github.com/ma-compbio/Higashi) framework

### Simulations and Energy Landscape
- Computation of chromatin energy states  
- Dimensionality reduction using MDS and diffusion maps  
- Construction of AMI and MOC similarity matrices  
- Simulation of cell fate decision-making using Monte Carlo methods  

   

---

## Software and Dependencies

### Core Packages

The following packages are required to run the analysis:

- **scKTLD** – TAD detection in single-cell Hi-C  
- **BandNorm** – Hi-C normalization  
- **Higashi** (optional) – Single-cell Hi-C embedding and compartment score inference  

### Python Libraries

- numpy  
- pandas  
- scipy  
- scikit-learn  
- h5py  
- cooler  
 

---

## Usage

Each analysis step can be run independently. A typical workflow follows:

**Preprocessing → Normalization → TAD detection → Compartment analysis → Simulations**

