
"""
Created on Tue Nov 11 18:21:34 2025
@author: mozhganoroujlu
"""
import pandas as pd
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors

def create_spatial_neighborhoods(glut_df, k_neighbors=15):
    """
    Create k-NN graph based on spatial coordinates
    """
    # Extract spatial coordinates
    coordinates = glut_df[['mds_component_1', 'mds_component_2']].values
   
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
   
    # Create neighborhood dictionary (excluding self)
    neighborhoods = {}
    for idx, (neighbor_indices, distance) in enumerate(zip(indices, distances)):
        # Skip the first neighbor (itself)
        neighborhoods[idx] = {
            'indices': neighbor_indices[1:], # Exclude self
            'distances': distance[1:],
            'barcode': glut_df.iloc[idx]['barcode']
        }
   
    return neighborhoods, coordinates

def get_balanced_neighbor(current_cell_index, neighborhoods, glut_df, spatial_weight=0.7, energy_weight=0.3):
    """
    Get neighbor considering both spatial proximity and energy similarity
    """
    current_cell = glut_df.iloc[current_cell_index]
    current_energy = current_cell['energy']
   
    neighbor_indices = neighborhoods[current_cell_index]['indices']
   
    if len(neighbor_indices) == 0:
        return None
   
    # Calculate scores for each neighbor
    scores = []
    for neighbor_idx in neighbor_indices:
        neighbor = glut_df.iloc[neighbor_idx]
       
        # Spatial score (inverse of distance - closer is better)
        current_coords = current_cell[['mds_component_1', 'mds_component_2']].values
        neighbor_coords = neighbor[['mds_component_1', 'mds_component_2']].values
        spatial_distance = np.linalg.norm(current_coords - neighbor_coords)
        spatial_score = 1.0 / (1.0 + spatial_distance)
       
        # Energy similarity score (closer energy is better)
        energy_diff = abs(neighbor['energy'] - current_energy)
        energy_score = 1.0 / (1.0 + energy_diff)
       
        # Combined score
        total_score = (spatial_weight * spatial_score) + (energy_weight * energy_score)
        scores.append(total_score)
   
    # Select neighbor with probability proportional to scores
    if sum(scores) > 0:
        probabilities = np.array(scores) / np.sum(scores)
        selected_idx = np.random.choice(neighbor_indices, p=probabilities)
        return glut_df.iloc[selected_idx]
    else:
        # Fallback: random selection
        selected_idx = random.choice(neighbor_indices)
        return glut_df.iloc[selected_idx]

def run_algorithm(tsv_file, 
                  detailed_results_file='/Users/mozhganoroujlu/Desktop/SA/detailed_results.xlsx',
                  acceptance_rates_file='/Users/mozhganoroujlu/Desktop/SA/acceptance_rates.xlsx',
                  final_summary_file='/Users/mozhganoroujlu/Desktop/SA/final_summary.xlsx',
                  num_opc_cells=356, runs_per_cell=10, k_neighbors=15, spatial_weight=0.7, energy_weight=0.3):
    """
    Runs the improved simulated annealing algorithm with spatial neighborhoods.
    - Filters for Non-neuron category.
    - Uses spatial coordinates for neighborhood definition.
    - Runs multiple times from different random OPC cells.
    - Creates the requested output files.
    """
    # Load data
    df = pd.read_csv(tsv_file, sep='\t')
   
    # Filter to Non-neuron category
    glut_df = df[df['category'] == 'Non-neuron'].reset_index(drop=True)
   
    # Precompute spatial neighborhoods
    print("Creating spatial neighborhoods...")
    neighborhoods, coordinates = create_spatial_neighborhoods(glut_df, k_neighbors=k_neighbors)
   
    # Get OPC cells for initial states
    opc_cells = glut_df[glut_df['celltype'] == 'OPC']
    if len(opc_cells) < num_opc_cells:
        raise ValueError(f"Not enough OPC cells ({len(opc_cells)}) for {num_opc_cells} initial states.")
   
    # Sample unique initial OPC cells
    initial_opc_cells = opc_cells.sample(num_opc_cells)
   
    # Parameters
    T_start = 0.2
    T_min = 0.000001
    alpha = 0.98
    outer_iterations = 200
    inner_iterations = 50
   
    # Data structures for output files
    detailed_results_data = {}  # Key: initial_state_index, Value: list of run results
    acceptance_rates_data = {}  # Key: initial_state_index, Value: list of acceptance rates
    all_final_results = []      # List of all final results for summary
   
    # Generate Ts once (same for all runs)
    Ts = []
    T = T_start
    for j in range(outer_iterations):
        Ts.append(T)
        T = T * alpha
        if T <= T_min:
            break
   
    total_runs = num_opc_cells * runs_per_cell
    current_run = 0
   
    # Run for each initial OPC cell
    for initial_idx, initial_cell in initial_opc_cells.iterrows():
        initial_state_index = initial_idx
        initial_barcode = initial_cell['barcode']
       
        # Initialize data structures for this initial state
        detailed_results_data[initial_state_index] = []
        acceptance_rates_data[initial_state_index] = []
       
        # Run multiple times from the same initial state
        for run_num in range(runs_per_cell):
            current_run += 1
            print(f"Running initial state {list(initial_opc_cells.index).index(initial_idx) + 1}/{num_opc_cells}, run {run_num + 1}/{runs_per_cell} (Total: {current_run}/{total_runs})")
           
            # Set initial state for this run
            current_idx = initial_idx
            current_cell = glut_df.iloc[current_idx]
            current_energy = current_cell['energy']
            current_celltype = current_cell['celltype']
            current_barcode = current_cell['barcode']
           
            # Reset T for run
            T = T_start
           
            # Acceptance rates for this run
            acceptance_rates = []
           
            for j in range(outer_iterations):
                accept_count = 0
               
                for i in range(1, inner_iterations + 1):
                    # Propose a new cell from spatial neighborhood
                    new_cell = get_balanced_neighbor(current_idx, neighborhoods, glut_df,
                                                   spatial_weight, energy_weight)
                   
                    if new_cell is not None:
                        new_energy = new_cell['energy']
                        new_celltype = new_cell['celltype']
                        new_barcode = new_cell['barcode']
                        new_idx = new_cell.name
                       
                        # Calculate delta_E
                        delta_E = new_energy - current_energy
                       
                        # Decide acceptance
                        accepted = False
                        if delta_E < 0:
                            accepted = True
                        else:
                            if T > 0:
                                prob = math.exp(-delta_E / T)
                                if random.random() < prob:
                                    accepted = True
                       
                        # Update if accepted
                        if accepted:
                            current_cell = new_cell
                            current_energy = new_energy
                            current_celltype = new_celltype
                            current_barcode = new_barcode
                            current_idx = new_idx
                            accept_count += 1
               
                # Calculate acceptance rate for this T
                acceptance_rate = accept_count / inner_iterations
                acceptance_rates.append(acceptance_rate)
               
                # Cool T
                T = T * alpha
                if T <= T_min:
                    break
           
            # Calculate average acceptance rate for this run
            avg_acceptance = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0
           
            # Store detailed results for this run
            detailed_results_data[initial_state_index].append({
                'run_num': run_num + 1,
                'final_state_barcode': current_barcode,
                'final_state_energy': current_energy,
                'final_state_cell_type': current_celltype
            })
           
            # Store acceptance rate for this run
            acceptance_rates_data[initial_state_index].append({
                'run_num': run_num + 1,
                'average_acceptance_rate': avg_acceptance
            })
           
            # Store final result for summary
            all_final_results.append({
                'initial_state_barcode': initial_barcode,
                'run_num': run_num + 1,
                'final_state_barcode': current_barcode,
                'final_state_cell_type': current_celltype,
                'final_state_energy': current_energy
            })
   
    # Create detailed_results Excel file
    print("Creating detailed_results Excel file...")
    with pd.ExcelWriter(detailed_results_file) as writer:
        for initial_idx, run_results in detailed_results_data.items():
            initial_barcode = glut_df.loc[initial_idx, 'barcode']
            sheet_name = f"Initial_{initial_barcode}"[:31]  # Excel sheet name limit
            df_detailed = pd.DataFrame(run_results)
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
   
    # Create acceptance_rates Excel file
    print("Creating acceptance_rates Excel file...")
    with pd.ExcelWriter(acceptance_rates_file) as writer:
        for initial_idx, rate_results in acceptance_rates_data.items():
            initial_barcode = glut_df.loc[initial_idx, 'barcode']
            sheet_name = f"Initial_{initial_barcode}"[:31]  # Excel sheet name limit
            df_rates = pd.DataFrame(rate_results)
            df_rates.to_excel(writer, sheet_name=sheet_name, index=False)
   
    # Create final_summary Excel file
    print("Creating final_summary Excel file...")
    df_all_final = pd.DataFrame(all_final_results)
    cell_type_counts = df_all_final['final_state_cell_type'].value_counts().reset_index()
    cell_type_counts.columns = ['cell_type', 'counted_final_state']
    
    with pd.ExcelWriter(final_summary_file) as writer:
        cell_type_counts.to_excel(writer, sheet_name='final_result', index=False)
   
    print(f"Detailed results saved to {detailed_results_file}")
    print(f"Acceptance rates saved to {acceptance_rates_file}")
    print(f"Final summary saved to {final_summary_file}")
    print(f"Total runs completed: {total_runs}")
   
    return cell_type_counts, neighborhoods, coordinates

# Example usage:
if __name__ == "__main__":
    final_summary, neighborhoods, coordinates = run_algorithm(
        '/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/non_neuron_filtered_celltypes.tsv',
        num_opc_cells=20,
        runs_per_cell=2,
        k_neighbors=150,
        spatial_weight=0.5,
        energy_weight=0.5
    )