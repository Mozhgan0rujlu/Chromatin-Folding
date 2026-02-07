
"""
Created on Tue Nov 11 18:53:05 2025
@author: mozhganoroujlu
"""
import pandas as pd
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def create_spatial_neighborhoods(glut_df, k_neighbors=20):
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
    neighbor_distances = neighborhoods[current_cell_index]['distances']
  
    if len(neighbor_indices) == 0:
        return None
  
    # Calculate scores for each neighbor
    scores = []
    for i, neighbor_idx in enumerate(neighbor_indices):
        neighbor = glut_df.iloc[neighbor_idx]
      
        # Spatial score (inverse of distance - closer is better)
        spatial_distance = neighbor_distances[i]
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

def plot_spatial_trajectory(original_df, trajectory_data, run_number=0):
    """
    Plot the spatial trajectory of one run with path
    """
    plt.figure(figsize=(14, 10))
  
    # Plot all cells colored by cell type
    cell_types = original_df['celltype'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))
  
    for i, cell_type in enumerate(cell_types):
        mask = original_df['celltype'] == cell_type
        plt.scatter(original_df.loc[mask, 'mds_component_1'],
                   original_df.loc[mask, 'mds_component_2'],
                   c=[colors[i]], label=cell_type, alpha=0.3, s=20)
  
    # Extract trajectory coordinates
    trajectory_coords = []
    trajectory_types = []
  
    for step in trajectory_data:
        barcode = step['barcode']
        cell = original_df[original_df['barcode'] == barcode].iloc[0]
        trajectory_coords.append([cell['mds_component_1'], cell['mds_component_2']])
        trajectory_types.append(cell['celltype'])
  
    trajectory_coords = np.array(trajectory_coords)
  
    # Plot the trajectory path
    plt.plot(trajectory_coords[:, 0], trajectory_coords[:, 1],
             'k-', alpha=0.5, linewidth=1, label='Trajectory Path')
  
    # Plot trajectory points with color coding by cell type
    for i, (x, y) in enumerate(trajectory_coords):
        color_idx = list(cell_types).index(trajectory_types[i])
        plt.scatter(x, y, c=[colors[color_idx]], s=30, alpha=0.7)
  
    # Highlight start and end points
    if len(trajectory_coords) > 0:
        start_x, start_y = trajectory_coords[0]
        end_x, end_y = trajectory_coords[-1]
        plt.scatter(start_x, start_y, marker='*', s=200, color='green', label='Start Cell')
        plt.scatter(end_x, end_y, marker='*', s=200, color='red', label='End Cell')
  
    plt.legend()
    end_celltype = trajectory_data[-1]['celltype']
    plt.title(f'Spatial Trajectory - Run {run_number + 1} - End Celltype: {end_celltype}')
    plt.show()

def run_algorithm(tsv_file, output_detailed='/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/non_neuron_sa_outputs/all_ami_non_neuron_monte_carlo_steps_detailedd.xlsx',
                  output_summary='/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/non_neuron_sa_outputs/all_ami_non_neuron_monte_carlo_summaryy.xlsx',
                  num_runs=50, k_neighbors=20, spatial_weight=0.8, energy_weight=0.2):
    """
    Runs the Monte Carlo algorithm with spatial neighborhoods.
    - Filters for Non-neuron category.
    - Uses spatial coordinates for neighborhood definition.
    - Runs multiple times from different random OPC cells.
    - Fixed temperature T=0.04.
    - Runs for 1000 Monte Carlo steps.
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
    if len(opc_cells) < num_runs:
        raise ValueError(f"Not enough OPC cells ({len(opc_cells)}) for {num_runs} runs.")
  
    # Sample unique initial rows
    initial_indices = opc_cells.sample(num_runs).index.tolist()
  
    # Parameters
    T = 0.06
    mc_steps = 12000
  
    # Collect results for all runs
    all_results = []
    all_final = []
  
    with pd.ExcelWriter(output_detailed) as writer:
        for run_idx in range(num_runs):
            # Set initial state for this run
            current_idx = initial_indices[run_idx]
            current_cell = glut_df.iloc[current_idx]
            current_energy = current_cell['energy']
            current_celltype = current_cell['celltype']
            current_barcode = current_cell['barcode']
          
            # Results list for this run
            results = []
          
            # Add initial state row
            results.append({
                'step': 0,
                'barcode': current_barcode,
                'celltype': current_celltype,
                'energy': current_energy,
                'action': 'start',
                'delta_energy': np.nan,
                'acceptance_prob': np.nan
            })
          
            accept_count = 0
          
            for step in range(1, mc_steps + 1):
                # Propose a new cell from spatial neighborhood
                new_cell = get_balanced_neighbor(current_idx, neighborhoods, glut_df,
                                                 spatial_weight, energy_weight)
              
                if new_cell is not None:
                    new_energy = new_cell['energy']
                    new_celltype = new_cell['celltype']
                    new_barcode = new_cell['barcode']
                    new_idx = new_cell.name
                  
                    # Calculate delta_E
                    delta_energy = new_energy - current_energy
                  
                    # Calculate acceptance probability
                    if delta_energy < 0:
                        acceptance_prob = 1.0
                    else:
                        acceptance_prob = math.exp(-delta_energy / T) if T > 0 else 0.0
                  
                    # Decide acceptance
                    accepted = random.random() < acceptance_prob
                    action = 'accept' if accepted else 'reject'
                  
                    if accepted:
                        current_idx = new_idx
                        current_energy = new_energy
                        current_celltype = new_celltype
                        current_barcode = new_barcode
                        accept_count += 1
                else:
                    # No neighbor found, stay in current state
                    delta_energy = np.nan
                    acceptance_prob = np.nan
                    action = 'reject'
              
                # Append current state
                results.append({
                    'step': step,
                    'barcode': current_barcode,
                    'celltype': current_celltype,
                    'energy': current_energy,
                    'action': action,
                    'delta_energy': delta_energy,
                    'acceptance_prob': acceptance_prob
                })
          
            # Save this run's results to sheet
            result_df = pd.DataFrame(results)
            result_df.to_excel(writer, sheet_name=f'Run_{run_idx+1}', index=False)
          
            # Collect final info
            last = results[-1]
            acceptance_rate = accept_count / mc_steps
            unique_barcodes = len(set(r['barcode'] for r in results))
            all_final.append({
                'run_number': run_idx + 1,
                'final_state_barcode': last['barcode'],
                'final_state_energy': last['energy'],
                'final_state_celltype': last['celltype'],
                'acceptance_rate': acceptance_rate,
                'total_steps': mc_steps,
                'unique_barcodes_visited': unique_barcodes
            })
          
            all_results.append(results)
          
            print(f"Completed run {run_idx + 1}/{num_runs}")
          
            # Plot for the first run only
            if run_idx == 0:
                print("Plotting spatial trajectory for run 1...")
                plot_spatial_trajectory(glut_df, results, run_number=run_idx)
  
    print(f"Detailed steps saved to {output_detailed}")
  
    # Save summary
    df_final = pd.DataFrame(all_final)
  
    from collections import Counter
    final_barcodes = [d['final_state_barcode'] for d in all_final]
    counts = Counter(final_barcodes)
    summary_data = []
    for barcode, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        cell = glut_df[glut_df['barcode'] == barcode].iloc[0]
        summary_data.append({
            'barcode': barcode,
            'count_as_final_state': count,
            'celltype': cell['celltype'],
            'energy': cell['energy']
        })
    df_summary = pd.DataFrame(summary_data)
  
    with pd.ExcelWriter(output_summary) as writer:
        df_final.to_excel(writer, sheet_name='Final_States', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f"Summary saved to {output_summary}")
  
    return df_final, neighborhoods, coordinates

# Example usage:
if __name__ == "__main__":
    final_results, neighborhoods, coordinates = run_algorithm(
        '/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/non_neuron_filtered_celltypes.tsv',
        num_runs=50,
        k_neighbors=20,
        spatial_weight=0.8,
        energy_weight=0.2
    )