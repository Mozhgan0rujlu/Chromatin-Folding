
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 18:53:05 2025

@author: mozhganoroujlu

Natural Equilibrium Search using Simulated Annealing
Goal: Find stable equilibrium states (basins of attraction) in MDS space
without bias, tracking where systems naturally settle and stay.
"""

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cdist
import warnings
import os
warnings.filterwarnings('ignore')

def create_distance_neighborhoods(glut_df, radius=0.2):
    """
    Create neighborhoods based on MDS distance for natural exploration
    """
    # Extract spatial coordinates
    coordinates = glut_df[['mds_component_1', 'mds_component_2']].values
    
    # Calculate pairwise distances
    print("Calculating distance neighborhoods...")
    
    neighborhoods = {}
    
    for idx in range(len(glut_df)):
        # Calculate distances from current point to all others
        current_coord = coordinates[idx:idx+1]  # Keep as 2D array
        
        # Calculate distances to all points
        distances = cdist(current_coord, coordinates).flatten()
        
        # Find neighbors within radius (excluding self)
        neighbor_mask = (distances > 0) & (distances <= radius)
        neighbor_indices = np.where(neighbor_mask)[0]
        
        neighborhoods[idx] = {
            'indices': neighbor_indices,
            'distances': distances[neighbor_mask],
            'barcode': glut_df.iloc[idx]['barcode']
        }
    
    return neighborhoods, coordinates

def get_adaptive_radius(T, T_start, min_radius=0.05, max_radius=0.3):
    """
    Adaptive neighborhood radius that shrinks with temperature
    """
    if T_start == 0:
        return max_radius
    
    # Linear shrinkage with temperature
    ratio = T / T_start
    radius = min_radius + (max_radius - min_radius) * ratio
    
    # Ensure minimum radius
    return max(radius, min_radius)

def get_natural_neighbor(current_idx, neighborhoods, glut_df, T, T_start):
    """
    Get neighbor using distance-based neighborhood for natural exploration
    """
    current_cell = glut_df.iloc[current_idx]
    
    # Get adaptive radius for current temperature
    current_radius = get_adaptive_radius(T, T_start)
    
    # Get all possible neighbors within current radius
    if 'distances' in neighborhoods[current_idx]:
        within_radius = neighborhoods[current_idx]['distances'] <= current_radius
        available_neighbors = neighborhoods[current_idx]['indices'][within_radius]
    else:
        available_neighbors = neighborhoods[current_idx]['indices']
    
    if len(available_neighbors) == 0:
        # If no neighbors in radius, expand to all neighbors
        available_neighbors = neighborhoods[current_idx]['indices']
    
    if len(available_neighbors) == 0:
        return None
    
    # Random selection for natural exploration
    new_idx = np.random.choice(available_neighbors)
    return glut_df.iloc[new_idx]

def check_stability(current_idx, neighborhoods, glut_df, T_min, n_checks=100):
    """
    Check if current state is stable by attempting moves at T_min
    Returns stability score (fraction of moves rejected)
    """
    if n_checks == 0:
        return 1.0  # Perfectly stable if no checks
    
    stable_count = 0
    current_cell = glut_df.iloc[current_idx]
    current_energy = current_cell['energy']
    
    for _ in range(n_checks):
        # Get a random neighbor
        neighbor_indices = neighborhoods[current_idx]['indices']
        if len(neighbor_indices) == 0:
            stable_count += 1  # No neighbors, so stable
            continue
        
        new_idx = np.random.choice(neighbor_indices)
        new_cell = glut_df.iloc[new_idx]
        new_energy = new_cell['energy']
        
        delta_E = new_energy - current_energy
        
        # At T_min, use strict Metropolis criterion
        if delta_E < 0:
            # Would move downhill - not stable
            pass  # stable_count doesn't increment
        elif T_min > 0:
            # Small chance to move uphill even at T_min
            prob = math.exp(-delta_E / T_min)
            if random.random() < prob:
                # Would move uphill - not stable
                pass
            else:
                stable_count += 1
        else:  # T_min = 0
            # At zero temperature, only accept downhill moves
            stable_count += 1  # Didn't move
    
    stability = stable_count / n_checks
    return stability

def analyze_equilibria(equilibrium_data, glut_df, 
                      stability_threshold=0.8, 
                      min_frequency=3,
                      energy_threshold=None):
    """
    Identify which barcodes are true equilibria
    """
    # Group by final barcode
    barcode_data = {}
    
    for run in equilibrium_data:
        barcode = run['final_barcode']
        if barcode not in barcode_data:
            barcode_data[barcode] = []
        barcode_data[barcode].append(run)
    
    # If no energy threshold provided, use bottom 10% of energies
    if energy_threshold is None:
        energy_threshold = glut_df['energy'].quantile(0.1)
    
    # Find true equilibria
    true_equilibria = []
    
    for barcode, runs in barcode_data.items():
        # Criteria 1: Appears as final state in multiple runs
        frequency = len(runs)
        if frequency < min_frequency:
            continue
        
        # Get cell info
        cell_info = glut_df[glut_df['barcode'] == barcode]
        if len(cell_info) == 0:
            continue
        cell_info = cell_info.iloc[0]
        
        # Criteria 2: Has high average stability
        stabilities = [r['stability_score'] for r in runs]
        avg_stability = np.mean(stabilities)
        if avg_stability < stability_threshold:
            continue
        
        # Criteria 3: Low energy (below threshold)
        if cell_info['energy'] > energy_threshold:
            continue
        
        # Calculate additional statistics
        energies = [r['final_energy'] for r in runs]
        avg_energy = np.mean(energies)
        
        # Get cell types from runs
        celltypes = Counter([r['final_celltype'] for r in runs])
        main_celltype = celltypes.most_common(1)[0][0]
        
        # Calculate dwell ratio (how often visited relative to total steps)
        total_steps = sum([r.get('total_steps', 10000) for r in runs])
        visit_counts = sum([r.get('visit_count', 1) for r in runs])
        dwell_ratio = visit_counts / total_steps if total_steps > 0 else 0
        
        true_equilibria.append({
            'barcode': barcode,
            'frequency': frequency,
            'avg_stability': avg_stability,
            'avg_energy': avg_energy,
            'celltype': main_celltype,
            'energy': cell_info['energy'],
            'mds_1': cell_info['mds_component_1'],
            'mds_2': cell_info['mds_component_2'],
            'dwell_ratio': dwell_ratio,
            'min_energy': min(energies),
            'max_energy': max(energies),
            'std_energy': np.std(energies),
            'stability_std': np.std(stabilities),
            'num_runs': len(runs)
        })
    
    # Sort by frequency (descending) and energy (ascending)
    true_equilibria.sort(key=lambda x: (-x['frequency'], x['avg_energy']))
    
    return true_equilibria

def plot_trajectory(run_idx, trajectory_data, glut_df, output_dir):
    """
    Plot trajectory in 2D MDS space for a single run
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create color map for cell types
    unique_celltypes = glut_df['celltype'].unique()
    celltype_to_color = {}
    cmap = plt.cm.tab20  # Using tab20 colormap for distinct colors
    for i, celltype in enumerate(unique_celltypes):
        celltype_to_color[celltype] = cmap(i % 20)
    
    # Plot all barcodes in background with cell type colors
    for celltype in unique_celltypes:
        celltype_data = glut_df[glut_df['celltype'] == celltype]
        if len(celltype_data) > 0:
            ax.scatter(celltype_data['mds_component_1'], celltype_data['mds_component_2'], 
                      c=[celltype_to_color[celltype]], s=20, alpha=0.2, 
                      label=f'{celltype}' if celltype in trajectory_data['visited_celltypes'] else None)
    
    # Extract trajectory information
    visited_states = trajectory_data['visited_states']
    accepted_states = trajectory_data['accepted_states']
    visited_celltypes = trajectory_data['visited_celltypes']
    
    # Get coordinates for all visited states
    visited_coords = []
    for barcode in visited_states:
        state_info = glut_df[glut_df['barcode'] == barcode]
        if len(state_info) > 0:
            visited_coords.append((state_info.iloc[0]['mds_component_1'], 
                                  state_info.iloc[0]['mds_component_2']))
    
    # Draw path connecting accepted states
    accepted_path_x = []
    accepted_path_y = []
    
    for i, (barcode, accepted) in enumerate(zip(visited_states, accepted_states)):
        state_info = glut_df[glut_df['barcode'] == barcode]
        if len(state_info) == 0:
            continue
        
        state_info = state_info.iloc[0]
        x, y = state_info['mds_component_1'], state_info['mds_component_2']
        
        if accepted:
            # Add to path for connecting lines
            accepted_path_x.append(x)
            accepted_path_y.append(y)
            
            # Plot accepted states with color based on cell type
            celltype = state_info['celltype']
            if celltype in celltype_to_color:
                ax.scatter(x, y, c=[celltype_to_color[celltype]], s=80, alpha=0.8, 
                          edgecolors='black', linewidth=1.5, zorder=5)
    
    # Draw lines connecting accepted states (the path)
    if len(accepted_path_x) > 1:
        ax.plot(accepted_path_x, accepted_path_y, 'k-', alpha=0.5, linewidth=1, zorder=3)
    
    # Mark first and final barcode with special markers
    if visited_states:
        # First state
        first_state = glut_df[glut_df['barcode'] == visited_states[0]]
        if len(first_state) > 0:
            first_state = first_state.iloc[0]
            ax.scatter(first_state['mds_component_1'], first_state['mds_component_2'], 
                      c='lime', s=300, marker='s', edgecolors='black', 
                      linewidth=3, zorder=6, label='Start')
        
        # Final state
        final_state = glut_df[glut_df['barcode'] == visited_states[-1]]
        if len(final_state) > 0:
            final_state = final_state.iloc[0]
            ax.scatter(final_state['mds_component_1'], final_state['mds_component_2'], 
                      c='red', s=300, marker='*', edgecolors='black', 
                      linewidth=3, zorder=6, label='End')
    
    # Add legend for cell types in trajectory only
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Add trajectory legend
    trajectory_legend = ax.legend(unique_handles, unique_labels, loc='upper left', 
                                 fontsize=9, title='Cell Types in Trajectory')
    ax.add_artist(trajectory_legend)
    
    # Add path legend
    path_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lime', 
                  markeredgecolor='black', markersize=10, markeredgewidth=2),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                  markeredgecolor='black', markersize=10, markeredgewidth=2),
        plt.Line2D([0], [0], color='k', linewidth=2, alpha=0.5)
    ]
    path_labels = ['Start', 'End', 'Trajectory Path']
    ax.legend(path_handles, path_labels, loc='upper right', fontsize=9)
    
    ax.set_xlabel('MDS Component 1', fontsize=12)
    ax.set_ylabel('MDS Component 2', fontsize=12)
    ax.set_title(f'Trajectory for Run {run_idx + 1}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectory_run_{run_idx + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_equilibrium_search(
    tsv_file,
    detailed_results_file='/Users/mozhganoroujlu/Desktop/SA/detailed_results.xlsx',
    acceptance_rates_file='/Users/mozhganoroujlu/Desktop/SA/acceptance_rates.xlsx',
    acceptance_rate_details_file='/Users/mozhganoroujlu/Desktop/SA/acceptance_rate_details.xlsx',
    final_summary_file='/Users/mozhganoroujlu/Desktop/SA/final_summary.xlsx',
    equilibrium_file='/Users/mozhganoroujlu/Desktop/SA/equilibrium_states.xlsx',
    output_dir='/Users/mozhganoroujlu/Desktop/SA/deepseek/',
    num_opc_cells=30,
    runs_per_cell=10,
    neighborhood_radius=0.15,
    T_start=1.0,
    T_min=0.001,
    alpha=0.98,
    outer_iterations=200,
    inner_iterations=50,
    stability_checks=100
):
    """
    Natural equilibrium search using simulated annealing
    Goal: Find stable states where systems naturally settle
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Filter to Non-neuron category
    glut_df = df[df['category'] == 'Non-neuron'].reset_index(drop=True)
    
    # Create distance-based neighborhoods
    print("Creating neighborhoods...")
    neighborhoods, coordinates = create_distance_neighborhoods(glut_df, radius=neighborhood_radius)
    
    # Get OPC cells for initial states
    opc_cells = glut_df[glut_df['celltype'] == 'OPC']
    if len(opc_cells) < num_opc_cells:
        print(f"Warning: Only {len(opc_cells)} OPC cells available, using all")
        num_opc_cells = len(opc_cells)
    
    # Sample unique initial OPC cells
    initial_opc_cells = opc_cells.sample(min(num_opc_cells, len(opc_cells)))
    
    # Data structures for output
    detailed_results_data = {}
    acceptance_rates_data = {}
    acceptance_details_data = {}
    equilibrium_data = []
    all_final_results = []
    
    # Store trajectory data for first 10 runs
    trajectory_plots_data = []
    
    # Generate temperature schedule once
    Ts = []
    T = T_start
    for j in range(outer_iterations):
        Ts.append(T)
        T = T * alpha
        if T <= T_min:
            break
    
    total_runs = len(initial_opc_cells) * runs_per_cell
    current_run = 0
    
    # Run for each initial OPC cell
    for initial_idx, initial_cell in initial_opc_cells.iterrows():
        initial_barcode = initial_cell['barcode']
        
        # Initialize data structures for this initial state
        detailed_results_data[initial_barcode] = []
        acceptance_rates_data[initial_barcode] = []
        acceptance_details_data[initial_barcode] = []
        
        # Run multiple times from the same initial state
        for run_num in range(runs_per_cell):
            current_run += 1
            print(f"Run {current_run}/{total_runs}: Initial {initial_barcode[:10]}..., Run {run_num + 1}/{runs_per_cell}")
            
            # Set initial state
            current_idx = glut_df[glut_df['barcode'] == initial_barcode].index[0]
            current_cell = glut_df.iloc[current_idx]
            current_energy = current_cell['energy']
            current_celltype = current_cell['celltype']
            current_barcode = current_cell['barcode']
            
            # Reset temperature
            T = T_start
            
            # Tracking for this run
            state_counter = Counter()  # Track visits to each state
            acceptance_history = []
            steps_at_current_state = 0
            last_state = current_barcode
            
            # Store trajectory for analysis and plotting
            trajectory = []
            visited_states = [current_barcode]
            accepted_states = [True]  # Initial state is "accepted"
            visited_celltypes = set([current_celltype])
            
            # For acceptance rate by temperature
            acceptance_by_T = {t: {'acceptances': 0, 'total': 0} for t in Ts}
            
            for j in range(outer_iterations):
                accept_count = 0
                current_T = Ts[j] if j < len(Ts) else T_min
                
                for i in range(inner_iterations):
                    # Get natural neighbor
                    new_cell = get_natural_neighbor(current_idx, neighborhoods, glut_df, T, T_start)
                    
                    if new_cell is not None:
                        new_energy = new_cell['energy']
                        new_celltype = new_cell['celltype']
                        new_barcode = new_cell['barcode']
                        new_idx = new_cell.name
                        
                        # Calculate delta_E
                        delta_E = new_energy - current_energy
                        
                        # Metropolis criterion
                        accepted = False
                        if delta_E < 0:
                            accepted = True
                        elif T > 0:
                            prob = math.exp(-delta_E / T)
                            if random.random() < prob:
                                accepted = True
                        
                        # Track acceptance by temperature
                        acceptance_by_T[current_T]['total'] += 1
                        if accepted:
                            acceptance_by_T[current_T]['acceptances'] += 1
                        
                        # Update if accepted
                        if accepted:
                            # Record transition
                            trajectory.append({
                                'step': j * inner_iterations + i,
                                'from_barcode': current_barcode,
                                'to_barcode': new_barcode,
                                'from_energy': current_energy,
                                'to_energy': new_energy,
                                'T': T
                            })
                            
                            # Update current state
                            current_cell = new_cell
                            current_energy = new_energy
                            current_celltype = new_celltype
                            current_barcode = new_barcode
                            current_idx = new_idx
                            accept_count += 1
                            
                            # Track state visits
                            state_counter[current_barcode] += 1
                            
                            # Track dwell time
                            if current_barcode == last_state:
                                steps_at_current_state += 1
                            else:
                                last_state = current_barcode
                                steps_at_current_state = 1
                        
                        # Record visited state for plotting
                        visited_states.append(new_barcode)
                        accepted_states.append(accepted)
                        visited_celltypes.add(new_celltype)
                    else:
                        visited_states.append(current_barcode)
                        accepted_states.append(False)
                
                # Calculate acceptance rate for this temperature
                acceptance_rate = accept_count / inner_iterations if inner_iterations > 0 else 0
                acceptance_history.append(acceptance_rate)
                
                # Cool temperature
                T = T * alpha
                if T <= T_min:
                    T = T_min
                    break
            
            # After annealing, check stability of final state
            stability = check_stability(current_idx, neighborhoods, glut_df, T_min, n_checks=stability_checks)
            
            # Calculate visit statistics
            total_steps = outer_iterations * inner_iterations
            visit_count = state_counter.get(current_barcode, 1)
            dwell_ratio = visit_count / total_steps if total_steps > 0 else 0
            
            # Store detailed results for this run
            detailed_results_data[initial_barcode].append({
                'run_num': run_num + 1,
                'final_state_barcode': current_barcode,
                'final_state_energy': current_energy,
                'final_state_cell_type': current_celltype,
                'stability_score': stability,
                'visit_count': visit_count,
                'dwell_ratio': dwell_ratio,
                'trajectory_length': len(trajectory)
            })
            
            # Store acceptance rates
            avg_acceptance = np.mean(acceptance_history) if acceptance_history else 0
            acceptance_rates_data[initial_barcode].append({
                'run_num': run_num + 1,
                'average_acceptance_rate': avg_acceptance,
                'final_acceptance_rate': acceptance_history[-1] if acceptance_history else 0
            })
            
            # Store acceptance details for this barcode
            acceptance_details_data[initial_barcode].append({
                'run_num': run_num + 1,
                'T_values': Ts[:len(acceptance_history)],
                'acceptance_rates': acceptance_history
            })
            
            # Store equilibrium data
            equilibrium_data.append({
                'initial_barcode': initial_barcode,
                'final_barcode': current_barcode,
                'final_energy': current_energy,
                'final_celltype': current_celltype,
                'stability_score': stability,
                'visit_count': visit_count,
                'dwell_ratio': dwell_ratio,
                'total_steps': total_steps,
                'avg_acceptance_rate': avg_acceptance
            })
            
            # Store final result for summary
            all_final_results.append({
                'initial_state_barcode': initial_barcode,
                'run_num': run_num + 1,
                'final_state_barcode': current_barcode,
                'final_state_cell_type': current_celltype,
                'final_state_energy': current_energy,
                'stability_score': stability
            })
            
            # Store trajectory data for first 10 runs
            if current_run <= 10:
                trajectory_plots_data.append({
                    'run_idx': current_run - 1,
                    'initial_barcode': initial_barcode,
                    'visited_states': visited_states,
                    'accepted_states': accepted_states,
                    'visited_celltypes': list(visited_celltypes)
                })
    
    # Create trajectory plots for first 10 runs
    print("\nCreating trajectory plots for first 10 runs...")
    for trajectory_data in trajectory_plots_data:
        plot_trajectory(trajectory_data['run_idx'], trajectory_data, glut_df, output_dir)
    
    # Create acceptance rate by temperature plot
    create_acceptance_rate_plot(acceptance_details_data, Ts, output_dir)
    
    # Create output files
    print("\nCreating output files...")
    
    # 1. Detailed results
    with pd.ExcelWriter(detailed_results_file) as writer:
        for initial_barcode, run_results in detailed_results_data.items():
            sheet_name = f"Init_{initial_barcode[:20]}"[:31]
            df_detailed = pd.DataFrame(run_results)
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 2. Acceptance rates
    with pd.ExcelWriter(acceptance_rates_file) as writer:
        for initial_barcode, rate_results in acceptance_rates_data.items():
            sheet_name = f"Init_{initial_barcode[:20]}"[:31]
            df_rates = pd.DataFrame(rate_results)
            df_rates.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 3. Acceptance rate details (NEW)
    with pd.ExcelWriter(acceptance_rate_details_file) as writer:
        for initial_barcode, run_details in acceptance_details_data.items():
            sheet_name = f"Init_{initial_barcode[:20]}"[:31]
            
            # Create DataFrame with 2 columns per run
            data_dict = {}
            for run_detail in run_details:
                run_num = run_detail['run_num']
                data_dict[f'Run_{run_num}_T'] = run_detail['T_values']
                data_dict[f'Run_{run_num}_acceptance_rate'] = run_detail['acceptance_rates']
            
            # Pad lists to same length
            max_len = max(len(lst) for lst in data_dict.values())
            for key in data_dict:
                data_dict[key] = data_dict[key] + [None] * (max_len - len(data_dict[key]))
            
            df_details = pd.DataFrame(data_dict)
            df_details.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 4. Final summary with new sheet
    df_all_final = pd.DataFrame(all_final_results)
    
    # Count final barcodes
    barcode_counts = df_all_final.groupby('final_state_barcode').agg(
        cell_type=('final_state_cell_type', 'first'),
        final_energy=('final_state_energy', 'first'),
        count=('run_num', 'size')
    ).reset_index()
    barcode_counts.columns = ['barcode', 'cell_type', 'final_energy', 'number_counted_as_final_state']
    
    # Sort by count
    barcode_counts = barcode_counts.sort_values('number_counted_as_final_state', ascending=False)
    
    cell_type_counts = df_all_final['final_state_cell_type'].value_counts().reset_index()
    cell_type_counts.columns = ['cell_type', 'counted_final_state']
    
    with pd.ExcelWriter(final_summary_file) as writer:
        # New sheet with final barcode counts
        barcode_counts.to_excel(writer, sheet_name='final_barcode_counts', index=False)
        
        # Summary sheet
        cell_type_counts.to_excel(writer, sheet_name='final_result', index=False)
        
        # All runs sheet
        df_all_final.to_excel(writer, sheet_name='all_runs', index=False)
        
        # Statistics sheet
        stats_df = pd.DataFrame({
            'statistic': ['Total Runs', 'Unique Final States', 'Average Stability', 'Average Final Energy'],
            'value': [
                len(all_final_results),
                df_all_final['final_state_barcode'].nunique(),
                df_all_final['stability_score'].mean(),
                df_all_final['final_state_energy'].mean()
            ]
        })
        stats_df.to_excel(writer, sheet_name='statistics', index=False)
    
    # 5. Equilibrium analysis
    true_equilibria = analyze_equilibria(equilibrium_data, glut_df)
    
    # Create equilibrium results file
    if true_equilibria:
        df_equilibria = pd.DataFrame(true_equilibria)
        
        # Add overall statistics
        equilibrium_stats = pd.DataFrame({
            'parameter': ['Total Equilibrium States', 'Average Frequency', 
                         'Average Stability', 'Average Energy',
                         'Most Common Cell Type'],
            'value': [
                len(true_equilibria),
                df_equilibria['frequency'].mean() if len(true_equilibria) > 0 else 0,
                df_equilibria['avg_stability'].mean() if len(true_equilibria) > 0 else 0,
                df_equilibria['avg_energy'].mean() if len(true_equilibria) > 0 else 0,
                df_equilibria['celltype'].mode()[0] if len(true_equilibria) > 0 else 'None'
            ]
        })
        
        with pd.ExcelWriter(equilibrium_file) as writer:
            df_equilibria.to_excel(writer, sheet_name='equilibrium_states', index=False)
            equilibrium_stats.to_excel(writer, sheet_name='statistics', index=False)
            
            # Add energy distribution of equilibria
            energy_stats = df_equilibria['energy'].describe()
            energy_stats.to_excel(writer, sheet_name='energy_distribution')
    else:
        print("No equilibrium states found with current criteria")
        # Create empty file with message
        with pd.ExcelWriter(equilibrium_file) as writer:
            pd.DataFrame({'message': ['No equilibrium states found with current criteria']}).to_excel(writer, index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Detailed results: {detailed_results_file}")
    print(f"Acceptance rates: {acceptance_rates_file}")
    print(f"Acceptance rate details: {acceptance_rate_details_file}")
    print(f"Final summary: {final_summary_file}")
    print(f"Equilibrium states: {equilibrium_file}")
    print(f"Found {len(true_equilibria)} equilibrium states")
    print(f"Trajectory plots saved in: {output_dir}")
    
    if true_equilibria:
        print("\nTop 10 Equilibrium States:")
        for i, eq in enumerate(true_equilibria[:10]):
            print(f"{i+1}. Barcode: {eq['barcode'][:15]}..., "
                  f"Energy: {eq['energy']:.4f}, "
                  f"Celltype: {eq['celltype']}, "
                  f"Frequency: {eq['frequency']}, "
                  f"Stability: {eq['avg_stability']:.2f}")
    
    return {
        'true_equilibria': true_equilibria,
        'all_final_results': all_final_results,
        'equilibrium_data': equilibrium_data,
        'neighborhoods': neighborhoods,
        'coordinates': coordinates,
        'glut_df': glut_df,
        'acceptance_details_data': acceptance_details_data
    }

def create_acceptance_rate_plot(acceptance_details_data, Ts, output_dir):
    """
    Create line graph for acceptance rate vs temperature
    """
    # Collect all acceptance rates by temperature
    acceptance_by_T = {}
    
    for initial_barcode, run_details in acceptance_details_data.items():
        for run_detail in run_details:
            T_values = run_detail['T_values']
            acceptance_rates = run_detail['acceptance_rates']
            
            for T, rate in zip(T_values, acceptance_rates):
                if T not in acceptance_by_T:
                    acceptance_by_T[T] = []
                acceptance_by_T[T].append(rate)
    
    # Calculate average acceptance rate for each temperature
    avg_acceptance = {}
    for T, rates in acceptance_by_T.items():
        if rates:
            avg_acceptance[T] = np.mean(rates)
    
    # Sort temperatures
    sorted_Ts = sorted(avg_acceptance.keys())
    sorted_rates = [avg_acceptance[T] for T in sorted_Ts]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_Ts, sorted_rates, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Average Acceptance Rate', fontsize=12)
    ax.set_title('Average Acceptance Rate vs Temperature', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Log scale for temperature if it spans multiple orders of magnitude
    if max(sorted_Ts) / min(sorted_Ts) > 100:
        ax.set_xscale('log')
        ax.set_xlabel('Temperature (T) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/acceptance_rate_vs_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_results(results, output_dir='/Users/mozhganoroujlu/Desktop/SA/deepseek/'):
    """
    Create visualization plots for the results
    """
    glut_df = results['glut_df']
    true_equilibria = results['true_equilibria']
    all_final_results = results['all_final_results']
    
    # Convert to DataFrame for easier handling
    final_df = pd.DataFrame(all_final_results)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Energy landscape with cell type colors
    unique_celltypes = glut_df['celltype'].unique()
    celltype_to_color = {}
    cmap = plt.cm.tab20
    for i, celltype in enumerate(unique_celltypes):
        celltype_to_color[celltype] = cmap(i % 20)
    
    for celltype in unique_celltypes:
        celltype_data = glut_df[glut_df['celltype'] == celltype]
        if len(celltype_data) > 0:
            sc = axes[0, 0].scatter(celltype_data['mds_component_1'], celltype_data['mds_component_2'],
                                   c=[celltype_to_color[celltype]], s=20, alpha=0.6, label=celltype)
    axes[0, 0].set_title('Cell Types in MDS Space', fontsize=12)
    axes[0, 0].set_xlabel('MDS Component 1')
    axes[0, 0].set_ylabel('MDS Component 2')
    # Add legend
    axes[0, 0].legend(loc='upper left', fontsize=8, ncol=2)
    
    # Plot 2: Final states frequency
    if len(final_df) > 0:
        # Count frequency of each final state
        freq_counts = final_df['final_state_barcode'].value_counts()
        
        # Create size array for scatter plot
        sizes = np.array([freq_counts.get(b, 1) * 50 for b in glut_df['barcode']])
        
        # Color by cell type
        colors = [celltype_to_color[ct] for ct in glut_df['celltype']]
        
        axes[0, 1].scatter(glut_df['mds_component_1'], glut_df['mds_component_2'],
                          c=colors, s=sizes, alpha=0.5)
        
        # Highlight true equilibria
        if true_equilibria:
            eq_barcodes = [eq['barcode'] for eq in true_equilibria]
            eq_df = glut_df[glut_df['barcode'].isin(eq_barcodes)]
            
            # Color by stability
            stability_map = {eq['barcode']: eq['avg_stability'] for eq in true_equilibria}
            eq_stabilities = [stability_map.get(b, 0.5) for b in eq_df['barcode']]
            
            sc2 = axes[0, 1].scatter(eq_df['mds_component_1'], eq_df['mds_component_2'],
                                    c=eq_stabilities, cmap='RdYlGn', s=200, alpha=0.8, 
                                    edgecolors='black', linewidth=1.5)
            plt.colorbar(sc2, ax=axes[0, 1], label='Stability Score')
        
        axes[0, 1].set_title('Final States Frequency\n(size = frequency, colored = cell type)', fontsize=12)
        axes[0, 1].set_xlabel('MDS Component 1')
        axes[0, 1].set_ylabel('MDS Component 2')
    
    # Plot 3: Cell type distribution
    celltype_counts = final_df['final_state_cell_type'].value_counts()
    # Color bars by cell type
    bar_colors = [celltype_to_color[ct] for ct in celltype_counts.index]
    axes[0, 2].bar(celltype_counts.index, celltype_counts.values, color=bar_colors)
    axes[0, 2].set_title('Final State Cell Type Distribution', fontsize=12)
    axes[0, 2].set_xlabel('Cell Type')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Energy distribution of final states
    axes[1, 0].hist(final_df['final_state_energy'], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(final_df['final_state_energy'].mean(), color='red', 
                      linestyle='dashed', linewidth=2, label=f'Mean: {final_df["final_state_energy"].mean():.3f}')
    axes[1, 0].set_title('Energy Distribution of Final States', fontsize=12)
    axes[1, 0].set_xlabel('Energy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 5: Stability distribution
    axes[1, 1].hist(final_df['stability_score'], bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].axvline(final_df['stability_score'].mean(), color='red', 
                      linestyle='dashed', linewidth=2, label=f'Mean: {final_df["stability_score"].mean():.3f}')
    axes[1, 1].set_title('Stability Score Distribution', fontsize=12)
    axes[1, 1].set_xlabel('Stability Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # Plot 6: Energy vs Stability
    if len(true_equilibria) > 0:
        eq_df = pd.DataFrame(true_equilibria)
        # Color by cell type
        eq_colors = [celltype_to_color[ct] for ct in eq_df['celltype']]
        scatter = axes[1, 2].scatter(eq_df['avg_energy'], eq_df['avg_stability'],
                                     c=eq_colors, s=100, alpha=0.7, edgecolors='black')
        axes[1, 2].set_title('Equilibrium States: Energy vs Stability\n(color = cell type)', fontsize=12)
        axes[1, 2].set_xlabel('Average Energy')
        axes[1, 2].set_ylabel('Average Stability')
        # Add legend for cell types
        handles = []
        labels = []
        for celltype in eq_df['celltype'].unique():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=celltype_to_color[celltype], markersize=8))
            labels.append(celltype)
        axes[1, 2].legend(handles, labels, loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/equilibrium_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional figure for equilibrium locations
    if true_equilibria:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Background: all cells colored by cell type
        for celltype in unique_celltypes:
            celltype_data = glut_df[glut_df['celltype'] == celltype]
            if len(celltype_data) > 0:
                ax2.scatter(celltype_data['mds_component_1'], celltype_data['mds_component_2'],
                           c=[celltype_to_color[celltype]], s=10, alpha=0.2, label=celltype)
        
        # Plot equilibrium states
        eq_df = pd.DataFrame(true_equilibria)
        
        # Size by frequency, color by stability
        sizes = 50 + eq_df['frequency'] * 20
        scatter = ax2.scatter(eq_df['mds_1'], eq_df['mds_2'],
                             c=eq_df['avg_stability'], cmap='RdYlGn',
                             s=sizes, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Add labels for top equilibria
        for i, row in eq_df.head(5).iterrows():
            ax2.annotate(f"{i+1}", (row['mds_1'], row['mds_2']),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=10, fontweight='bold')
        
        ax2.set_title('Equilibrium States in MDS Space', fontsize=14)
        ax2.set_xlabel('MDS Component 1')
        ax2.set_ylabel('MDS Component 2')
        
        # Add colorbars
        plt.colorbar(scatter, ax=ax2, label='Stability (circles)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equilibrium_locations.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to run the equilibrium search
    """
    # Parameters for natural equilibrium search
    params = {
        'tsv_file': '/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/non_neuron_filtered_celltypes.tsv',
        'num_opc_cells': 356,           # Number of starting OPC cells
        'runs_per_cell': 10,           # Runs per starting cell
        'neighborhood_radius': 0.15,    # Max distance for natural exploration
        'T_start': 0.15,               # Start temperature (higher for more exploration)
        'T_min': 0.00001,             # Minimum temperature (freezing point)
        'alpha': 0.98,                # Cooling rate (slower = better annealing)
        'outer_iterations': 200,      # Annealing steps
        'inner_iterations': 50,       # Steps per temperature
        'stability_checks': 100,      # Stability verification steps
        'output_dir': '/Users/mozhganoroujlu/Desktop/SA/deepseek/'
    }
    
    print("=" * 70)
    print("NATURAL EQUILIBRIUM SEARCH")
    print("Finding stable basins of attraction in MDS space")
    print("=" * 70)
    
    print("\nParameters:")
    for key, value in params.items():
        if key not in ['tsv_file', 'output_dir']:
            print(f"  {key}: {value}")
    
    # Run the equilibrium search
    results = run_equilibrium_search(**params)
    
    # Visualize results
    visualize_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    
    true_equilibria = results['true_equilibria']
    if true_equilibria:
        print(f"\nFound {len(true_equilibria)} equilibrium states")
        print("\nTop 5 Most Stable Equilibria:")
        print("-" * 100)
        print(f"{'Rank':<5} {'Barcode':<20} {'Energy':<10} {'Celltype':<15} {'Frequency':<12} {'Stability':<10} {'Location (MDS)'}")
        print("-" * 100)
        
        for i, eq in enumerate(true_equilibria[:5]):
            print(f"{i+1:<5} {eq['barcode'][:18]:<20} "
                  f"{eq['energy']:<10.4f} {eq['celltype']:<15} "
                  f"{eq['frequency']:<12} {eq['avg_stability']:<10.2f} "
                  f"({eq['mds_1']:.3f}, {eq['mds_2']:.3f})")
        
        # Calculate some statistics
        energies = [eq['energy'] for eq in true_equilibria]
        stabilities = [eq['avg_stability'] for eq in true_equilibria]
        frequencies = [eq['frequency'] for eq in true_equilibria]
        
        print("\nStatistics:")
        print(f"  Lowest energy equilibrium: {min(energies):.4f}")
        print(f"  Highest stability: {max(stabilities):.2f}")
        print(f"  Most frequent equilibrium: {max(frequencies)} runs")
        print(f"  Average stability: {np.mean(stabilities):.2f}")
        
        # Check if equilibria are in the low-energy basin you described
        low_energy_eq = [eq for eq in true_equilibria 
                        if eq['energy'] <= -0.8 and 
                        -0.5 <= eq['mds_2'] <= -0.2 and
                        -0.1 <= eq['mds_1'] <= 0.3]
        
        if low_energy_eq:
            print(f"\nFound {len(low_energy_eq)} equilibria in your described low-energy basin")
            for eq in low_energy_eq[:3]:
                print(f"  - {eq['barcode'][:15]}...: Energy={eq['energy']:.4f}, "
                      f"Stability={eq['avg_stability']:.2f}")
        else:
            print("\nNo equilibria found in your described low-energy basin")
            print("This suggests either:")
            print("  1. The basin is too narrow to find via natural exploration")
            print("  2. The equilibria are in different locations")
            print("  3. Need more runs or different parameters")
    else:
        print("\nNo equilibrium states found with current criteria.")
        print("Consider adjusting parameters:")
        print("  - Increase num_opc_cells or runs_per_cell")
        print("  - Increase neighborhood_radius")
        print("  - Use higher T_start for more exploration")
        print("  - Lower stability_threshold in analyze_equilibria()")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()