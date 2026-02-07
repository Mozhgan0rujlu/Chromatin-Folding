#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:09:44 2026

@author: mozhganoroujlu
"""

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

def run_equilibrium_search(
    tsv_file,
    detailed_results_file='/Users/mozhganoroujlu/Desktop/SA/detailed_results.xlsx',
    acceptance_rates_file='/Users/mozhganoroujlu/Desktop/SA/acceptance_rates.xlsx',
    final_summary_file='/Users/mozhganoroujlu/Desktop/SA/final_summary.xlsx',
    equilibrium_file='/Users/mozhganoroujlu/Desktop/SA/equilibrium_states.xlsx',
    num_opc_cells=30,
    runs_per_cell=10,
    neighborhood_radius=0.2,
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
    equilibrium_data = []
    all_final_results = []
    
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
            
            # Store trajectory for analysis
            trajectory = []
            
            for j in range(outer_iterations):
                accept_count = 0
                
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
                
                # Calculate acceptance rate for this temperature
                acceptance_rate = accept_count / inner_iterations
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
            dwell_ratio = visit_count / total_steps
            
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
    
    # 3. Final summary
    df_all_final = pd.DataFrame(all_final_results)
    cell_type_counts = df_all_final['final_state_cell_type'].value_counts().reset_index()
    cell_type_counts.columns = ['cell_type', 'counted_final_state']
    
    with pd.ExcelWriter(final_summary_file) as writer:
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
    
    # 4. Equilibrium analysis
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
    print(f"Final summary: {final_summary_file}")
    print(f"Equilibrium states: {equilibrium_file}")
    print(f"Found {len(true_equilibria)} equilibrium states")
    
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
        'glut_df': glut_df
    }

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
    
    # Plot 1: Energy landscape
    sc1 = axes[0, 0].scatter(glut_df['mds_component_1'], glut_df['mds_component_2'],
                            c=glut_df['energy'], cmap='viridis_r', s=20, alpha=0.6)
    axes[0, 0].set_title('Energy Landscape (darker = lower energy)', fontsize=12)
    axes[0, 0].set_xlabel('MDS Component 1')
    axes[0, 0].set_ylabel('MDS Component 2')
    plt.colorbar(sc1, ax=axes[0, 0])
    
    # Plot 2: Final states frequency
    if len(final_df) > 0:
        # Count frequency of each final state
        freq_counts = final_df['final_state_barcode'].value_counts()
        
        # Create size array for scatter plot
        sizes = np.array([freq_counts.get(b, 1) * 50 for b in glut_df['barcode']])
        
        axes[0, 1].scatter(glut_df['mds_component_1'], glut_df['mds_component_2'],
                          c='gray', s=sizes, alpha=0.3)
        
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
        
        axes[0, 1].set_title('Final States Frequency\n(size = frequency, colored = equilibria)', fontsize=12)
        axes[0, 1].set_xlabel('MDS Component 1')
        axes[0, 1].set_ylabel('MDS Component 2')
    
    # Plot 3: Cell type distribution
    celltype_counts = final_df['final_state_cell_type'].value_counts()
    axes[0, 2].bar(celltype_counts.index, celltype_counts.values)
    axes[0, 2].set_title('Final State Cell Type Distribution', fontsize=12)
    axes[0, 2].set_xlabel('Cell Type')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Energy distribution of final states
    axes[1, 0].hist(final_df['final_state_energy'], bins=30, alpha=0.7, edgecolor='black')
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
        scatter = axes[1, 2].scatter(eq_df['avg_energy'], eq_df['avg_stability'],
                                     c=eq_df['frequency'], cmap='viridis', s=100, alpha=0.7)
        axes[1, 2].set_title('Equilibrium States: Energy vs Stability\n(color = frequency)', fontsize=12)
        axes[1, 2].set_xlabel('Average Energy')
        axes[1, 2].set_ylabel('Average Stability')
        plt.colorbar(scatter, ax=axes[1, 2], label='Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/equilibrium_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional figure for equilibrium locations
    if true_equilibria:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Background: all cells colored by energy
        background = ax2.scatter(glut_df['mds_component_1'], glut_df['mds_component_2'],
                                c=glut_df['energy'], cmap='viridis_r', s=10, alpha=0.3)
        
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
        plt.colorbar(background, ax=ax2, label='Energy (background)')
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
        'num_opc_cells': 10,           # Number of starting OPC cells
        'runs_per_cell': 5,           # Runs per starting cell
        'neighborhood_radius': 0.2,    # Max distance for natural exploration
        'T_start': 0.2,               # Start temperature (higher for more exploration)
        'T_min': 0.00001,               # Minimum temperature (freezing point)
        'alpha': 0.98,                # Cooling rate (slower = better annealing)
        'outer_iterations': 200,      # Annealing steps
        'inner_iterations': 50,       # Steps per temperature
        'stability_checks': 100       # Stability verification steps
    }
    
    print("=" * 70)
    print("NATURAL EQUILIBRIUM SEARCH")
    print("Finding stable basins of attraction in MDS space")
    print("=" * 70)
    
    print("\nParameters:")
    for key, value in params.items():
        if key != 'tsv_file':
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