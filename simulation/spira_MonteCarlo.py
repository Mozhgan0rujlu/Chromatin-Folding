

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Generate spiral with increasing radial spacing - FIXED VERSION
def generate_pi_spiral(center, max_radius, n_turns=20, points_per_turn=100):
    """
    Generate a spiral with increasing radial spacing between turns
    R_n - R_{n-1} > R_{n-1} - R_{n-2} > ... > R_2 - R_1
    """
    # Total points in the spiral
    n_points = n_turns * points_per_turn
    
    # Generate angles (multiple full turns)
    theta = np.linspace(0, n_turns * 2 * np.pi, n_points)
    
    # Generate turn boundaries
    turn_boundaries = np.arange(0, n_turns + 1)  # 0, 1, 2, ..., n_turns
    
    # Create radii at turn boundaries with increasing differences
    # Start with a base difference and increase it
    base_diff = max_radius / (n_turns * (n_turns + 1) / 2)  # This ensures sum of differences = max_radius
    
    # Create increasing differences: d1, d2, d3, ... with d_n > d_{n-1}
    differences = []
    for i in range(1, n_turns + 1):
        # Make differences increase: use i as multiplier
        differences.append(base_diff * i)
    
    # Adjust to ensure sum equals max_radius
    differences = np.array(differences)
    differences = differences / np.sum(differences) * max_radius
    
    # Calculate cumulative radii at turn boundaries
    turn_radii = np.zeros(n_turns + 1)
    for i in range(1, n_turns + 1):
        turn_radii[i] = turn_radii[i-1] + differences[i-1]
    
    # Now interpolate radii for all points along the spiral
    turns_from_angle = theta / (2 * np.pi)  # Convert angle to turn number
    
    # Interpolate radii for each point based on its turn number
    radii = np.interp(turns_from_angle, turn_boundaries, turn_radii)
    
    # Convert polar to Cartesian coordinates
    x_spiral = center[0] + radii * np.cos(theta)
    y_spiral = center[1] + radii * np.sin(theta)
    
    # Create turn markers (for coloring)
    turn_indices = np.arange(0, n_points, points_per_turn)
    if turn_indices[-1] != n_points - 1:
        turn_indices = np.append(turn_indices, n_points - 1)
    turn_radii_actual = radii[turn_indices]
    
    return x_spiral, y_spiral, theta, radii, turn_indices, turn_radii_actual, differences

# Find which spiral turn each point is closest to
def map_points_to_spiral(points, center, x_spiral, y_spiral, theta, radii):
    """Map each data point to the nearest point on the spiral"""
    # Calculate distances from center for all points
    points_centered = points - center
    points_r = np.sqrt(points_centered[:, 0]**2 + points_centered[:, 1]**2)
    points_theta = np.arctan2(points_centered[:, 1], points_centered[:, 0])
    
    # Normalize angles to [0, 2π]
    points_theta = np.mod(points_theta, 2 * np.pi)
    
    # For each point, find the closest spiral point
    spiral_points = []
    for i in range(len(points)):
        r = points_r[i]
        th = points_theta[i]
        
        # Find spiral point with similar angle
        angle_diff = np.abs(theta - th)
        # Consider periodicity
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        
        # Weighted distance considering both angle and radius
        angle_weight = 2.0  # Weight for angle matching
        radius_weight = 1.0  # Weight for radius matching
        
        # Find best match
        weighted_dist = angle_weight * angle_diff + radius_weight * np.abs(radii - r)
        closest_idx = np.argmin(weighted_dist)
        
        spiral_points.append((x_spiral[closest_idx], y_spiral[closest_idx]))
    
    return np.array(spiral_points)

# Calculate distances between consecutive turns
def calculate_turn_distances(turn_radii):
    """Calculate distances between consecutive spiral turns"""
    distances = []
    for i in range(1, len(turn_radii)):
        distances.append(turn_radii[i] - turn_radii[i-1])
    return distances

# Monte Carlo simulation with correct Metropolis criterion
def monte_carlo_spiral_simulation(df, center_idx, spiral_turn_radii, 
                                  temperature=0.04, n_iterations=1000):
    """
    Perform Monte Carlo simulation starting from center barcode.
    Each barcode is visited exactly once in 918 steps.
    Metropolis criterion:
    - If ΔE < 0: accept
    - Else: accept with probability exp(-ΔE/T)
    """
    # Get center information
    center = df.iloc[center_idx]
    current_barcode = center['barcode']
    current_energy = center['energy']
    current_celltype = center['celltype']
    
    # Calculate distances of all points from center
    center_coords = df[['mds_component_1', 'mds_component_2']].iloc[center_idx].values
    all_coords = df[['mds_component_1', 'mds_component_2']].values
    distances = np.sqrt(np.sum((all_coords - center_coords)**2, axis=1))
    
    # Add distance column to dataframe
    df = df.copy()
    df['distance_from_center'] = distances
    
    # Current state
    current_idx = center_idx
    current_max_radius = spiral_turn_radii[1]  # Start with first radius (R1)
    current_radius_idx = 1
    
    # Tracking
    accepted_states = []
    rejected_states = []
    radius_history = []
    visited_barcodes = set([current_barcode])
    
    print("=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"Starting barcode: {current_barcode}")
    print(f"Starting energy: {current_energy:.4f}")
    print(f"Starting celltype: {current_celltype}")
    print(f"Temperature: {temperature}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Starting exploration radius: {spiral_turn_radii[1]:.4f}")
    print("-" * 60)
    
    # Monte Carlo iterations
    iteration = 0
    while iteration < n_iterations and len(visited_barcodes) < len(df):
        # Get all points within current maximum radius
        candidates = df[df['distance_from_center'] <= current_max_radius]
        
        # Exclude current point and already visited barcodes
        candidates = candidates[~candidates['barcode'].isin(visited_barcodes)]
        
        if len(candidates) == 0:
            # No unvisited candidates in current radius, expand to next radius
            if current_radius_idx < len(spiral_turn_radii) - 1:
                current_radius_idx += 1
                current_max_radius = spiral_turn_radii[current_radius_idx]
                continue
            else:
                # No more radii to expand to, break
                break
        
        # Randomly select a candidate
        candidate = candidates.sample(1)
        candidate_idx = candidate.index[0]
        candidate_energy = candidate['energy'].values[0]
        candidate_barcode = candidate['barcode'].values[0]
        
        # Mark this barcode as visited
        visited_barcodes.add(candidate_barcode)
        
        # Calculate energy difference
        delta_energy = candidate_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_energy < 0:
            # Always accept if energy decreases
            accept = True
            acceptance_prob = 1.0
        else:
            # Accept with probability exp(-ΔE/T)
            acceptance_prob = np.exp(-delta_energy / temperature)
            accept = np.random.random() < acceptance_prob
        
        if accept:
            # Accept move
            current_idx = candidate_idx
            current_energy = candidate_energy
            current_barcode = candidate_barcode
            current_celltype = candidate['celltype'].values[0]
            
            # Update distance for new center
            current_coords = df[['mds_component_1', 'mds_component_2']].iloc[current_idx].values
            distances = np.sqrt(np.sum((all_coords - current_coords)**2, axis=1))
            df['distance_from_center'] = distances
            
            # Reset to first radius for new state
            current_radius_idx = 1
            current_max_radius = spiral_turn_radii[1]
            
            accepted_states.append({
                'iteration': iteration,
                'barcode': current_barcode,
                'energy': current_energy,
                'celltype': current_celltype,
                'radius': current_max_radius,
                'delta_energy': delta_energy,
                'acceptance_prob': acceptance_prob
            })
            
            if iteration % 100 == 0 or iteration < 10:
                print(f"Iteration {iteration+1}: Accepted move to {current_barcode}")
                print(f"  Energy: {current_energy:.4f} (ΔE = {delta_energy:+.4f})")
                print(f"  Celltype: {current_celltype}")
                print(f"  Acceptance probability: {acceptance_prob:.4f}")
                
        else:
            # Reject move - expand search radius
            rejected_states.append({
                'iteration': iteration,
                'barcode': candidate_barcode,
                'energy': candidate_energy,
                'celltype': candidate['celltype'].values[0],
                'delta_energy': delta_energy,
                'acceptance_prob': acceptance_prob
            })
            
            # Expand radius to next turn if available
            if current_radius_idx < len(spiral_turn_radii) - 1:
                current_radius_idx += 1
                current_max_radius = spiral_turn_radii[current_radius_idx]
            
            if iteration % 100 == 0 or iteration < 10:
                print(f"Iteration {iteration+1}: Rejected move to {candidate_barcode}")
                print(f"  ΔE = {delta_energy:+.4f}, Acceptance probability: {acceptance_prob:.4f}")
                print(f"  Expanded radius to: {current_max_radius:.4f}")
        
        radius_history.append(current_max_radius)
        iteration += 1
    
    # Final state
    final_state = df.iloc[current_idx]
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total iterations completed: {iteration}")
    print(f"Unique barcodes visited: {len(visited_barcodes)}")
    print(f"Accepted moves: {len(accepted_states)}")
    print(f"Rejected moves: {len(rejected_states)}")
    if iteration > 0:
        print(f"Acceptance rate: {len(accepted_states)/iteration:.2%}")
    print("-" * 60)
    print("FINAL STATE:")
    print(f"Barcode: {final_state['barcode']}")
    print(f"Cell type: {final_state['celltype']}")
    print(f"Energy: {final_state['energy']:.4f}")
    print(f"Category: {final_state['category']}")
    print(f"MDS Coordinates: ({final_state['mds_component_1']:.4f}, {final_state['mds_component_2']:.4f})")
    print("=" * 60)
    
    return {
        'final_state': final_state,
        'accepted_states': accepted_states,
        'rejected_states': rejected_states,
        'radius_history': radius_history,
        'visited_barcodes': visited_barcodes,
        'acceptance_rate': len(accepted_states)/iteration if iteration > 0 else 0,
        'total_iterations': iteration
    }

# Visualization function with Monte Carlo results
def visualize_spiral_mapping_with_mc(data_points, celltypes, energy_values, 
                                    center_idx=None, barcode=None, mc_results=None):
    """
    Visualize the spiral mapping with Monte Carlo simulation results
    """
    # Set professional style
    available_styles = plt.style.available
    clean_styles = ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot', 'bmh', 'classic']
    for style in clean_styles:
        if style in available_styles:
            plt.style.use(style)
            break
    
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Choose random center if not specified
    if center_idx is None:
        center_idx = random.randint(0, len(data_points) - 1)
    
    center = data_points[center_idx]
    
    # Calculate maximum distance from center to any point
    distances_from_center = np.sqrt(np.sum((data_points - center)**2, axis=1))
    max_distance = np.max(distances_from_center)
    max_radius = max_distance * 1.2
    
    # Generate spiral
    x_spiral, y_spiral, theta, radii, turn_indices, turn_radii, differences = generate_pi_spiral(
        center, max_radius, n_turns=15, points_per_turn=150
    )
    
    # Map points to spiral
    spiral_projection = map_points_to_spiral(data_points, center, x_spiral, y_spiral, theta, radii)
    
    # Calculate turn distances
    turn_distances = calculate_turn_distances(turn_radii)
    
    # Define pastel colors for the 3 cell types
    pastel_colors = {
        'OGC': '#FFB6C1',  # Pastel pink
        'OPC': '#ADD8E6',  # Pastel blue
        'ASC': '#FFFF00'   # Pastel yellow
    }
    
    # Get unique cell types
    unique_celltypes = np.unique(celltypes)
    
    # Create color mapping dictionary
    color_dict = {}
    for cell_type in unique_celltypes:
        if cell_type in pastel_colors:
            color_dict[cell_type] = pastel_colors[cell_type]
        else:
            color_dict[cell_type] = '#D3D3D3'
    
    # Plot 1: Spiral with all points
    spiral_colors = cm.viridis(np.linspace(0, 1, len(turn_indices)))
    
    # Plot spiral with gradient coloring
    for i in range(len(turn_indices)-1):
        idx1, idx2 = turn_indices[i], turn_indices[i+1]
        ax1.plot(x_spiral[idx1:idx2], y_spiral[idx1:idx2],
                color=spiral_colors[i], linewidth=2.5, alpha=0.8, zorder=10)
    
    # Plot all data points using pastel colors
    for cell_type in unique_celltypes:
        indices = np.where(celltypes == cell_type)[0]
        if len(indices) > 0:
            ax1.scatter(data_points[indices, 0], data_points[indices, 1],
                       c=[color_dict[cell_type]],
                       alpha=0.65, s=10, label=cell_type, 
                       zorder=5, edgecolor='none')
    
    # Highlight the Monte Carlo path if available
    if mc_results:
        # Highlight visited barcodes
        visited_mask = np.array([barcode in mc_results['visited_barcodes'] 
                                for barcode in df['barcode']])
        ax1.scatter(data_points[visited_mask, 0], data_points[visited_mask, 1],
                   c='red', s=15, alpha=0.4, zorder=6, 
                   label='Visited in MC', marker='s')
    
    # Highlight the center point
    center_celltype = celltypes[center_idx]
    center_color = color_dict.get(center_celltype, 'red')
    ax1.scatter(center[0], center[1],
               marker='*', s=50, color='red', edgecolor='red', 
               linewidth=1.5, label=f'Start: {center_celltype}', zorder=15)
    
    ax1.set_xlabel('MDS Component 1', fontsize=14, fontweight='normal')
    ax1.set_ylabel('MDS Component 2', fontsize=14, fontweight='normal')
    ax1.set_title('Spiral Visualization with Monte Carlo Simulation', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(-0.75, 0.4)
    ax1.set_ylim(-0.75, 1.0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Radial distance analysis
    turns = np.arange(1, len(turn_distances) + 1)
    bars = ax2.bar(turns, turn_distances, color=cm.viridis(np.linspace(0.3, 0.9, len(turn_distances))),
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Turn Number (n)', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Radial Distance: Rₙ - Rₙ₋₁', fontsize=14, fontweight='normal')
    ax2.set_title('Increasing Distance Between Consecutive Turns', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Annotate the increasing pattern
    for i, (bar, dist) in enumerate(zip(bars, turn_distances)):
        ax2.text(bar.get_x() + bar.get_width()/2, dist + 0.02*max(turn_distances),
                f'{dist:.2f}', ha='center', va='bottom', fontsize=9, fontweight='normal')
    
    # Plot 3: Monte Carlo statistics
    if mc_results:
        ax3.axis('off')
        
        # Create text box with Monte Carlo results
        mc_text = "MONTE CARLO RESULTS\n"
        mc_text += "=" * 30 + "\n"
        mc_text += f"Total Iterations: {mc_results['total_iterations']}\n"
        mc_text += f"Barcodes Visited: {len(mc_results['visited_barcodes'])}\n"
        mc_text += f"Accepted Moves: {len(mc_results['accepted_states'])}\n"
        mc_text += f"Rejected Moves: {len(mc_results['rejected_states'])}\n"
        mc_text += f"Acceptance Rate: {mc_results['acceptance_rate']:.1%}\n\n"
        
        final = mc_results['final_state']
        mc_text += "FINAL STATE:\n"
        mc_text += f"Barcode: {final['barcode']}\n"
        mc_text += f"Celltype: {final['celltype']}\n"
        mc_text += f"Energy: {final['energy']:.4f}\n"
        mc_text += f"Category: {final['category']}\n\n"
        
        mc_text += "METROPOLIS CRITERION:\n"
        mc_text += "1. If ΔE < 0: Always accept\n"
        mc_text += "2. If ΔE ≥ 0: Accept with P = exp(-ΔE/T)\n\n"
        
        mc_text += "ALGORITHM RULES:\n"
        mc_text += "1. Start at OPC center\n"
        mc_text += "2. Propose within current radius\n"
        mc_text += "3. If accept → reset to radius 1\n"
        mc_text += "4. If reject → expand radius\n"
        mc_text += "5. Visit each barcode once\n"
        
        ax3.text(0.05, 0.95, mc_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    
    return fig, ax1, ax2, ax3, turn_distances, turn_radii

# Energy analysis function
def analyze_energy_distribution(df, mc_results):
    """Analyze energy distribution of visited vs unvisited barcodes"""
    visited_barcodes = mc_results['visited_barcodes']
    df_visited = df[df['barcode'].isin(visited_barcodes)]
    df_unvisited = df[~df['barcode'].isin(visited_barcodes)]
    
    print("\n" + "=" * 60)
    print("ENERGY DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Visited barcodes: {len(df_visited)}")
    print(f"Unvisited barcodes: {len(df_unvisited)}")
    print(f"Total barcodes: {len(df)}")
    
    if len(df_visited) > 0:
        print(f"\nVisited barcodes energy:")
        print(f"  Mean: {df_visited['energy'].mean():.4f}")
        print(f"  Min: {df_visited['energy'].min():.4f}")
        print(f"  Max: {df_visited['energy'].max():.4f}")
        print(f"  Std: {df_visited['energy'].std():.4f}")
    
    if len(df_unvisited) > 0:
        print(f"\nUnvisited barcodes energy:")
        print(f"  Mean: {df_unvisited['energy'].mean():.4f}")
        print(f"  Min: {df_unvisited['energy'].min():.4f}")
        print(f"  Max: {df_unvisited['energy'].max():.4f}")
        print(f"  Std: {df_unvisited['energy'].std():.4f}")
    
    # Celltype distribution of visited barcodes
    print(f"\nCelltype distribution of visited barcodes:")
    visited_celltype_counts = df_visited['celltype'].value_counts()
    for celltype, count in visited_celltype_counts.items():
        print(f"  {celltype}: {count} ({count/len(df_visited):.1%})")
    
    return df_visited, df_unvisited

# Main execution
if __name__ == "__main__":
    # Load data from TSV file
    df = pd.read_csv('/Users/mozhganoroujlu/Desktop/MOZHGUN/cell_fate/hi_c/codes_figures/folders/normalized_contacts/non_neuron_filtered_celltypes.tsv', sep='\t')
    
    # Print dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total barcodes: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check if dataset has exactly 918 barcodes
    if len(df) != 918:
        print(f"WARNING: Dataset has {len(df)} barcodes, but Monte Carlo simulation is set for 918 steps")
        print(f"Will run for min(918, {len(df)}) steps")
    
    # Extract data points and celltypes
    data_points = df[['mds_component_1', 'mds_component_2']].values
    celltypes = df['celltype'].values
    energy_values = df['energy'].values
    
    # Print celltype distribution
    print("\n" + "=" * 60)
    print("CELLTYPE DISTRIBUTION:")
    print("=" * 60)
    celltype_counts = df['celltype'].value_counts()
    for celltype, count in celltype_counts.items():
        print(f"{celltype}: {count} cells ({count/len(df):.1%})")
    print()
    
    # Energy statistics
    print("ENERGY STATISTICS:")
    print(f"Mean energy: {df['energy'].mean():.4f}")
    print(f"Min energy: {df['energy'].min():.4f}")
    print(f"Max energy: {df['energy'].max():.4f}")
    print(f"Energy std: {df['energy'].std():.4f}")
    print()
    
    # Select random center from OPC celltype
    opc_df = df[df['celltype'] == 'OPC']
    if opc_df.empty:
        print("WARNING: No points with celltype 'OPC' found. Selecting random center from all cells.")
        center_row = df.sample(1)
    else:
        center_row = opc_df.sample(1)
    
    center_idx = center_row.index[0]
    center_barcode = center_row['barcode'].values[0]
    center_celltype = center_row['celltype'].values[0]
    center_energy = center_row['energy'].values[0]
    
    # Print information about selected center
    print("=" * 60)
    print("SELECTED CENTER INFORMATION:")
    print("=" * 60)
    print(f"Barcode: {center_barcode}")
    print(f"Celltype: {center_celltype}")
    print(f"Energy: {center_energy:.4f}")
    print(f"Coordinates: ({data_points[center_idx][0]:.4f}, {data_points[center_idx][1]:.4f})")
    print(f"Category: {center_row['category'].values[0]}")
    print()
    
    # First, generate spiral to get turn radii for Monte Carlo
    center_coords = data_points[center_idx]
    distances_from_center = np.sqrt(np.sum((data_points - center_coords)**2, axis=1))
    max_distance = np.max(distances_from_center)
    max_radius = max_distance * 1.2
    
    # Generate spiral to get turn radii
    _, _, _, _, _, turn_radii, _ = generate_pi_spiral(
        center_coords, max_radius, n_turns=15, points_per_turn=150
    )
    
    # Run Monte Carlo simulation with corrected Metropolis criterion
    print("=" * 60)
    print("STARTING MONTE CARLO SIMULATION")
    print("=" * 60)
    print("Metropolis Acceptance Criterion:")
    print("1. If ΔE = E_new - E_old < 0: Always accept")
    print("2. If ΔE ≥ 0: Accept with probability P = exp(-ΔE/T)")
    print(f"Temperature: T = 1.0")
    print(f"Target iterations: 918 (one per barcode)")
    print("=" * 60)
    
    mc_results = monte_carlo_spiral_simulation(
        df, center_idx, turn_radii, temperature=0.05, n_iterations=1000
    )
    
    # Visualize with Monte Carlo results
    fig, ax1, ax2, ax3, turn_distances, turn_radii = visualize_spiral_mapping_with_mc(
        data_points, celltypes, energy_values, 
        center_idx=center_idx, barcode=center_barcode, mc_results=mc_results
    )
    
    # Save the figure
    plt.savefig('pi_spiral_monte_carlo_corrected.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('pi_spiral_monte_carlo_corrected.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Energy analysis
    df_visited, df_unvisited = analyze_energy_distribution(df, mc_results)
    
    # Detailed Monte Carlo analysis
    print("\n" + "=" * 60)
    print("DETAILED MONTE CARLO ANALYSIS:")
    print("=" * 60)
    
    final_state = mc_results['final_state']
    initial_energy = df.iloc[center_idx]['energy']
    final_energy = final_state['energy']
    energy_change = final_energy - initial_energy
    
    print(f"Initial energy: {initial_energy:.4f}")
    print(f"Final energy: {final_energy:.4f}")
    print(f"Energy change: {energy_change:+.4f}")
    if abs(initial_energy) > 1e-10:
        print(f"Percent change: {100*energy_change/abs(initial_energy):+.2f}%")
    
    # Accepted moves analysis
    accepted_states = mc_results['accepted_states']
    if accepted_states:
        print(f"\nAccepted moves analysis:")
        print(f"  Number of accepted moves: {len(accepted_states)}")
        
        # Energy changes in accepted moves
        accepted_delta_energies = [state['delta_energy'] for state in accepted_states]
        negative_delta = sum(1 for de in accepted_delta_energies if de < 0)
        positive_delta = len(accepted_delta_energies) - negative_delta
        
        print(f"  Moves with ΔE < 0: {negative_delta} ({negative_delta/len(accepted_states):.1%})")
        print(f"  Moves with ΔE ≥ 0: {positive_delta} ({positive_delta/len(accepted_states):.1%})")
        
        if positive_delta > 0:
            print(f"  Average acceptance probability for ΔE ≥ 0 moves: "
                  f"{np.mean([s['acceptance_prob'] for s in accepted_states if s['delta_energy'] >= 0]):.4f}")
    
    # Radius exploration statistics
    radius_history = mc_results['radius_history']
    unique_radii = np.unique(radius_history)
    print(f"\nRadius exploration:")
    print(f"  Unique radii used: {len(unique_radii)}")
    print(f"  Maximum radius reached: {max(radius_history):.4f}")
    print(f"  Most common radius: {np.bincount(np.digitize(radius_history, unique_radii)).argmax()}")
    
    # Save detailed results to file
    results_df = pd.DataFrame({
        'iteration': range(len(mc_results['radius_history'])),
        'current_radius': mc_results['radius_history']
    })
    
    # Add accepted states information
    if mc_results['accepted_states']:
        accepted_df = pd.DataFrame(mc_results['accepted_states'])
        results_df = pd.merge(results_df, accepted_df[['iteration', 'barcode', 'celltype', 'energy', 'delta_energy', 'acceptance_prob']], 
                             on='iteration', how='left')
    
    results_df.to_csv('monte_carlo_simulation_detailed_results.csv', index=False)
    
    # Save visited barcodes
    visited_df = df[df['barcode'].isin(mc_results['visited_barcodes'])]
    visited_df.to_csv('visited_barcodes_in_mc.csv', index=False)
    
    print(f"\nDetailed results saved to 'monte_carlo_simulation_detailed_results.csv'")
    print(f"Visited barcodes saved to 'visited_barcodes_in_mc.csv'")
    
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY:")
    print("=" * 60)
    print(f"Monte Carlo simulation completed with {mc_results['total_iterations']} iterations.")
    print(f"Unique barcodes visited: {len(mc_results['visited_barcodes'])}")
    print(f"Acceptance rate: {mc_results['acceptance_rate']:.2%}")
    print(f"\nFINAL STATE:")
    print(f"Barcode: {final_state['barcode']}")
    print(f"Cell type: {final_state['celltype']}")
    print(f"Energy: {final_state['energy']:.4f}")
    print(f"Category: {final_state['category']}")
    print(f"\nFigures saved as 'pi_spiral_monte_carlo_corrected.png/.pdf'")
    print("=" * 60)
