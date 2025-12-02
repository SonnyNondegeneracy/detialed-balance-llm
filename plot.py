"""
Plotting tools for potential analysis.

This module provides functions to create various plots for analyzing
transition graphs and potential functions.
"""

import json
import os
import numpy as np
import networkx as nx
from collections import Counter
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from build_potential import PotentialBuilder

# Set matplotlib parameters for PRL single-column format
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 7


def plot_loop_balance(database_path, output_dir='/data/sonny/idea-agent/funsearch-version/final_git/figures', 
                      loop_threshold=0, sample_ratio=5, random_seed=42):
    """
    Find loops in the graph and plot forward vs backward transition weights.
    
    Parameters:
    -----------
    database_path : str
        Path to the JSON database file
    output_dir : str
        Directory to save the output figure
    loop_threshold : float
        Threshold for filtering loops (default: 0)
    sample_ratio : int
        Ratio for random sampling (1/sample_ratio points will be plotted, default: 5)
    random_seed : int
        Random seed for reproducible sampling (default: 42)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {database_path}...")
    with open(database_path, 'r') as f:
        ideas = json.load(f)
    
    # Build graph
    graph = nx.DiGraph()
    node_counts = Counter()
    
    print("Building graph...")
    for idea in ideas:
        target_expr = idea['target_expression']
        for source in idea['source_expressions']:
            source_expr = source['expression']
            node_counts[source_expr] += 1
            if source_expr == target_expr:
                continue
            if graph.has_edge(source_expr, target_expr):
                graph[source_expr][target_expr]['weight'] += 1
            else:
                graph.add_edge(source_expr, target_expr, weight=1)
    
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Collect loop data
    loop_data = []
    seen_cycles = set()
    
    print("\nSearching for 3-cycles...")
    for f in graph.nodes():
        for g in graph.successors(f):
            for h in graph.successors(g):
                if h == f or h == g or f == g:
                    continue
                if graph.has_edge(h, f):
                    # Found cycle f -> g -> h -> f
                    # Check if reverse cycle exists
                    if graph.has_edge(f, h) and graph.has_edge(h, g) and graph.has_edge(g, f):
                        cycle_key = tuple(sorted([f, g, h]))
                        if cycle_key in seen_cycles:
                            continue
                        seen_cycles.add(cycle_key)
                        
                        try:
                            n_f, n_g, n_h = node_counts[f], node_counts[g], node_counts[h]
                            n_fg, n_gh, n_hf = graph[f][g]['weight'], graph[g][h]['weight'], graph[h][f]['weight']
                            n_fh, n_hg, n_gf = graph[f][h]['weight'], graph[h][g]['weight'], graph[g][f]['weight']
                            
                            # if 0 in [n_f, n_g, n_h, n_fg, n_gh, n_hf, n_fh, n_hg, n_gf]:
                            #     continue
                            # if 1 in [n_f, n_g, n_h, n_fg, n_gh, n_hf, n_fh, n_hg, n_gf]:
                            #     continue
                            if min(n_f, n_g, n_h, n_fg, n_gh, n_hf, n_fh, n_hg, n_gf) <= loop_threshold:
                                continue
                            
                            log_forward = math.log(n_fg/n_f) + math.log(n_gh/n_g) + math.log(n_hf/n_h)
                            log_backward = math.log(n_fh/n_f) + math.log(n_hg/n_h) + math.log(n_gf/n_g)
                            
                            rel_err_fwd_sq = (1/n_fg + 1/n_f) + (1/n_gh + 1/n_g) + (1/n_hf + 1/n_h)
                            rel_err_bwd_sq = (1/n_fh + 1/n_f) + (1/n_hg + 1/n_h) + (1/n_gf + 1/n_g)
                            error_forward = math.sqrt(rel_err_fwd_sq)
                            error_backward = math.sqrt(rel_err_bwd_sq)
                            rel_err_sq = rel_err_fwd_sq + rel_err_bwd_sq
                            
                            if rel_err_sq < 100:
                                loop_data.append({
                                    'size': 3,
                                    'log_forward': log_forward,
                                    'log_backward': log_backward,
                                    'error_forward': error_forward,
                                    'error_backward': error_backward,
                                    'rel_err_sq': rel_err_sq,
                                    'cycle': (f, g, h)
                                })
                        except (ZeroDivisionError, KeyError, ValueError):
                            continue
    
    print(f"Found {len(loop_data)} 3-cycles with small errors")
    
    if not loop_data:
        print("No suitable loops found!")
        return
    
    loop_data.sort(key=lambda x: x['rel_err_sq'])
    
    # Prepare plot data with random sampling
    x_data = np.array([d['log_forward'] for d in loop_data])
    y_data = np.array([d['log_backward'] for d in loop_data])
    errors_x = np.array([d['error_forward'] for d in loop_data])
    errors_y = np.array([d['error_backward'] for d in loop_data])
    
    # Calculate chi-squared
    total_errors_sq = errors_x**2 + errors_y**2
    weights = 1.0 / (total_errors_sq + 1e-10)
    weighted_differences = (y_data - x_data) * np.sqrt(weights)
    chi_squared = np.sum(weighted_differences**2)
    ndf = len(x_data)
    chi2_per_ndf = chi_squared / ndf if ndf > 0 else 0
    
    # Random sampling for plotting
    np.random.seed(random_seed)
    total_points = len(x_data)
    sample_indices = np.random.choice(total_points, size=total_points//sample_ratio, replace=False)
    sample_indices = np.sort(sample_indices)
    
    x_data_plot = x_data[sample_indices]
    y_data_plot = y_data[sample_indices]
    errors_x_plot = errors_x[sample_indices]
    errors_y_plot = errors_y[sample_indices]
    
    print(f"\nPlotting {len(sample_indices)} out of {total_points} points (random sampling)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    
    ax.errorbar(x_data_plot, y_data_plot, 
               xerr=errors_x_plot, yerr=errors_y_plot,
               fmt='o', alpha=0.6, markersize=4,
               ecolor='lightgray', capsize=2,
               label='3 states cycle', color='blue')
    
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    ax.plot([min_val-1.5, max_val], [min_val-1.5, max_val],
            'r--', linewidth=1.5, label='Detailed balance', zorder=10)
    
    ax.set_xlabel(r'$\sum \log \mathcal{T}(\text{forward})$', fontsize=9)
    ax.set_ylabel(r'$\sum \log \mathcal{T}(\text{backward})$', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    
    textstr = f'$N$ = {ndf}\n$\\chi^2/\\mathrm{{ndf}}$ = {chi2_per_ndf:.2f}'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                     edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'loop_balance_plot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()
    
    return loop_data


def plot_detailed_balance(database_path, output_dir='/data/sonny/idea-agent/funsearch-version/final_git/figures',
                         reject_mode='high_reject', N_threshold=5, sample_ratio=5, random_seed=42):
    """
    Plot detailed balance: log(w_fg/w_gf) vs beta*(V_f - V_g).
    
    Parameters:
    -----------
    database_path : str
        Path to the JSON database file
    output_dir : str
        Directory to save the output figure
    reject_mode : str
        Mode for PotentialBuilder ('high_reject' or 'low_reject')
    N_threshold : int
        Threshold for low_reject mode
    sample_ratio : int
        Ratio for random sampling (1/sample_ratio points will be plotted, default: 5)
    random_seed : int
        Random seed for reproducible sampling (default: 42)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build potential
    print("Building potential...")
    builder = PotentialBuilder(database_path, reject_mode=reject_mode, N_threshold=N_threshold)
    
    print("Optimizing potentials...")
    optimized_potentials = builder.optimize_potentials(max_iter=20000)
    
    graph = builder.get_graph()
    node_counts = builder.node_counts
    
    # Collect data points
    x_data = []
    y_data = []
    x_errors = []
    
    threshold = 0
    processed_pairs = set()
    
    print("Collecting data points...")
    for f, g in graph.edges():
        if (g, f) in processed_pairs:
            continue
            
        n_fg = graph[f][g]['weight']
        n_gf = graph[g][f]['weight'] if graph.has_edge(g, f) else 0
        
        N = 1
        if n_fg > N and n_gf > N:
            N_f = node_counts.get(f, 1)
            N_g = node_counts.get(g, 1)
            w_fg = n_fg / N_f
            w_gf = n_gf / N_g
            
            if w_fg < threshold or w_gf < threshold:
                continue
                
            V_f = optimized_potentials.get(f, 0)
            V_g = optimized_potentials.get(g, 0)
            
            if abs(abs(V_f)-20.0) < 1 or abs(abs(V_g)-20.0) < 1:
                continue
            if abs(V_f - V_g) > np.log(50228):
                continue
            
            x_val = np.log(w_fg / w_gf)
            y_val = V_f - V_g
            x_err = np.sqrt((1/n_fg + 1/N_f) + (1/n_gf + 1/N_g))
            
            x_data.append(x_val)
            y_data.append(y_val)
            x_errors.append(x_err)
            
            processed_pairs.add((f, g))
    
    print(f"Collected {len(x_data)} data points")
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_errors = np.array(x_errors)
    
    # Calculate statistics
    from scipy.stats import pearsonr
    correlation_coef, p_value = pearsonr(x_data, y_data)
    chi_squared = np.sum((y_data - x_data)**2)
    ndf = len(x_data)
    chi2_per_ndf = chi_squared / ndf
    
    # Random sampling with random flipping
    np.random.seed(random_seed)
    total_points = len(x_data)
    sample_indices = np.random.choice(total_points, size=total_points//sample_ratio, replace=False)
    reverse_flags = np.random.choice([1, -1], size=len(sample_indices))
    
    y_data_plot = y_data[sample_indices] * reverse_flags
    x_data_plot = x_data[sample_indices] * reverse_flags
    yerr_plot = x_errors[sample_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    
    ax.errorbar(y_data_plot, x_data_plot, yerr=yerr_plot, 
                fmt='o', alpha=0.6, markersize=3, 
                ecolor='lightgray', capsize=1.5, label='Data points')
    
    min_val = min(x_data_plot.min(), y_data_plot.min())-2
    max_val = max(x_data_plot.max(), y_data_plot.max())+2
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=1.5, label='Detailed balance')
    
    ax.set_xlim(y_data_plot.min()-2, y_data_plot.max()+2)
    ax.set_ylim(x_data_plot.min()-2, x_data_plot.max()+2)
    
    ax.set_xlabel(r"$\beta \left(V(f) - V(g)\right)$", fontsize=9)
    ax.set_ylabel(r'$\log\left(\frac{\mathcal{T}(f, g)}{\mathcal{T}(g, f)}\right)$', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    
    textstr = f'$N$ = {ndf}\n$r$ = {correlation_coef:.4f}\n$\\chi^2/\\mathrm{{ndf}}$ = {chi2_per_ndf:.2f}'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'detailed_balance_plot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()
    
    print(f"\nStatistics:")
    print(f"  Data points: {ndf}")
    print(f"  Pearson correlation (r): {correlation_coef:.4f}")
    print(f"  χ²/ndf: {chi2_per_ndf:.4f}")
    
    return x_data, y_data, x_errors


def draw_potential_graph(database_path, potential_func, output_dir='./figures',
                        p0_threshold=0.05, num_nodes_to_plot=70, random_seed=42):
    """
    Draw graph with nodes positioned by their potential and log(MSE).
    
    Parameters:
    -----------
    database_path : str
        Path to the JSON database file
    potential_func : callable
        Function that takes an expression string and returns its potential V(f)
    output_dir : str
        Directory to save the output figure
    p0_threshold : float
        Threshold for edge weight filtering (default: 0.05)
    num_nodes_to_plot : int
        Number of top nodes to plot (default: 70)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    def score_to_mse(score):
        """Convert score to MSE."""
        return 0.06020935250966546 * (10 ** (-4/60 * (score - 20.00)))
    
    print(f"Loading data from {database_path}...")
    with open(database_path, 'r') as f:
        ideas = json.load(f)
    
    # Build graph
    graph = nx.DiGraph()
    node_counts = Counter()
    score_map = {}
    
    print("Building graph...")
    for idea in ideas:
        target_expr = idea['target_expression']
        target_score = idea.get('target_score', 20.0)
        
        if target_expr not in score_map:
            score_map[target_expr] = target_score
        
        for source in idea['source_expressions']:
            source_expr = source['expression']
            source_score = source.get('score', 20.0)
            
            if source_expr not in score_map:
                score_map[source_expr] = source_score
            
            node_counts[source_expr] += 1
            if source_expr == target_expr:
                continue
            if graph.has_edge(source_expr, target_expr):
                graph[source_expr][target_expr]['weight'] += 1
            else:
                graph.add_edge(source_expr, target_expr, weight=1)
    
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Select subset of nodes
    if graph.number_of_nodes() > num_nodes_to_plot:
        np.random.seed(random_seed)
        top_nodes = sorted(graph.nodes(), key=lambda node: node_counts.get(node, 0), reverse=True)[:2*num_nodes_to_plot:2]
        subgraph = graph.subgraph(top_nodes)
    else:
        subgraph = graph
    
    print(f"Plotting subgraph with {subgraph.number_of_nodes()} nodes")
    
    # Position nodes
    pos = {}
    for node in subgraph.nodes():
        score = score_map.get(node, 20.0)
        mse = score_to_mse(score)
        x = np.log10(mse)
        y = potential_func(node)
        pos[node] = (x, y)
    
    # Separate edges by potential direction
    downward_edges = []
    downward_widths = []
    upward_edges = []
    upward_widths = []
    
    for u, v in subgraph.edges():
        N = node_counts.get(u, 0)
        if N > 0:
            n = graph[u][v]['weight']
            w = n / N
            if w >= p0_threshold:
                V_u = potential_func(u)
                V_v = potential_func(v)
                
                if V_u > V_v:
                    downward_edges.append((u, v))
                    downward_widths.append(w * 1.5)
                else:
                    upward_edges.append((u, v))
                    upward_widths.append(w * 1.5)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    
    # Node colors based on log(MSE)
    node_log_mse = []
    for node in subgraph.nodes():
        score = score_map.get(node, 20.0)
        mse = score_to_mse(score)
        node_log_mse.append(np.log10(mse))
    
    if node_log_mse:
        min_color_val, max_color_val = min(node_log_mse), max(node_log_mse)
    else:
        min_color_val, max_color_val = 0, 1
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=30, node_color=node_log_mse,
                          cmap=plt.cm.viridis_r, alpha=0.8, vmin=min_color_val, vmax=max_color_val,
                          linewidths=0.3, edgecolors='black', ax=ax)
    
    nx.draw_networkx_edges(subgraph, pos, edgelist=downward_edges, edge_color='g',
                          alpha=0.5, arrows=True, width=downward_widths,
                          arrowsize=5, arrowstyle='->', connectionstyle='arc3,rad=0.1', ax=ax)
    
    nx.draw_networkx_edges(subgraph, pos, edgelist=upward_edges, edge_color='r',
                          alpha=0.5, arrows=True, width=upward_widths,
                          arrowsize=5, arrowstyle='->', connectionstyle='arc3,rad=0.1', ax=ax)
    
    ax.set_xlabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=12)
    ax.set_ylabel(r"Potential $V(f)$", fontsize=12)
    ax.set_axis_on()
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=min_color_val, vmax=max_color_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10}(\mathrm{MSE})$', rotation=270, labelpad=20)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='white', linewidth=0, label='Upward',
               marker='^', markersize=10, markerfacecolor='r', markeredgecolor='r', linestyle='None'),
        Line2D([0], [0], color='white', linewidth=0, label='Downward',
               marker='v', markersize=10, markerfacecolor='g', markeredgecolor='g', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'potential_graph.pdf')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Graph saved to {output_path}")
    plt.close()


def plot_potential_distribution(database_path, potential_func, output_dir='./figures'):
    """
    Plot potential distribution with Gaussian fit.
    
    Parameters:
    -----------
    database_path : str
        Path to the JSON database file
    potential_func : callable
        Function that takes an expression string and returns its potential V(f)
    output_dir : str
        Directory to save the output figure
    """
    from scipy.optimize import curve_fit
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {database_path}...")
    with open(database_path, 'r') as f:
        ideas = json.load(f)
    
    # Build graph
    graph = nx.DiGraph()
    node_counts = Counter()
    
    print("Building graph...")
    for idea in ideas:
        target_expr = idea['target_expression']
        for source in idea['source_expressions']:
            source_expr = source['expression']
            node_counts[source_expr] += 1
            if source_expr == target_expr:
                continue
            if graph.has_edge(source_expr, target_expr):
                graph[source_expr][target_expr]['weight'] += 1
            else:
                graph.add_edge(source_expr, target_expr, weight=1)
    
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Collect potentials for nodes with count > 1
    all_potentials = [potential_func(f) for f in graph.nodes() if node_counts.get(f, 0) > 1]
    print(f"{len(all_potentials)} nodes with count > 1 for potential distribution")
    
    # Plot histogram
    fig = plt.figure(figsize=(3.4, 2.6))
    hist, bin_edges, patches = plt.hist(all_potentials, bins=50, color='blue', alpha=0.7,
                                       edgecolor='black', linewidth=0.3, label='Data')
    
    # Gaussian fit
    def gaussian_potential(V, mu, sigma, A):
        return A * np.exp(-(V - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu_init = np.mean(all_potentials)
    sigma_init = np.std(all_potentials)
    A_init = hist.max() * np.sqrt(2 * np.pi) * sigma_init
    
    try:
        popt, pcov = curve_fit(gaussian_potential, bin_centers, hist,
                              p0=[mu_init, sigma_init, A_init], maxfev=20000)
        mu_fit, sigma_fit, A_fit = popt
        print(f"Gaussian fit: mu={mu_fit:.4g}, sigma={sigma_fit:.4g}")
        
        V_fit = np.linspace(min(all_potentials), max(all_potentials), 200)
        g_fit = gaussian_potential(V_fit, mu_fit, sigma_fit, A_fit)
        plt.plot(V_fit, g_fit, 'r--', lw=1.2, label='Fit')
        plt.legend(frameon=False, loc='best')
    except Exception as e:
        print(f"Gaussian fit failed: {e}")
    
    plt.xlabel(r'$\beta V(f)$')
    plt.ylabel('Count')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'potential_distribution.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()
    
    return all_potentials


# Example usage
if __name__ == "__main__":
    database_path = './data/ideasearchfitter_database.json'
    output_dir = './figures/example'
    
    print("=" * 80)
    print("Generating Loop Balance Plot")
    print("=" * 80)
    plot_loop_balance(database_path, output_dir=output_dir, loop_threshold=0, 
                     sample_ratio=5, random_seed=42)
    
    print("\n" + "=" * 80)
    print("Generating Detailed Balance Plot")
    print("=" * 80)
    plot_detailed_balance(database_path, output_dir=output_dir, reject_mode='low_reject', 
                         N_threshold=1, sample_ratio=5, random_seed=42)
    
    print("\n" + "=" * 80)
    print("Generating Potential Graph")
    print("=" * 80)
    # First optimize potentials to get the potential function
    builder = PotentialBuilder(database_path, reject_mode='low_reject', N_threshold=1)
    optimized_potentials = builder.optimize_potentials(max_iter=20000)
    
    # Define potential function
    def V(f):
        return optimized_potentials.get(f, 0.0)
    
    draw_potential_graph(database_path, V, output_dir=output_dir, 
                        p0_threshold=0.05, num_nodes_to_plot=70, random_seed=42)
    
    print("\n" + "=" * 80)
    print("Generating Potential Distribution Plot")
    print("=" * 80)
    plot_potential_distribution(database_path, V, output_dir=output_dir)
