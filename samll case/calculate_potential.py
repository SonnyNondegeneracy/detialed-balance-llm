import json
import os
import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm
from scipy.optimize import minimize

def K(x):
    """
    Kernel function for the action calculation.
    """
    z = -x
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))

def calculate_action(graph, potentials, node_counts, total_nodes):
    """
    Calculates the action for a given graph and potential function.
    Modified: w estimation changed to N(f->g)/(20000/ total_node)
    """
    action_numerator = 0
    
    for f, g in graph.edges():
        n = graph[f][g]['weight']
        # Modified w estimation: N(f->g) / (20000/ total_node)
        w = min(1, n / (20000/total_nodes))
        action_numerator += w * K(potentials.get(f, 0) - potentials.get(g, 0))
    
    if total_nodes == 0:
        return 0
        
    action = action_numerator / total_nodes
    return action

def initialize_potentials_topological(graph):
    """
    Initializes potentials based on topological sorting. For graphs with cycles,
    it uses the condensation graph (collapsing strongly connected components).
    """
    print("\nInitializing potentials using topological sorting method...")
    potentials = {}
    nodes = list(graph.nodes())

    if not nx.is_directed_acyclic_graph(graph):
        sccs = list(nx.strongly_connected_components(graph))
        print(f"  Graph has cycles. Detected {len(sccs)} strongly connected components.")
        
        condensation = nx.condensation(graph)
        
        scc_levels = {}
        for node in nx.topological_sort(condensation):
            predecessors = list(condensation.predecessors(node))
            if not predecessors:
                scc_levels[node] = 0
            else:
                scc_levels[node] = max(scc_levels.get(p, 0) for p in predecessors) + 1
        
        scc_mapping = condensation.graph['mapping']
        for node in nodes:
            scc_id = scc_mapping[node]
            potentials[node] = -scc_levels[scc_id]
    else:
        print("  Graph is a DAG.")
        for node in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(node))
            if not predecessors:
                potentials[node] = 0.0
            else:
                potentials[node] = min(potentials.get(p, 0) for p in predecessors) - 1.0
    
    # Normalize potentials to have zero mean
    mean_potential = np.mean(list(potentials.values()))
    for node in potentials:
        potentials[node] -= mean_potential

    return potentials

def optimize_potentials(graph, node_counts, total_nodes, max_iter=20000):
    """
    Optimizes potentials to minimize the action using the L-BFGS-B algorithm from scipy.
    Modified: w estimation changed to N(f->g)/(20000, total_node)
    """
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    initial_potentials_dict = initialize_potentials_topological(graph)
    
    # Ensure all nodes have an initial potential
    for node in nodes:
        if node not in initial_potentials_dict:
            initial_potentials_dict[node] = 0.0
            
    v0 = np.array([initial_potentials_dict[node] for node in nodes])

    if total_nodes == 0:
        # Return a zero potential for all nodes if there are no edges to optimize
        return {node: 0.0 for node in nodes}

    # Pre-calculate edge data for efficiency
    edges = []
    for f, g in graph.edges():
        n_fg = graph[f][g]['weight']
        # Modified w estimation: N(f->g) / (20000, total_node)
        w_fg = n_fg / (20000 / total_nodes)
        f_idx = node_to_idx[f]
        g_idx = node_to_idx[g]
        edges.append({'f_idx': f_idx, 'g_idx': g_idx, 'w': w_fg})

    def objective_and_grad(v):
        """
        Calculates both the objective function (action) and its gradient.
        This is more efficient than calculating them separately.
        """
        action_numerator = 0
        grads = np.zeros_like(v)
        
        for edge in edges:
            f_idx, g_idx, w = edge['f_idx'], edge['g_idx'], edge['w']
            delta_v = v[f_idx] - v[g_idx]
            
            # Objective function component using K
            action_numerator += w * K(delta_v)
            
            # Gradient component using numerical differentiation of K
            eps = 1e-8
            K_grad = (K(delta_v + eps) - K(delta_v - eps)) / (2 * eps)
            common_grad_term = w * K_grad
            grads[f_idx] += common_grad_term
            grads[g_idx] -= common_grad_term
            
        action = action_numerator / total_nodes
        normalized_grad = grads / total_nodes
        
        return action, normalized_grad

    print(f"Starting optimization with L-BFGS-B (max_iter={max_iter})...")
    
    result = minimize(
        fun=objective_and_grad,
        x0=v0,
        jac=True,  # Indicates that fun returns both objective and gradient
        method='L-BFGS-B',
        options={'maxiter': max_iter}
    )

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    optimized_v = result.x
    
    # Normalize potentials to have zero mean to ensure a unique solution
    mean_potential = np.mean(optimized_v)
    optimized_v -= mean_potential
    
    optimized_potentials = {idx_to_node[i]: v for i, v in enumerate(optimized_v)}
    
    print("Optimization finished.")
    
    # Calculate and print final energy
    final_action = calculate_action(graph, optimized_potentials, node_counts, total_nodes)
    print(f"\n{'='*60}")
    print(f"FINAL OPTIMIZED ENERGY (ACTION): {final_action:.6f}")
    print(f"{'='*60}\n")
    
    return optimized_potentials, final_action

def main():
    """
    Main function to load data, build graph, optimize potentials, and save them.
    Modified: Uses data from database_claude-4.json with new w estimation method
    """
    data_path = './data/word_database_claude-4.json'
    data_path = './data/word_database_gemini-2.5-flash-nothinking.json'
    output_path = './data/optimized_potentials.json'
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        ideas = json.load(f)
    
    # 构建图
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
    print(f'Totally {sum([w[2]["weight"] for w in graph.edges(data=True)])} edges are legal.')
    
    # Calculate total nodes for the new w estimation method
    total_nodes = graph.number_of_nodes()
    print(f"Total nodes: {total_nodes}")

    # Optimize potentials to minimize action
    optimized_potentials, final_action = optimize_potentials(graph, node_counts, total_nodes, max_iter=20000)
    
    # Print results in the requested format
    print("\n" + "="*60)
    print("OPTIMIZED POTENTIALS AND TRANSITIONS")
    print("="*60)
    
    # Find the node with minimum potential to use as reference (shift to 0)
    # ATTITUD
    min_node = min(optimized_potentials.items(), key=lambda x: x[1])
    reference_node = min_node[0]
    reference_potential = min_node[1]
    
    print(f"\n# Define the energy levels and their potentials")
    print(f"# (Shifted so that '{reference_node}' = 0)")
    print("levels = {")
    
    # Sort nodes by their shifted potential
    sorted_nodes = sorted(optimized_potentials.items(), key=lambda x: x[1])
    for node, potential in sorted_nodes:
        shifted_potential = potential - reference_potential
        print(f"    '{node}': {shifted_potential:.1f},")
    print("}")
    
    print(f"\nlevel_names = list(levels.keys())")
    print(f"level_energies = list(levels.values())")
    
    # Print transition counts
    print(f"\n# Transition data N(f->g)")
    print("transitions_counts = {")
    
    # Sort edges by source node potential (ascending) for better readability
    edges_with_potentials = []
    for f, g in graph.edges():
        weight = graph[f][g]['weight']
        f_potential = optimized_potentials[f] - reference_potential
        edges_with_potentials.append((f, g, weight, f_potential))
    
    edges_with_potentials.sort(key=lambda x: (x[3], x[0], x[1]))
    
    for f, g, weight, _ in edges_with_potentials:
        print(f"    ('{f}', '{g}'): {weight},")
    print("}")
    
    print("\n" + "="*60)
    
    # Save the optimized potentials
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(optimized_potentials, f, indent=2)
        
    print(f"\nSuccessfully saved {len(optimized_potentials)} optimized potentials to: {output_path}")

if __name__ == "__main__":
    main()
