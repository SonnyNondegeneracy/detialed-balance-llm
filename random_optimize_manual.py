import re
import json
import os
import numpy as np
import networkx as nx
from collections import Counter
# from potential_optimized import potential_optimized, default_params
from ideasearch_potential import potential_optimized_batch,default_params

VOCAB_DICT = {
    'sin': 0, 'cos': 1, 'tan': 2, 'arcsin': 3, 'arccos': 4, 'arctan': 5, 'tanh': 6, 'log': 7, 'log10': 8, 'exp': 9,
    'square': 10, 'sqrt': 11, 'abs': 12, '*': 13, '**': 14, '/': 15, '+': 16, '-': 17, '1': 18, '2': 19, 'pi': 20,
    'log_v_k_nu': 21, 'param1': 22, 'param2': 23, 'param3': 24, 'param4': 25, 'param5': 26, 'param6': 27,
    'param7': 28, 'param8': 29, 'param9': 30, '(': 31, ')': 32, ' ': 33
}

def V_hand_optimized(f: str, params: dict = {}) -> float:
    token_pattern = r'sin|cos|tan|arcsin|arccos|arctan|tanh|log10|log|exp|square|sqrt|abs|\*\*|\*|/|\+|-|\(|\)|\s+|param[1-9]|log_v_k_nu|1|2|pi'
    tokens = re.findall(token_pattern, f)
    token_ids = [VOCAB_DICT[token] for token in tokens if token in VOCAB_DICT]
    return potential_optimized_batch([token_ids], params)[0]

def K(x):
    z = -x
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))

def calculate_action(graph, potentials, node_counts):
    action_numerator = 0
    for f, g in graph.edges():
        N = node_counts.get(f, 0)
        if N > 1:
            n = graph[f][g]['weight']
            w = n / N
            action_numerator += w * K(potentials.get(f, 0) - potentials.get(g, 0))
    
    total_nodes_with_out_edges = len([node for node, count in node_counts.items() if count > 1])
    if total_nodes_with_out_edges == 0:
        return 0
    return action_numerator / total_nodes_with_out_edges

def load_graph_from_json(file_path):
    graph = nx.DiGraph()
    node_counts = Counter()
    with open(file_path, 'r') as f:
        ideas = json.load(f)
    for idea in ideas:
        target_expr = idea['target_expression']
        for source in idea['source_expressions']:
            source_expr = source['expression']
            node_counts[source_expr] += 1
            if source_expr != target_expr:
                if graph.has_edge(source_expr, target_expr):
                    graph[source_expr][target_expr]['weight'] += 1
                else:
                    graph.add_edge(source_expr, target_expr, weight=1)
    return graph, node_counts

def optimize_with_random_descent(graph, node_counts, n_iterations=100, n_random_samples=20, 
                                  perturbation_scale=0.2, best_k=5, save_interval=10,
                                  init_params_file=None):
    """
    Optimize using random descent: sample random perturbations and keep the best.
    Each iteration selects ONE parameter to adjust (instead of all parameters).
    
    Args:
        graph: The graph structure
        node_counts: Counter of node occurrences
        n_iterations: Number of optimization iterations
        n_random_samples: Number of random samples per iteration
        perturbation_scale: Scale of random perturbations
        best_k: Keep best k samples for next iteration
        save_interval: Save progress every N iterations
        init_params_file: Path to JSON file with initial parameters (optional)
    
    Returns:
        optimized_params: Best parameters found
        min_action: Minimum action value achieved
        history: Optimization history
    """
    param_keys = [k for k, v in default_params.items() if isinstance(v, (int, float))]
    
    # Load initial parameters from file if provided
    if init_params_file and os.path.exists(init_params_file):
        print(f"Loading initial parameters from {init_params_file}...")
        with open(init_params_file, 'r') as f:
            loaded_data = json.load(f)
            if 'optimized_params' in loaded_data:
                init_params = loaded_data['optimized_params']
                print(f"Loaded {len(init_params)} parameters from file")
                print(f"Previous best action: {loaded_data.get('min_action', 'N/A')}")
            else:
                init_params = default_params
                print("File format not recognized, using default parameters")
    else:
        init_params = default_params
        print("Using default parameters")
    
    all_nodes = list(graph.nodes())
    token_pattern = r'sin|cos|tan|arcsin|arccos|arctan|tanh|log10|log|exp|square|sqrt|abs|\*\*|\*|/|\+|-|\(|\)|\s+|param[1-9]|log_v_k_nu|1|2|pi'
    pre_tokenized_nodes = {
        f: [VOCAB_DICT[token] for token in re.findall(token_pattern, f) if token in VOCAB_DICT]
        for f in all_nodes
    }
    
    node_list = list(pre_tokenized_nodes.keys())
    token_ids_list = [pre_tokenized_nodes[f] for f in node_list]

    def objective(x):
        params = {k: v for k, v in zip(param_keys, x)}
        potential_values = potential_optimized_batch(token_ids_list, params)
        potentials = {f: potential_values[i] for i, f in enumerate(node_list)}
        action = calculate_action(graph, potentials, node_counts)
        return action

    # Initialize with loaded or default parameters
    x0 = np.array([init_params.get(k, default_params[k]) for k in param_keys])
    
    # Keep track of best candidates
    best_candidates = [(x0.copy(), objective(x0))]
    
    history = {
        'iterations': [],
        'best_actions': [],
        'mean_actions': []
    }
    
    print(f"Starting random descent optimization...")
    print(f"Initial action: {best_candidates[0][1]:.6f}")
    print(f"Parameters to optimize: {len(param_keys)}")
    print()
    
    for iteration in range(n_iterations):
        # Generate random samples around best candidates
        samples = []
        
        for base_x, _ in best_candidates[:best_k]:
            for _ in range(n_random_samples // len(best_candidates[:best_k])):
                # Select ONE random parameter to adjust
                param_idx = np.random.randint(0, len(base_x))
                
                # Create new candidate by perturbing only the selected parameter
                new_x = base_x.copy()
                perturbation = np.random.normal(0, perturbation_scale * np.abs(base_x[param_idx]))
                new_x[param_idx] = base_x[param_idx] + perturbation
                
                try:
                    action = objective(new_x)
                    samples.append((new_x, action))
                except Exception as e:
                    continue
        
        if not samples:
            print(f"Iteration {iteration + 1}: No valid samples generated")
            continue
        
        # Sort by action (lower is better)
        samples.sort(key=lambda x: x[1])
        
        # Update best candidates
        all_candidates = best_candidates + samples
        all_candidates.sort(key=lambda x: x[1])
        best_candidates = all_candidates[:best_k]
        
        # Record history
        best_action = best_candidates[0][1]
        mean_action = np.mean([action for _, action in samples])
        
        history['iterations'].append(iteration + 1)
        history['best_actions'].append(best_action)
        history['mean_actions'].append(mean_action)
        
        # Print progress
        if (iteration + 1) % save_interval == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}:")
            print(f"  Best action: {best_action:.6f}")
            print(f"  Mean action: {mean_action:.6f}")
            print(f"  Improvement from start: {best_candidates[0][1] - objective(x0):.6f}")
            
            # Save intermediate results
            intermediate_params = {k: v for k, v in zip(param_keys, best_candidates[0][0])}
            with open('./data/best_params_intermediate.json', 'w') as f:
                json.dump({
                    'params': intermediate_params,
                    'action': float(best_action),
                    'iteration': iteration + 1
                }, f, indent=2)
    
    optimized_params = {k: v for k, v in zip(param_keys, best_candidates[0][0])}
    min_action = best_candidates[0][1]
    
    return optimized_params, min_action, history

def optimize_with_gradient_descent(graph, node_counts, n_iterations=100, learning_rate=0.01,
                                    epsilon=1e-4, momentum=0.9, save_interval=10):
    """
    Optimize using gradient descent with finite differences.
    
    Args:
        graph: The graph structure
        node_counts: Counter of node occurrences
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent
        epsilon: Finite difference step size
        momentum: Momentum coefficient
        save_interval: Save progress every N iterations
    
    Returns:
        optimized_params: Best parameters found
        min_action: Minimum action value achieved
        history: Optimization history
    """
    param_keys = [k for k, v in default_params.items() if isinstance(v, (int, float))]
    
    all_nodes = list(graph.nodes())
    token_pattern = r'sin|cos|tan|arcsin|arccos|arctan|tanh|log10|log|exp|square|sqrt|abs|\*\*|\*|/|\+|-|\(|\)|\s+|param[1-9]|log_v_k_nu|1|2|pi'
    pre_tokenized_nodes = {
        f: [VOCAB_DICT[token] for token in re.findall(token_pattern, f) if token in VOCAB_DICT]
        for f in all_nodes
    }
    
    node_list = list(pre_tokenized_nodes.keys())
    token_ids_list = [pre_tokenized_nodes[f] for f in node_list]

    def objective(x):
        params = {k: v for k, v in zip(param_keys, x)}
        potential_values = potential_optimized_batch(token_ids_list, params)
        potentials = {f: potential_values[i] for i, f in enumerate(node_list)}
        action = calculate_action(graph, potentials, node_counts)
        return action

    def compute_gradient(x, f_x):
        """Compute gradient using finite differences"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f_x) / epsilon
        return grad

    # Initialize
    x = np.array([default_params[k] for k in param_keys])
    velocity = np.zeros_like(x)
    
    history = {
        'iterations': [],
        'actions': [],
        'gradient_norms': []
    }
    
    print(f"Starting gradient descent optimization...")
    print(f"Initial action: {objective(x):.6f}")
    print(f"Parameters to optimize: {len(param_keys)}")
    print()
    
    best_x = x.copy()
    best_action = objective(x)
    
    for iteration in range(n_iterations):
        # Compute current objective
        f_x = objective(x)
        
        # Compute gradient
        grad = compute_gradient(x, f_x)
        grad_norm = np.linalg.norm(grad)
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        
        # Track best
        if f_x < best_action:
            best_action = f_x
            best_x = x.copy()
        
        # Record history
        history['iterations'].append(iteration + 1)
        history['actions'].append(f_x)
        history['gradient_norms'].append(grad_norm)
        
        # Print progress
        if (iteration + 1) % save_interval == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}:")
            print(f"  Current action: {f_x:.6f}")
            print(f"  Best action: {best_action:.6f}")
            print(f"  Gradient norm: {grad_norm:.6f}")
            
            # Save intermediate results
            intermediate_params = {k: v for k, v in zip(param_keys, best_x)}
            with open('./data/best_params_intermediate.json', 'w') as f:
                json.dump({
                    'params': intermediate_params,
                    'action': float(best_action),
                    'iteration': iteration + 1
                }, f, indent=2)
    
    optimized_params = {k: v for k, v in zip(param_keys, best_x)}
    
    return optimized_params, best_action, history

if __name__ == "__main__":
    graph_file_path = './data/ideasearchfitter_database.json'
    
    print(f"Loading graph from {graph_file_path}...")
    graph, node_counts = load_graph_from_json(graph_file_path)
    print(f"Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print()
    
    # Choose optimization method
    print("Choose optimization method:")
    print("1. Random Descent (faster, good for exploration)")
    print("2. Gradient Descent (slower, more precise)")
    
    method = input("Enter choice (1 or 2, default=1): ").strip() or "1"
    
    if method == "2":
        print("\n=== Using Gradient Descent ===")
        optimized_params, min_action, history = optimize_with_gradient_descent(
            graph, node_counts,
            n_iterations=50,
            learning_rate=0.01,
            save_interval=5
        )
    else:
        print("\n=== Using Random Descent ===")
        # Path to best_params.json for initialization
        init_params_path = './data/best_params.json'
        
        optimized_params, min_action, history = optimize_with_random_descent(
            graph, node_counts,
            n_iterations=100,
            n_random_samples=20,
            perturbation_scale=0.1,
            save_interval=10,
            init_params_file=init_params_path
        )
    
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Final Action: {min_action:.6f}")
    
    # Save best parameters
    output_data = {
        'optimized_params': optimized_params,
        'min_action': float(min_action),
        'optimization_method': 'gradient_descent' if method == "2" else 'random_descent',
        'history': history
    }
    
    output_file = './data/best_params.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBest parameters saved to: {output_file}")
    
    print("\n=== Optimized Parameters (Top 10) ===")
    sorted_params = sorted(optimized_params.items(), 
                          key=lambda x: abs(x[1] - default_params.get(x[0], 0)), 
                          reverse=True)
    
    for i, (param_name, param_value) in enumerate(sorted_params[:10]):
        default_value = default_params.get(param_name, 0)
        change = param_value - default_value
        print(f"{i+1}. {param_name}: {param_value:.4f} (default: {default_value:.4f}, change: {change:+.4f})")
    
    print("\n=== Testing with Sample Functions ===")
    sample_functions = list(graph.nodes())[:5]
    for func in sample_functions:
        p_default = V_hand_optimized(func)
        p_optimized = V_hand_optimized(func, optimized_params)
        print(f"Function: {func[:60]}...")
        print(f"  Default: {p_default:.4f}, Optimized: {p_optimized:.4f}, Diff: {p_optimized - p_default:+.4f}")
