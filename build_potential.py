import json
import numpy as np
import networkx as nx
from collections import Counter
from scipy.optimize import minimize


class PotentialBuilder:
    """
    A class to build and optimize potential functions from a database of transitions.
    
    This class:
    1. Loads transition data from a JSON database
    2. Constructs a directed graph with weighted edges based on transition frequencies
    3. Calculates action based on the graph and potential function
    4. Optimizes potentials to minimize the action
    """
    
    def __init__(self, database_path, reject_mode='high_reject', N_threshold=1):
        """
        Initialize the PotentialBuilder.
        
        Parameters:
        -----------
        database_path : str
            Path to the JSON database file containing transition data
        reject_mode : str
            Mode for calculating N_0(f):
            - 'high_reject': Use 20000 / total_nodes as N_0(f) for all nodes
            - 'low_reject': Use sum of all outgoing transitions from f, 
                           and filter out nodes with N_0(f) <= N_threshold
        N_threshold : int
            Threshold for filtering nodes in 'low_reject' mode (default: 5)
        """
        self.database_path = database_path
        self.reject_mode = reject_mode
        self.N_threshold = N_threshold
        
        # Initialize data structures
        self.graph = None
        self.node_counts = None
        self.N_0 = None
        self.filtered_nodes = None
        
        # Load and build graph
        self._load_and_build_graph()
    
    def _load_and_build_graph(self):
        """
        Load data from JSON database and build the directed graph.
        """
        print(f"Loading data from {self.database_path}...")
        with open(self.database_path, 'r') as f:
            ideas = json.load(f)
        
        print(f"Loaded {len(ideas)} ideas from JSON file.")
        
        # Build graph and count transitions
        self.graph = nx.DiGraph()
        self.node_counts = Counter()
        
        print("Building graph...")
        for idea in ideas:
            target_expr = idea['target_expression']
            for source in idea['source_expressions']:
                source_expr = source['expression']
                self.node_counts[source_expr] += 1
                
                if source_expr == target_expr:
                    continue
                    
                if self.graph.has_edge(source_expr, target_expr):
                    self.graph[source_expr][target_expr]['weight'] += 1
                else:
                    self.graph.add_edge(source_expr, target_expr, weight=1)
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Calculate N_0(f) based on reject_mode
        self._calculate_N_0()
        
        # Calculate edge weights w(f->g) = min(1, N(f->g) / N_0(f))
        self._calculate_edge_weights()
    
    def _calculate_N_0(self):
        """
        Calculate N_0(f) for each node based on the reject_mode.
        """
        self.N_0 = {}
        self.filtered_nodes = set()
        
        if self.reject_mode == 'high_reject':
            # N_0(f) = 20000 / total_nodes for all nodes
            total_nodes = self.graph.number_of_nodes()
            N_0_value = 20000.0 / total_nodes if total_nodes > 0 else 1.0
            
            for node in self.graph.nodes():
                self.N_0[node] = N_0_value
            
            print(f"Using high_reject mode: N_0 = {N_0_value:.2f} for all nodes")
            
        elif self.reject_mode == 'low_reject':
            # N_0(f) = sum of all outgoing transitions from f
            for node in self.graph.nodes():
                N_0_f = self.node_counts.get(node, 0)
                self.N_0[node] = N_0_f
                
                # Filter out nodes with N_0(f) <= N_threshold
                if N_0_f <= self.N_threshold:
                    self.filtered_nodes.add(node)
            
            print(f"Using low_reject mode: Filtered {len(self.filtered_nodes)} nodes with N_0 <= {self.N_threshold}")
        else:
            raise ValueError(f"Unknown reject_mode: {self.reject_mode}. Use 'high_reject' or 'low_reject'.")
    
    def _calculate_edge_weights(self):
        """
        Calculate edge weights w(f->g) = min(1, N(f->g) / N_0(f)).
        Store as 'normalized_weight' attribute on edges.
        """
        for f, g in self.graph.edges():
            N_fg = self.graph[f][g]['weight']
            N_0_f = self.N_0.get(f, 1.0)
            w_fg = min(1.0, N_fg / N_0_f)
            self.graph[f][g]['normalized_weight'] = w_fg
    
    def get_graph(self):
        """
        Get the directed graph with weighted edges.
        
        Returns:
        --------
        graph : networkx.DiGraph
            The directed graph with edges having 'weight' (raw count) and 
            'normalized_weight' (w(f->g) = min(1, N(f->g) / N_0(f))) attributes
        """
        return self.graph
    
    @staticmethod
    def K(x):
        """
        Kernel function for the action calculation.
        K(x) = log(1 + exp(-x)) for x = -delta_V
        
        Parameters:
        -----------
        x : float or np.ndarray
            Input value(s)
        
        Returns:
        --------
        float or np.ndarray
            K(x) value(s)
        """
        z = -x
        return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    
    def calculate_action(self, potentials):
        """
        Calculate the action for given potentials.
        
        Action = (1/N_total) * sum_{f->g} w(f->g) * K(V(f) - V(g))
        
        where N_total is the number of nodes with outgoing edges (N_0(f) > 1).
        
        Parameters:
        -----------
        potentials : dict
            Dictionary mapping node expressions to potential values
        
        Returns:
        --------
        float
            The calculated action value
        """
        action_numerator = 0
        
        # Count nodes with outgoing edges (excluding filtered nodes in low_reject mode)
        if self.reject_mode == 'low_reject':
            total_nodes_with_out_edges = len([
                node for node in self.graph.nodes() 
                if node not in self.filtered_nodes and self.N_0.get(node, 0) > self.N_threshold
            ])
        else:
            total_nodes_with_out_edges = len([
                node for node in self.graph.nodes() 
                if self.N_0.get(node, 0) > self.N_threshold
            ])
        
        if total_nodes_with_out_edges == 0:
            return 0.0
        
        # Calculate action
        for f, g in self.graph.edges():
            # Skip edges where source or target is filtered in low_reject mode
            if self.reject_mode == 'low_reject' and (f in self.filtered_nodes or g in self.filtered_nodes):
                continue
            
            N_0_f = self.N_0.get(f, 1.0)
            if N_0_f > self.N_threshold:
                w_fg = self.graph[f][g]['normalized_weight']
                V_f = potentials.get(f, 0.0)
                V_g = potentials.get(g, 0.0)
                action_numerator += w_fg * self.K(V_f - V_g)
        
        action = action_numerator / total_nodes_with_out_edges
        return action
    
    def _initialize_potentials_topological(self):
        """
        Initialize potentials based on topological sorting.
        For graphs with cycles, uses the condensation graph.
        
        Returns:
        --------
        dict
            Initial potentials for all nodes
        """
        print("\nInitializing potentials using topological sorting method...")
        potentials = {}
        nodes = list(self.graph.nodes())
        
        if not nx.is_directed_acyclic_graph(self.graph):
            sccs = list(nx.strongly_connected_components(self.graph))
            print(f"  Graph has cycles. Detected {len(sccs)} strongly connected components.")
            
            condensation = nx.condensation(self.graph)
            
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
            for node in nx.topological_sort(self.graph):
                predecessors = list(self.graph.predecessors(node))
                if not predecessors:
                    potentials[node] = 0.0
                else:
                    potentials[node] = min(potentials.get(p, 0) for p in predecessors) - 1.0
        
        # Normalize potentials to have zero mean
        mean_potential = np.mean(list(potentials.values()))
        for node in potentials:
            potentials[node] -= mean_potential
        
        return potentials
    
    def optimize_potentials(self, max_iter=20000, verbose=True):
        """
        Optimize potentials to minimize the action using L-BFGS-B algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations for optimization (default: 20000)
        verbose : bool
            Whether to print optimization progress (default: True)
        
        Returns:
        --------
        dict
            Optimized potentials mapping node expressions to potential values
        """
        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        # Initialize potentials
        initial_potentials_dict = self._initialize_potentials_topological()
        
        # Ensure all nodes have an initial potential
        for node in nodes:
            if node not in initial_potentials_dict:
                initial_potentials_dict[node] = 0.0
        
        v0 = np.array([initial_potentials_dict[node] for node in nodes])
        
        # Count nodes with outgoing edges
        if self.reject_mode == 'low_reject':
            total_nodes_with_out_edges = len([
                node for node in nodes 
                if node not in self.filtered_nodes and self.N_0.get(node, 0) > self.N_threshold
            ])
        else:
            total_nodes_with_out_edges = len([
                node for node in nodes 
                if self.N_0.get(node, 0) > self.N_threshold
            ])
        
        if total_nodes_with_out_edges == 0:
            if verbose:
                print("No nodes with outgoing edges to optimize.")
            return {node: 0.0 for node in nodes}
        
        # Pre-calculate edge data for efficiency
        edges = []
        for f, g in self.graph.edges():
            # Skip edges where source or target is filtered in low_reject mode
            if self.reject_mode == 'low_reject' and (f in self.filtered_nodes or g in self.filtered_nodes):
                continue
            
            N_0_f = self.N_0.get(f, 1.0)
            if N_0_f > self.N_threshold:
                w_fg = self.graph[f][g]['normalized_weight']
                f_idx = node_to_idx[f]
                g_idx = node_to_idx[g]
                edges.append({'f_idx': f_idx, 'g_idx': g_idx, 'w': w_fg})
        
        def objective_and_grad(v):
            """
            Calculate both the objective function (action) and its gradient.
            """
            action_numerator = 0
            grads = np.zeros_like(v)
            
            for edge in edges:
                f_idx, g_idx, w = edge['f_idx'], edge['g_idx'], edge['w']
                delta_v = v[f_idx] - v[g_idx]
                
                # Objective function component using K
                action_numerator += w * self.K(delta_v)
                
                # Gradient component using numerical differentiation of K
                eps = 1e-8
                K_grad = (self.K(delta_v + eps) - self.K(delta_v - eps)) / (2 * eps)
                common_grad_term = w * K_grad
                grads[f_idx] += common_grad_term
                grads[g_idx] -= common_grad_term
            
            action = action_numerator / total_nodes_with_out_edges
            normalized_grad = grads / total_nodes_with_out_edges
            
            return action, normalized_grad
        
        if verbose:
            print(f"Starting optimization with L-BFGS-B (max_iter={max_iter})...")
        
        result = minimize(
            fun=objective_and_grad,
            x0=v0,
            jac=True,  # Indicates that fun returns both objective and gradient
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': verbose}
        )
        
        if not result.success and verbose:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        optimized_v = result.x
        
        # Normalize potentials to have zero mean
        mean_potential = np.mean(optimized_v)
        optimized_v -= mean_potential
        
        optimized_potentials = {idx_to_node[i]: v for i, v in enumerate(optimized_v)}
        
        if verbose:
            print("Optimization finished.")
            final_action = self.calculate_action(optimized_potentials)
            print(f"Final action: {final_action:.6f}")
        
        return optimized_potentials


# Example usage
if __name__ == "__main__":
    # Example 1: Using high_reject mode
    print("=" * 80)
    print("Example 1: High Reject Mode")
    print("=" * 80)
    
    database_path = './data/word_database_gpt5-nano.json'
    
    builder_high = PotentialBuilder(database_path, reject_mode='high_reject')
    
    # Get the graph
    graph = builder_high.get_graph()
    print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Optimize potentials
    optimized_potentials_high = builder_high.optimize_potentials(max_iter=20000)
    
    # Calculate action with optimized potentials
    action_high = builder_high.calculate_action(optimized_potentials_high)
    print(f"\nFinal action (high_reject): {action_high:.6f}")
    
    print("\n" + "=" * 80)
    print("Example 2: Low Reject Mode")
    print("=" * 80)
    
    # Example 2: Using low_reject mode
    database_path = './data/ideasearchfitter_database.json'

    builder_low = PotentialBuilder(database_path, reject_mode='low_reject', N_threshold=1)
    
    # Optimize potentials
    optimized_potentials_low = builder_low.optimize_potentials(max_iter=20000)
    
    # Calculate action with optimized potentials
    action_low = builder_low.calculate_action(optimized_potentials_low)
    print(f"\nFinal action (low_reject): {action_low:.6f}")
    
    # Show some example potentials
    print("\n" + "=" * 80)
    print("Sample of optimized potentials (high_reject mode):")
    print("=" * 80)
    sample_nodes = list(optimized_potentials_high.keys())[:5]
    for node in sample_nodes:
        print(f"{node[:60]}... : {optimized_potentials_high[node]:.4f}")
