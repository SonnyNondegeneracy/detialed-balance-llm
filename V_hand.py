import re
import json
import os
from ideasearch_potential import potential_optimized_batch,default_params

_ALLOWED_FUNCTIONS = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "tanh", "log", "log10", "exp", "square", "sqrt", "abs",
]
_OPERATORS = ['*', '**', '/', '+', '-']
_CONSTANTS = ['1', '2', 'pi']
_VARIABLES = ['log_v_k_nu'] + [f'param{i}' for i in range(1, 10)]
_SPECIAL_CHARS = ['(', ')', ' ']

VOCAB_DICT = {'sin': 0, 'cos': 1, 'tan': 2, 'arcsin': 3, 'arccos': 4, 'arctan': 5, 'tanh': 6, 'log': 7, 'log10': 8, 'exp': 9, 'square': 10, 'sqrt': 11, 'abs': 12, '*': 13, '**': 14, '/': 15, '+': 16, '-': 17, '1': 18, '2': 19, 'pi': 20, 'log_v_k_nu': 21, 'param1': 22, 'param2': 23, 'param3': 24, 'param4': 25, 'param5': 26, 'param6': 27, 'param7': 28, 'param8': 29, 'param9': 30, '(': 31, ')': 32, ' ': 33}

# Load optimized parameters if available
def load_optimized_params():
    """Load optimized parameters from best_params.json if it exists."""
    params_file = os.path.join(os.path.dirname(__file__), 'best_params.json')
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
                return data.get('optimized_params', {})
        except Exception as e:
            print(f"Warning: Could not load optimized parameters: {e}")
            return {}
    return {}

# Load optimized parameters at module level
OPTIMIZED_PARAMS = load_optimized_params()

def V_hand(f: str, use_optimized: bool = True) -> float:
    """
    Hand-crafted potential function based on specific patterns in the function string.
    
    Args:
        f: The function string to evaluate
        use_optimized: If True, use optimized parameters; if False, use default parameters
    
    Returns:
        The potential value for the function
    """
    #  Build token sequence from ids, skipping unknowns and pure spaces
    token_pattern = r'sin|cos|tan|arcsin|arccos|arctan|tanh|log10|log|exp|square|sqrt|abs|\*\*|\*|/|\+|-|\(|\)|\s+|param[1-9]|log_v_k_nu|1|2|pi'
    tokens = re.findall(token_pattern, f)
    token_ids = [VOCAB_DICT[token] for token in tokens if token in VOCAB_DICT]
    
    # Choose parameters based on use_optimized flag
    params = OPTIMIZED_PARAMS if (use_optimized and OPTIMIZED_PARAMS) else {}
    
    # Calculate potential using potential_optimized_batch
    potential_values = potential_optimized_batch([token_ids], params)
    potential_value = potential_values[0]
    return potential_value

if __name__ == "__main__":
    test_functions = [
        "sin(param1) + cos(param2)",
        "exp(log_v_k_nu) * param3",
        "param4 ** 2 + 3 * param5",
        "sqrt(param6) - tan(param7)",
        "unknown_function(param8)"
    ]
    
    print("Testing V_hand function with optimized and default parameters:")
    print("=" * 80)
    
    if OPTIMIZED_PARAMS:
        print("✓ Optimized parameters loaded successfully")
        print(f"  Number of optimized parameters: {len(OPTIMIZED_PARAMS)}")
    else:
        print("⚠ No optimized parameters found, using defaults")
    
    print("\nTest Results:")
    print("-" * 80)
    
    for func in test_functions:
        p_optimized = V_hand(func, use_optimized=True)
        p_default = V_hand(func, use_optimized=False)
        diff = p_optimized - p_default
        
        print(f"\nFunction: {func}")
        print(f"  Optimized: {p_optimized:.4f}")
        print(f"  Default:   {p_default:.4f}")
        print(f"  Diff:      {diff:+.4f}")
