import numpy as np
import re
import math

# Default values for parameters if not provided
default_params = {
    'id_to_token': {
        0: 'sin', 1: 'cos', 2: 'tan', 3: 'arcsin', 4: 'arccos', 5: 'arctan',
        6: 'tanh', 7: 'log', 8: 'log10', 9: 'exp', 10: 'square', 11: 'sqrt',
        12: 'abs', 13: '*', 14: '**', 15: '/', 16: '+', 17: '-', 18: '1',
        19: '2', 20: 'pi', 21: 'log_v_k_nu', 22: 'param1', 23: 'param2',
        24: 'param3', 25: 'param4', 26: 'param5', 27: 'param6', 28: 'param7',
        29: 'param8', 30: 'param9', 31: '(', 32: ')', 33: ' '
    },
    'empty_input_potential': -1.0,
    'paren_penalty': 2.0,
    'extra_char_penalty': 0.3,
    'extra_char_threshold': 1,
    'length_penalty_divisor': 4.0,
    'max_depth_penalty': 0.5,
    'max_depth_threshold': 1,
    'func_penalty': 0.28,
    'div_pow_penalty': 0.5,
    'abs_penalty': 0.75,
    'trig_penalty': 0.75,
    'nested_expr_penalty': 1.0,
    'div_zero_risk_penalty': 0.9,
    'pow_risk_penalty': 0.5,
    'sqrt_risk_penalty': 0.2,
    'no_params_penalty': 1.0,
    'few_params_penalty': 0.5,
    'few_params_threshold': 3,
    'optimal_params_min': 3,
    'optimal_params_max': 6,
    'optimal_params_bonus': 0.6,
    'excess_params_penalty': 1.0,
    'excess_params_threshold': 6,
    'freq_var_weight': 0.05,
    'freq_var_cap': 8.0,
    'entropy_bonus': 0.6,
    'log_v_bonus': 1.1,
    'log_bonus': 0.6,
    'pattern_affinity_bonus': 1.0,
    'pattern_count_divisor': 6.0,
    'linear_logv_weight': 0.8,
    'centered_linear_weight': 0.65,
    'nonlinear_weight': 0.7,
    'exp_weight': 0.35,
    'proximity_cap': 1.8,
    'proximity_bonus': 0.9,
    'simple_bonus': 1.0,
    'simple_length_threshold': 50,
    'simple_func_threshold': 2,
    'short_bonus': 0.5,
    'short_length_threshold': 25,
    'max_energy': 3.0,
    'K': 1.0,
    'pattern_affinity_threshold': 0.85,
    'pattern_affinity_adjustment': 0.05,
    'min_potential': -1.0,
    'max_potential': 1.0,
    'nan_inf_default': 0.0,
    'overall_factor': 2.0
}

def potential_optimized_batch(token_ids_list: list, params: dict) -> np.ndarray:
    """
    Batch version of potential using numpy vectorization for speed.
    Follows the exact logic of potential() but processes multiple expressions at once.
    
    Args:
        token_ids_list: A list of token_id lists, each representing a mathematical expression.
        params: A dictionary containing the parameters for calculating the potential.
    
    Returns:
        A numpy array of potentials (energies) for all expressions.
    """
    # Use provided params, falling back to defaults
    p = {**default_params, **params}
    
    n = len(token_ids_list)
    if n == 0:
        return np.array([])
    
    id_to_token = p['id_to_token']
    
    # Pre-allocate arrays for vectorized operations
    potentials = np.zeros(n, dtype=np.float64)
    
    # Process each expression
    for i, token_ids in enumerate(token_ids_list):
        # Reconstruct expression string from tokens
        s = "".join(id_to_token.get(t, "") for t in token_ids)
        s = (s or "").strip()
        s_lower = s.lower()

        # 1) Input validity check
        if not s_lower:
            potentials[i] = p['empty_input_potential']
            continue

        # 2) Syntax completeness check
        depth = 0
        max_depth = 0
        bad_paren = False
        for ch in s:
            if ch == '(':
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    bad_paren = True
                    depth = 0
        if depth != 0:
            bad_paren = True

        # 3) Feature extraction
        funcs = re.findall(r'\b(?:exp|log|ln|log10|sqrt|tanh|sin|cos|tan|abs|pow|ceil|floor|log_v_k_nu)\b', s_lower)
        num_funcs = len(funcs)

        num_exp = s_lower.count('exp')
        num_log = s_lower.count('log') + s_lower.count('ln') + s_lower.count('log10')
        num_sqrt = s_lower.count('sqrt')
        num_abs = s_lower.count('abs')
        num_trig = s_lower.count('sin') + s_lower.count('cos') + s_lower.count('tan')
        num_div = s_lower.count('/')
        num_pow = s_lower.count('**') + s_lower.count('^')

        param_list = re.findall(r'\bparam\d+\b', s_lower)
        unique_params = sorted(set(param_list))
        num_params = len(unique_params)
        param_counts = [param_list.count(p_name) for p_name in unique_params]
        total_params = sum(param_counts)

        if num_params > 0:
            mean_params = total_params / num_params
            freq_var = sum((c - mean_params) ** 2 for c in param_counts) / num_params
            entropy = -sum((c / total_params) * math.log((c / total_params) + 1e-12) for c in param_counts) if total_params > 0 else 0.0
            entropy_norm = entropy / (math.log(num_params) + 1e-12) if num_params > 1 else 0.0
        else:
            freq_var, entropy_norm = 0.0, 0.0

        # Nikuradse-2 related structure recognition
        has_log_v = 'log_v_k_nu' in s_lower
        linear_logv = bool(re.search(r'\bparam\d+\s*\*\s*log_v_k_nu\b', s_lower))
        centered_linear = bool(re.search(r'\bparam\d+\s*\*\s*\(\s*log_v_k_nu\s*[-]\s*param\d+\s*\)', s_lower))
        logistic_present = len(re.findall(r'1\s*/\s*\(\s*1\s*\+\s*exp', s_lower)) > 0
        tanh_present = len(re.findall(r'\btanh\s*\(', s_lower)) > 0
        softplus_present = len(re.findall(r'log\s*\(\s*1\s*\+\s*exp', s_lower)) > 0

        pattern_count = int(has_log_v) + int(linear_logv) + int(centered_linear) + int(logistic_present) + int(tanh_present) + int(softplus_present)
        pattern_affinity = pattern_count / p['pattern_count_divisor']

        nested_expr = bool(re.search(r'exp\s*\(', s_lower)) or bool(re.search(r'log\s*\(', s_lower))

        div_zero_risk = '/' in s_lower
        pow_risk = num_pow > 0
        sqrt_risk = num_sqrt > 0 and not bool(re.search(r'sqrt\s*\(\s*abs', s_lower))

        # 4) Energy calculation and mapping to [-1, 1]
        energy = 0.0

        # Syntax completeness penalty
        if bad_paren:
            energy += p['paren_penalty']
        extra_chars = len(re.findall(r'[^0-9a-zA-Z_\+\-\*\/\^\.\(\),\s]', s_lower))
        energy += max(0, extra_chars - p['extra_char_threshold']) * p['extra_char_penalty']
        energy += math.log1p(len(s)) / p['length_penalty_divisor']
        energy += max(0, max_depth - p['max_depth_threshold']) * p['max_depth_penalty']

        # Basic function and operator complexity penalty
        energy += num_funcs * p['func_penalty']
        energy += (num_div + num_pow) * p['div_pow_penalty']
        energy += num_abs * p['abs_penalty']
        energy += num_trig * p['trig_penalty']

        # Risk penalty
        energy += p['nested_expr_penalty'] if nested_expr else 0.0
        energy += p['div_zero_risk_penalty'] if div_zero_risk else 0.0
        energy += p['pow_risk_penalty'] if pow_risk else 0.0
        energy += p['sqrt_risk_penalty'] if sqrt_risk else 0.0

        # Parameter diversity adjustment
        if num_params == 0:
            energy += p['no_params_penalty']
        elif num_params < p['few_params_threshold']:
            energy += p['few_params_penalty'] * (p['few_params_threshold'] - num_params)
        elif p['optimal_params_min'] <= num_params <= p['optimal_params_max']:
            energy -= p['optimal_params_bonus']
        else:
            energy += (num_params - p['excess_params_threshold'])

        energy += p['freq_var_weight'] * min(freq_var, p['freq_var_cap'])
        energy -= p['entropy_bonus'] * entropy_norm

        # Nikuradse-2 prior structure reward
        if has_log_v:
            energy -= p['log_v_bonus']
        elif num_log > 0:
            energy -= p['log_bonus']

        # Structure matching reward
        energy -= p['pattern_affinity_bonus'] * pattern_affinity

        # Structure similarity weighted penalty
        proximity_score = 0.0
        if has_log_v and num_params > 0:
            proximity_score = (
                p['linear_logv_weight'] * int(linear_logv)
                + p['centered_linear_weight'] * int(centered_linear)
                + p['nonlinear_weight'] * (int(logistic_present) + int(tanh_present) + int(softplus_present))
                + p['exp_weight'] * num_exp
            )
            proximity_score = min(p['proximity_cap'], proximity_score)
        energy -= p['proximity_bonus'] * proximity_score

        # Simplicity preference
        simple_pattern = re.compile(r'^[0-9a-zA-Z_\s\+\-\*\/\.\(\),]+$')
        truly_simple = bool(simple_pattern.match(s_lower)) and num_funcs <= p['simple_func_threshold'] and num_pow == 0
        if truly_simple and len(s) < p['simple_length_threshold']:
            energy -= p['simple_bonus']
        elif len(s) < p['short_length_threshold']:
            energy -= p['short_bonus']

        # Avoid unstable operations
        if '0' in s_lower and ('/' in s_lower or '**' in s_lower):
            energy += 0  # Final mapping

        if energy < 0:
            energy = 0.0
        max_energy = p['max_energy']
        if energy > max_energy:
            energy = max_energy

        K = p['K']
        norm = 1 - math.exp(-energy / K)
        val = -1 + 2 * norm

        # Fine-tuning for key pattern matching
        if pattern_affinity >= p['pattern_affinity_threshold'] and (logistic_present or tanh_present or softplus_present or has_log_v):
            val -= p['pattern_affinity_adjustment']

        if math.isnan(val) or math.isinf(val):
            val = p['nan_inf_default']
        val = max(p['min_potential'], min(p['max_potential'], val))
        
        potentials[i] = float(round(val, 5))*p['overall_factor']
    
    return potentials
