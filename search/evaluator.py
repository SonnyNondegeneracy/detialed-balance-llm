import json
from collections import Counter
import networkx as nx
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Tuple
from safe import is_sandbox_safe
from pywheels import get_answer
import re
from config import filter

_ALLOWED_FUNCTIONS = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "tanh", "log", "log10", "exp", "square", "sqrt", "abs",
]
_OPERATORS = ['*', '**', '/', '+', '-']
_CONSTANTS = ['1', '2', 'pi']
_VARIABLES = ['log_v_k_nu'] + [f'param{i}' for i in range(1, 10)]
_SPECIAL_CHARS = ['(', ')', ' ']

VOCAB_DICT = {'sin': 0, 'cos': 1, 'tan': 2, 'arcsin': 3, 'arccos': 4, 'arctan': 5, 'tanh': 6, 'log': 7, 'log10': 8, 'exp': 9, 'square': 10, 'sqrt': 11, 'abs': 12, '*': 13, '**': 14, '/': 15, '+': 16, '-': 17, '1': 18, '2': 19, 'pi': 20, 'log_v_k_nu': 21, 'param1': 22, 'param2': 23, 'param3': 24, 'param4': 25, 'param5': 26, 'param6': 27, 'param7': 28, 'param8': 29, 'param9': 30, '(': 31, ')': 32, ' ': 33}

def evaluate(string,translator = 'gpt5-nano')->Tuple[float, str]:
    # 生成函数V
    translator_prompt = f"""
你要将提供给你的论述转换成一个Python函数potential(token_ids:list)->float。
这个函数计算一个表达式的潜力值，表达式由numexpression字符串拆分并转换成的token列表表示。
tokens列表中的每个token及对应的词汇表索引如下：
{VOCAB_DICT}
请确保生成的代码只包含函数定义和必要的导入语句，不要包含任何测试代码或主程序代码。
请严格按照以下格式生成代码：
```python
def potential(token_ids:list)->float:
    # 你的代码实现
```
请确保代码符合Python语法规范，并且能够正确处理输入的tokens列表，返回一个浮点数。
调用示例：
```python
tokens = [9, 31, 7, 32, 33, 13, 33, 24]  # 对应表达式 "exp(log_v_k_nu) * param3"
potential_value = potential(tokens)
print(potential_value)
>> 0.0  # 示例输出
```
这是给你的论述：
{string}
这是函数potential的表达式：
{filter(string)}
请生成代码，不要包含任何解释或额外信息。
"""
    code = get_answer(translator_prompt, model='gpt5-nano')
    code_pattern = r'```python(.*?)```'
    code = re.search(code_pattern, code, re.DOTALL).group(1).strip()

    # 安全检查
    if not is_sandbox_safe(code):
        return 0.0, "代码不安全，包含禁止的操作"

    name_space = {}
    exec(code, name_space)
    potential = name_space['potential']
    def V(f):
        """
        Hand-crafted potential function based on specific patterns in the function string.
        """
        # 首先将字符串拆成tokens列表
        token_pattern = r'sin|cos|tan|arcsin|arccos|arctan|tanh|log10|log|exp|square|sqrt|abs|\*\*|\*|/|\+|-|\(|\)|\s+|param[1-9]|log_v_k_nu|1|2|pi'
        tokens = re.findall(token_pattern, f)
        token_ids = [VOCAB_DICT[token] for token in tokens if token in VOCAB_DICT]
        return potential(token_ids)

    def K(x):
        """
        Kernel function for the action calculation.
        """
        z = -x
        return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))

    def calculate_action(graph, node_counts):
        """
        Calculates the action for a given graph and potential function.
        """
        action_numerator = 0
        
        for f, g in graph.edges():
            N = node_counts.get(f, 0)
            if N > 1:
                n = graph[f][g]['weight']
                w = n / N
                action_numerator += w * K(V(f) - V(g))
                
        total_nodes = len([node for node, count in node_counts.items() if count > 1])
        
        if total_nodes == 0:
            return 0
            
        action = action_numerator / total_nodes
        return action
    
    # 生成图
    file_path = './data/ideasearchfitter_database.json'
    
    # Using ijson to stream-read the large JSON file
    graph = nx.DiGraph()
    node_counts = Counter()

    # print(f"Processing data from {file_path}...")
    
    with open(file_path, 'r') as f:
        ideas = json.load(f)
        # print(f'loaded {len(ideas)} ideas from JSON file.')

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

    # print("Graph construction complete.")
    # print(f"Total nodes: {graph.number_of_nodes()}")
    # print(f"Total edges: {graph.number_of_edges()}")
    
    # 计算action
    action = calculate_action(graph, node_counts)
    # print(f"Calculated action: {action}")

    return max(100.0*(1.0-action), 0.0), f"成功计算action = {action}，函数为：" + code

if __name__ == "__main__":
    # 示例论述
    test_string = "函数的潜力值应基于其复杂性和使用的数学操作来评估。更复杂的函数应具有更高的潜力值，而简单的函数应具有较低的潜力值。"
    score, msg = evaluate(test_string)
    print(f"Score: {score}, Message: {msg}")