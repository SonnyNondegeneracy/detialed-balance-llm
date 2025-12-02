#################################
# safe_detection.py
# 本程序用于测试函数安全性
# Updated by Sonny 2025/10/18
#################################

import ast
from typing import Tuple
def is_sandbox_safe(code: str) -> bool:
    """
    计算代码是否沙盒安全
    code: str, Python代码字符串
    return: bool, True表示安全，False表示不安全
    """
    # 黑名单检测
    blacklist = [
        'import os',
        'import sys',
        'import subprocess',
        'from os',
        'from sys',
        'from subprocess',
        'eval(',
        'exec(',
        '__import__(',
        'open(',
        'input(',
        'while True',
        'for _ in iter(int, 1):',
    ]
    for forbidden in blacklist:
        if forbidden in code:
            return False
        
    # 解析代码为AST进行静态检测
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    
    class SafeVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name in ['os', 'sys', 'subprocess']:
                    raise ValueError("Unsafe import detected")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module in ['os', 'sys', 'subprocess']:
                raise ValueError("Unsafe import detected")
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'open', 'input', '__import__']:
                    raise ValueError("Unsafe function call detected")
            self.generic_visit(node)

        def visit_While(self, node):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                raise ValueError("Infinite loop detected")
            self.generic_visit(node)

        def visit_For(self, node):
            if (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == 'iter' and
                len(node.iter.args) == 2 and
                isinstance(node.iter.args[1], ast.Constant) and
                node.iter.args[1].value == 1):
                raise ValueError("Infinite loop detected")
            self.generic_visit(node)
    visitor = SafeVisitor()
    try:
        visitor.visit(tree)
    except ValueError:
        return False

    return True