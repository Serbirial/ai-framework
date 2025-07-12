import ast
import operator
from log import log

# Supported operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Num):  # For Python <3.8
        return node.n
    elif isinstance(node, ast.Constant):  # For Python >=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    else:
        raise ValueError("Only basic math expressions are allowed.")

def do_safe_math(parameters: dict) -> dict:
    """
    Safely evaluate a math expression using limited AST parsing.
    
    Args:
        parameters: A dict with key "expression", which is a math string like "2 + 3 * 4"
    
    Returns:
        A dict with the result or an error.
    """
    expr = parameters.get("expression")
    if not expr or not isinstance(expr, str):
        return {"error": "Missing or invalid 'expression' parameter."}
    log("SAFE-MATH-INPUT", expr)
    try:
        tree = ast.parse(expr, mode='eval')
        result = _eval_node(tree.body)
        log("SAFE-MATH-OUTPUT", result)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


EXPORT = {
    "execute_math": {
        "help": "Use this to execute and run math EXLCUSIVELY using +, -, *, /, %, //, and **.",
        "callable": do_safe_math,
        "params": {"expression": "23 + (7 * 2) / 3"}
    }}