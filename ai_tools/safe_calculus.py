import sympy as sp

def run_calculus(params: dict) -> dict:
    try:
        calc_type = params.get("type")
        expr_str = params.get("expression")
        var_str = params.get("variable")
        at = params.get("at", None)

        if not all([calc_type, expr_str, var_str]):
            return {"error": "Missing one or more required parameters: type, expression, variable."}

        # Create symbolic variable and expression
        x = sp.Symbol(var_str)
        expr = sp.sympify(expr_str)

        if calc_type == "derivative":
            result = sp.diff(expr, x)
            return {"result": str(result)}

        elif calc_type == "integral":
            if isinstance(at, list) and len(at) == 2:
                lower = sp.sympify(at[0])
                upper = sp.sympify(at[1])
                result = sp.integrate(expr, (x, lower, upper))
                return {"result": str(result), "definite": True, "bounds": [str(lower), str(upper)]}
            else:
                result = sp.integrate(expr, x)
                return {"result": str(result), "definite": False}

        elif calc_type == "limit":
            if at is None:
                return {"error": "Limit requires an 'at' parameter."}

            # Check direction: at can be a number, or {"value": x, "dir": "+" or "-"}
            if isinstance(at, dict):
                value = sp.sympify(at.get("value"))
                direction = at.get("dir", "+")
                if direction == "+":
                    dir_flag = "plus"
                elif direction == "-":
                    dir_flag = "minus"
                else:
                    return {"error": f"Invalid direction: {direction}"}
                result = sp.limit(expr, x, value, dir=dir_flag)
            else:
                value = sp.sympify(at)
                result = sp.limit(expr, x, value)

            return {"result": str(result), "at": str(value)}

        else:
            return {"error": f"Unknown calculus type: {calc_type}"}

    except Exception as e:
        return {"error": f"Failed to compute calculus expression: {str(e)}"}

EXPORT = {
    "run_calculus": {
        "help": (
            "Use this to compute calculus-based expressions such as:\n"
            "- Derivatives (e.g., slope of x^2 + 3x)\n"
            "- Integrals (e.g., area under sin(x) from 0 to pi)\n"
            "- Limits (e.g., limit of 1/x as x → 0)\n"
            "Required parameters:\n"
            "- type: one of 'derivative', 'integral', or 'limit'\n"
            "- expression: the formula\n"
            "- variable: the symbol (e.g., 'x')\n"
            "- at: value (for limits) or [lower, upper] bounds (for definite integrals)\n"
        ),
        "callable": run_calculus,
        "params": {
            "type": "derivative | integral | limit",
            "expression": "required formula",
            "variable": "symbol like x",
            "at": "point or [start, end] (optional depending on type)"
        }
    }
}
