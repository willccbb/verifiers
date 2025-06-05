import ast, operator as op
from fastmcp import FastMCP

calc = FastMCP("calculator")
_OP = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
       ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
def _eval(n) -> float:                           
    if isinstance(n, ast.Constant):
        return float(n.value)
    if isinstance(n, ast.UnaryOp):
        return _OP[type(n.op)](_eval(n.operand)) # type: ignore
    if isinstance(n, ast.BinOp):
        return _OP[type(n.op)](_eval(n.left), _eval(n.right)) # type: ignore
    raise ValueError("bad expr")

@calc.tool(description="Safely evaluate an arithmetic expression.")
def evaluate(expression: str) -> float:
    """Return the numeric result, e.g. `"12/(2+1)" → 4.0`."""
    return _eval(ast.parse(expression, mode="eval").body) # type: ignore

if __name__ == "__main__":
    calc.run(port=8004, transport="streamable-http")