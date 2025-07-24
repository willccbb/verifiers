import ast
import subprocess
import textwrap


def _jupyterize(src: str) -> str:
    src = textwrap.dedent(src)
    tree = ast.parse(src, mode="exec")

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Extract the last expression
        last = tree.body.pop()
        body_code = ast.unparse(ast.Module(tree.body, []))
        expr_code = ast.unparse(last.value)  # type: ignore
        # Display the last expression value like Jupyter does
        return f"{body_code}\n_ = {expr_code}\nif _ is not None: print(_)"
    else:
        return src


def python(code: str) -> str:
    """Evaluates Python code like a Jupyter notebook cell, returning output and/or the last expression value.

    Args:
        code (str): A block of Python code

    Returns:
        The output (stdout + last expression if not None) or error message

    Examples:
        {"code": "import numpy as np\nnp.array([1, 2, 3]) + np.array([4, 5, 6])"} -> "[5 7 9]"
        {"code": "x = 5\ny = 10\nx + y"} -> "15"
        {"code": "a, b = 3, 4\na, b"} -> "(3, 4)"
    """

    try:
        # Run the code block in subprocess with 10-second timeout
        result = subprocess.run(
            ["python", "-c", _jupyterize(code)],
            timeout=10,
            text=True,
            capture_output=True,
        )

        output = result.stdout.strip()[:1000]
        error = result.stderr.strip()[:1000]
        if error:
            return error
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"
