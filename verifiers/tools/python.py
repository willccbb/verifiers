def python(code: str) -> str:
    """Evaluates a block of Python code and returns output of print() statements. Allowed libraries: astropy, biopython, networkx, numpy, scipy, sympy.
    
    Args:
        code: A block of Python code

    Returns:
        The output of the code (truncated to 1000 chars) or an error message

    Examples:
        {"code": "import numpy as np; print(np.array([1, 2, 3]) + np.array([4, 5, 6]))"} -> "[5 7 9]"
        {"code": "import scipy; print(scipy.linalg.inv(np.array([[1, 2], [3, 4]])))"} -> "[[-2.   1. ] [ 1.5 -0.5]]"
        {"code": "import sympy; x, y = sympy.symbols('x y'); print(sympy.integrate(x**2, x))"} -> "x**3/3"
    """

    import subprocess
    try:
        # Run the code block in subprocess with 10-second timeout
        result = subprocess.run(
            ['python', '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            text=True
        )
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        output = result.stdout.strip() if result.stdout else ""
        if len(output) > 1000:
            output = output[:1000] + "... (truncated to 1000 chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"