def calculator(expression: str) -> str:
    """Evaluates a single line of Python math expression. No imports or variables allowed.

    Args:
        expression: A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

    Returns:
        The result of the calculation or an error message

    Examples:
        "2 + 2" -> "4"
        "3 * (17 + 4)" -> "63"
        "100 / 5" -> "20.0"
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}" 