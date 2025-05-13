from typing import Dict, Any

# Import from smolagents package if available, otherwise use local stub
try:
    from smolagents.tools import Tool
except ImportError:
    # Create a stub Tool class for environments without smolagents
    class Tool:
        def __init__(self, *args, **kwargs):
            self.is_initialized = True
            
        def forward(self, *args, **kwargs):
            raise NotImplementedError("This is a stub - real implementation requires smolagents")


class CalculatorTool(Tool):
    """A calculator tool for evaluating mathematical expressions."""
    
    name = "calculator"
    description = "Evaluates a single line of Python math expression. No imports or variables allowed."
    inputs = {
        "expression": {
            "type": "string",
            "description": "A mathematical expression using only numbers and basic operators (+,-,*,/,**,())"
        }
    }
    output_type = "string"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = set("0123456789+-*/.() ")
        self.is_initialized = True
    
    def forward(self, expression: str) -> str:
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
        if not all(c in self.allowed for c in expression):
            return "Error: Invalid characters in expression"
        
        try:
            # Safely evaluate the expression with no access to builtins
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"