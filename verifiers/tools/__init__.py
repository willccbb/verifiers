from .ask import ask
from .calculator import calculator
from .search import search
from .python import python

# Import SmolaAgents tools when available
try:
    from .smolagents import CalculatorTool
    __all__ = ["ask", "calculator", "search", "python", "CalculatorTool"]
except ImportError:
    __all__ = ["ask", "calculator", "search", "python"] 