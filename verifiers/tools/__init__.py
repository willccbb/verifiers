from .ask import ask
from .calculator import calculator
from .python import python
from .search import search

# Import SmolaAgents tools when available
try:
    from .smolagents import CalculatorTool

    __all__ = ["ask", "calculator", "search", "python", "CalculatorTool"]
except ImportError:
    __all__ = ["ask", "calculator", "search", "python"]
