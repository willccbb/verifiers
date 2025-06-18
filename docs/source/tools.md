# Tools

Tools extend model capabilities by providing access to external functions like calculators, code execution, and search. The verifiers framework makes tool integration simple and reliable.

## Tool Architecture

Tools in verifiers are regular Python functions with:
1. **Clear signatures** - Type hints for parameters
2. **Descriptive docstrings** - Used for tool discovery
3. **String inputs/outputs** - For LLM compatibility
4. **Error handling** - Graceful failure messages

## Basic Tool Definition

```python
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions.
    
    Args:
        expression: A mathematical expression to evaluate
        
    Returns:
        The result of the calculation as a string
        
    Examples:
        calculator("2 + 2") -> "4"
        calculator("sqrt(16)") -> "4.0"
        calculator("sin(pi/2)") -> "1.0"
    """
    try:
        import math
        import numpy as np
        
        # Create safe namespace with math functions
        namespace = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
            'log': math.log,
            'exp': math.exp,
            'abs': abs,
            'round': round,
        }
        
        # Evaluate expression
        result = eval(expression, namespace)
        return str(result)
        
    except Exception as e:
        return f"Error: {str(e)}"
```

## Tool Integration Pattern

### 1. Define Tools

```python
def python_executor(code: str) -> str:
    """Execute Python code and return output.
    
    Args:
        code: Python code to execute
        
    Returns:
        Output from code execution or error message
    """
    try:
        import io
        import contextlib
        
        # Capture output
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {'__builtins__': safe_builtins})
        
        return output.getvalue().strip()
        
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"

def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Search results as formatted text
    """
    # Implementation details...
    return formatted_results
```

### 2. Create Tool Environment

```python
from verifiers.envs import ToolEnv
from verifiers.parsers import XMLParser

# Define environment with tools
env = ToolEnv(
    dataset=dataset,
    tools=[calculator, python_executor, web_search],
    parser=XMLParser(["reasoning", ("tool", "answer")]),
    system_prompt=TOOL_SYSTEM_PROMPT
)
```

### 3. Tool Usage Format

Models invoke tools using XML format:

```xml
<reasoning>
I need to calculate the compound interest. Let me use Python for this.
</reasoning>

<tool>
{
    "name": "python_executor",
    "arguments": {
        "code": "principal = 1000\nrate = 0.05\ntime = 10\namount = principal * (1 + rate) ** time\nprint(f'Final amount: ${amount:.2f}')"
    }
}
</tool>
```

The environment automatically:
1. Parses tool invocations
2. Validates arguments
3. Executes tools safely
4. Returns results in `<result>` tags

## Advanced Tool Patterns

### Stateful Tools

Some tools need to maintain state across calls:

```python
class PythonSession:
    """Stateful Python interpreter."""
    
    def __init__(self):
        self.namespace = {
            '__builtins__': safe_builtins,
            'numpy': np,
            'pandas': pd,
        }
    
    def execute(self, code: str) -> str:
        """Execute code in persistent namespace."""
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, self.namespace)
            
            return output.getvalue().strip() or "Code executed successfully"
            
        except Exception as e:
            return f"Error: {str(e)}"

# Create tool function that uses the session
python_session = PythonSession()

def python(code: str) -> str:
    """Execute Python code with persistent state."""
    return python_session.execute(code)
```

### Async Tools

For tools that need async operations:

```python
import asyncio
import aiohttp

async def async_web_search(query: str) -> str:
    """Asynchronously search the web."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.search.com/search",
            params={"q": query}
        ) as response:
            data = await response.json()
            return format_search_results(data)

# Wrapper for sync environments
def web_search(query: str) -> str:
    """Search the web (sync wrapper)."""
    return asyncio.run(async_web_search(query))
```

### Composite Tools

Tools that combine multiple operations:

```python
def analyze_data(csv_data: str, analysis_type: str = "summary") -> str:
    """Analyze CSV data with various methods.
    
    Args:
        csv_data: CSV formatted data
        analysis_type: Type of analysis (summary, correlation, plot)
        
    Returns:
        Analysis results
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_data))
        
        if analysis_type == "summary":
            return df.describe().to_string()
            
        elif analysis_type == "correlation":
            return df.corr().to_string()
            
        elif analysis_type == "plot":
            # Create visualization
            fig, ax = plt.subplots()
            df.plot(ax=ax)
            
            # Save to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            
            return f"[Plot saved as base64: {img_base64[:50]}...]"
            
        else:
            return f"Unknown analysis type: {analysis_type}"
            
    except Exception as e:
        return f"Error analyzing data: {str(e)}"
```

## Tool Safety

### Input Validation

```python
def safe_calculator(expression: str) -> str:
    """Calculator with input validation."""
    # Whitelist allowed characters
    allowed = set("0123456789+-*/().,sqrtsincostan pie")
    if not all(c in allowed or c.isspace() for c in expression):
        return "Error: Invalid characters in expression"
    
    # Limit expression length
    if len(expression) > 100:
        return "Error: Expression too long"
    
    # Prevent dangerous operations
    dangerous = ["import", "exec", "eval", "__"]
    if any(d in expression.lower() for d in dangerous):
        return "Error: Unsafe operation"
    
    return calculator(expression)
```

### Resource Limits

```python
import signal
import resource

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def limited_python_executor(code: str, timeout: int = 5) -> str:
    """Execute Python with resource limits."""
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Limit memory (100MB)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (100 * 1024 * 1024, 100 * 1024 * 1024)
        )
        
        # Execute code
        result = python_executor(code)
        
        # Cancel timeout
        signal.alarm(0)
        
        return result
        
    except TimeoutError:
        return "Error: Execution timed out"
    except MemoryError:
        return "Error: Memory limit exceeded"
    finally:
        signal.alarm(0)
```

### Sandboxing

```python
def create_safe_namespace():
    """Create restricted execution namespace."""
    safe_builtins = {
        'abs': abs,
        'bool': bool,
        'dict': dict,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'max': max,
        'min': min,
        'print': print,
        'range': range,
        'round': round,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        # Explicitly exclude dangerous functions
        # No: exec, eval, compile, open, __import__
    }
    
    return {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        '__doc__': None,
    }
```

## Tool Documentation

Good documentation helps models use tools effectively:

```python
def data_analyzer(
    data: str,
    operation: str = "describe",
    columns: str = None,
    group_by: str = None
) -> str:
    """Analyze tabular data with pandas operations.
    
    Performs statistical analysis on CSV-formatted data.
    
    Args:
        data: CSV-formatted data with header row
        operation: Analysis operation to perform
            - "describe": Summary statistics
            - "correlate": Correlation matrix
            - "group": Group by aggregation
            - "pivot": Pivot table
        columns: Comma-separated column names to analyze (optional)
        group_by: Column name to group by (for group operation)
        
    Returns:
        Analysis results as formatted text
        
    Examples:
        # Basic statistics
        data_analyzer("x,y\\n1,2\\n3,4", "describe")
        
        # Correlation between specific columns
        data_analyzer(data, "correlate", columns="price,quantity")
        
        # Group aggregation
        data_analyzer(data, "group", group_by="category")
        
    Notes:
        - Data must be valid CSV format
        - Column names are case-sensitive
        - Returns error message if operation fails
    """
    # Implementation...
```

## Tool Execution Flow

### 1. Discovery Phase

The environment provides tool schemas to the model:

```python
def get_tool_schemas(tools):
    """Generate schemas from tool functions."""
    schemas = []
    
    for tool in tools:
        schema = {
            "name": tool.__name__,
            "description": tool.__doc__.split('\n')[0],
            "parameters": extract_parameters(tool),
            "examples": extract_examples(tool.__doc__)
        }
        schemas.append(schema)
    
    return schemas
```

### 2. Invocation Phase

Model generates tool calls:

```xml
<tool>
{
    "name": "calculator",
    "arguments": {
        "expression": "2 * 3 + 4"
    }
}
</tool>
```

### 3. Execution Phase

Environment executes tools:

```python
def execute_tool_call(tool_call, tools_dict):
    """Execute a parsed tool call."""
    try:
        tool_name = tool_call["name"]
        arguments = tool_call.get("arguments", {})
        
        if tool_name not in tools_dict:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool_func = tools_dict[tool_name]
        result = tool_func(**arguments)
        
        return result
        
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

### 4. Result Integration

Results are provided back to the model:

```xml
<result>
10
</result>
```

## Best Practices

### 1. Clear Function Signatures

```python
# Good: Clear types and names
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    frequency: int = 1
) -> str:
    """Calculate compound interest."""
    pass

# Bad: Ambiguous parameters
def calc(p, r, t, n=1):
    """Calculate interest."""
    pass
```

### 2. Informative Error Messages

```python
def robust_tool(input_data: str) -> str:
    """Process data with detailed error reporting."""
    if not input_data:
        return "Error: No input data provided"
    
    if len(input_data) > 10000:
        return f"Error: Input too large ({len(input_data)} chars, max 10000)"
    
    try:
        result = process_data(input_data)
        return result
    except ValueError as e:
        return f"Error: Invalid data format - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error - {type(e).__name__}: {str(e)}"
```

### 3. Consistent Output Formats

```python
def search_tool(query: str) -> str:
    """Search with consistent output format."""
    results = perform_search(query)
    
    # Always return structured format
    output = f"Found {len(results)} results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        output += f"{i}. {result['title']}\n"
        output += f"   URL: {result['url']}\n"
        output += f"   Summary: {result['summary']}\n\n"
    
    return output
```

### 4. Tool Composition

```python
# Tools can use other tools
def research_assistant(topic: str) -> str:
    """Research a topic using multiple tools."""
    # Search for information
    search_results = web_search(topic)
    
    # Extract key points
    extraction_prompt = f"Extract key facts from:\n{search_results}"
    facts = python_executor(f"print('{extraction_prompt}')")
    
    # Generate summary
    summary = summarize_text(facts)
    
    return f"Research Summary:\n{summary}\n\nSources:\n{search_results}"
```

## Tool Evaluation

Use ToolRubric to evaluate tool usage:

```python
from verifiers.rubrics import ToolRubric

rubric = ToolRubric(
    tools=[calculator, python_executor, web_search],
    weights=[0.5, 0.3, 0.2]  # Weights for different aspects
)

# Automatically evaluates:
# - Correct tool selection
# - Successful execution
# - Appropriate usage frequency
# - Error handling
```

Tools are a powerful way to extend model capabilities while maintaining safety and reliability. By following these patterns, you can create tools that are easy for models to discover and use effectively.