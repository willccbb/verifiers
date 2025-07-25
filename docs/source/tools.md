# Tools

Tools extend model capabilities by providing access to external functions like calculators, code execution, and search. The verifiers framework makes tool integration simple and reliable.

## Tool Architecture

Tools in verifiers are regular Python functions with:
1. **Clear signatures** - Type hints for parameters
2. **Descriptive docstrings** - Used for tool discovery
3. **String inputs/outputs** - For LLM compatibility
4. **Error handling** - Graceful failure messages

## Available Tools

The `verifiers.utils.tools` module provides several built-in tools:

### Calculator

```python
from verifiers.utils.tools import calculator

# Basic mathematical expressions
calculator("2 + 2")        # "4"
calculator("3 * (17 + 4)") # "63"
calculator("100 / 5")      # "20.0"
```

### Python Executor

```python
from verifiers.utils.tools import python

# Execute Python code safely
python("print('Hello, world!')")  # "Hello, world!"
python("import math; print(math.sqrt(16))")  # "4.0"

# Code execution with timeout protection
python("sum(range(1000000))")  # Returns result or timeout error
```

### Web Search

```python
from verifiers.utils.tools import search, search_ddg

# Search using Brave Search API
search("who invented the lightbulb")  # Returns formatted results with titles and snippets

# Alternative DuckDuckGo search
search_ddg("python programming tutorial", num_results=3)  # Returns concise summaries
```

### Ask Tool

```python
from verifiers.utils.tools import ask

# Ask questions about web pages
ask("What is the capital of France?", "https://en.wikipedia.org/wiki/France")
# Returns: "The capital of France is Paris."
```

## Tool Environment Usage

### Basic ToolEnv Pattern

```python
import verifiers as vf
from verifiers.utils.tools import python, calculator

# Create environment with tools
vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[python, calculator],
    system_prompt="""Think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to tools to help solve problems. Tools can be called with JSON:
<tool>
{"name": "python", "args": {"code": "print(2+2)"}}
</tool>""",
    max_turns=3
)
```

### Load Environment with Tools

Many environment modules include tools by default:

```python
# Math environment with Python tools included
vf_env = vf.load_environment("math-python")

# Smolagents integration with advanced tools
vf_env = vf.load_environment("smolagents-math-tools")
```

## Custom Tool Definition

### Basic Tool Template

```python
def my_calculator(expression: str) -> str:
    """Evaluate mathematical expressions.
    
    Args:
        expression: A mathematical expression to evaluate
        
    Returns:
        The result of the calculation as a string
        
    Examples:
        my_calculator("2 + 2") -> "4"
        my_calculator("sqrt(16)") -> "4.0"
        my_calculator("sin(pi/2)") -> "1.0"
    """
    try:
        import math
        
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

### File System Tool

```python
def read_file(filepath: str) -> str:
    """Read contents of a text file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        File contents as string or error message
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(filepath: str, content: str) -> str:
    """Write content to a text file.
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message or error message
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
```

## Tool Usage Patterns

### XML-Based Tool Calling

Models invoke tools using XML format in ToolEnv:

```xml
<think>
I need to calculate the compound interest. Let me use Python for this.
</think>

<tool>
{
    "name": "python",
    "args": {
        "code": "principal = 1000\nrate = 0.05\ntime = 10\namount = principal * (1 + rate) ** time\nprint(f'Final amount: ${amount:.2f}')"
    }
}
</tool>
```

The environment automatically:
1. Parses tool invocations from XML tags
2. Validates JSON arguments
3. Executes tools safely with error handling
4. Returns results to continue the conversation

### Native Tool Calling

For models with native tool calling support, use the appropriate parser:

```python
# Enable auto tool choice in vLLM with appropriate parser
sampling_args = {
    "extra_body": {
        "tool_choice": "auto"
    }
}

vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[python, calculator],
    sampling_args=sampling_args
)
```

## Advanced Tool Patterns

### Stateful Tools

Some tools need to maintain state across calls:

```python
class PythonSession:
    """Stateful Python interpreter."""
    
    def __init__(self):
        import builtins
        # Create safe builtins for code execution
        safe_builtins = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'len': len, 'range': range, 'enumerate': enumerate,
            'print': print, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        }
        
        self.namespace = {
            '__builtins__': safe_builtins,
            'math': __import__('math'),
            'numpy': __import__('numpy'),
        }
    
    def execute(self, code: str) -> str:
        """Execute code in persistent namespace."""
        try:
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, self.namespace)
            
            return output.getvalue().strip() or "Code executed successfully"
            
        except Exception as e:
            return f"Error: {str(e)}"

# Create tool function that uses the session
python_session = PythonSession()

def stateful_python(code: str) -> str:
    """Execute Python code with persistent state across calls."""
    return python_session.execute(code)
```

### Tool with External Dependencies

```python
def weather_lookup(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: City name or coordinates
        
    Returns:
        Weather information as formatted string
    """
    try:
        import requests
        import os
        
        api_key = os.getenv('WEATHER_API_KEY')
        if not api_key:
            return "Error: Weather API key not configured"
            
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        
        return f"Weather in {location}: {temp}°C, {desc}"
        
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
```

### Parallel Tool Support

ToolEnv supports parallel tool calls by default:

```python
# Models can call multiple tools simultaneously
tools_xml = """
<tool>
{"name": "calculator", "args": {"expression": "2 + 2"}}
</tool>

<tool>
{"name": "python", "args": {"code": "print('Hello from Python!')"}}
</tool>
"""

# Both tools execute in parallel and results are returned together
```

## Tool Integration with Environment Modules

### Creating Tool-Enabled Environments

```python
# math_with_tools.py
import verifiers as vf
from verifiers.utils.tools import python, calculator

def load_environment(**kwargs):
    dataset = vf.load_example_dataset("math", split="train")
    
    system_prompt = """Solve math problems step by step. You have access to Python and calculator tools.

Think step-by-step inside <think>...</think> tags, then either:
- Call a tool: <tool>{"name": "python", "args": {"code": "..."}}</tool>
- Give your answer: <answer>42</answer>"""

    parser = vf.XMLParser(fields=["think", "tool", "answer"], answer_field="answer")
    
    def correct_answer_func(parser, completion, answer) -> float:
        response = parser.parse_answer(completion) or ''
        return 1.0 if response.strip() == answer.strip() else 0.0
    
    rubric = vf.Rubric(
        funcs=[correct_answer_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser
    )
    
    return vf.ToolEnv(
        dataset=dataset,
        tools=[python, calculator],
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=5,
        **kwargs
    )
```

### Tool Rubrics

Use `ToolRubric` to track tool usage:

```python
from verifiers.utils.tools import python, calculator

# Create rubric that tracks tool calls
tool_rubric = vf.ToolRubric(tools=[python, calculator])

# Combine with other rubrics
combined_rubric = vf.RubricGroup([
    main_rubric,      # Task-specific rewards
    tool_rubric       # Tool usage tracking
])

vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[python, calculator],
    rubric=combined_rubric
)
```

## Tool Security and Sandboxing

### Code Execution Safety

The built-in `python` tool includes several safety measures:

- **Timeout protection**: Code execution times out after 10 seconds
- **Limited imports**: Only safe modules are available
- **Restricted builtins**: Dangerous functions are not accessible
- **Process isolation**: Each execution is isolated

### Custom Sandboxing

For additional safety, consider:

```python
def secure_python(code: str) -> str:
    """Execute Python with additional security measures."""
    
    # Filter dangerous keywords
    dangerous = ['import os', 'import sys', 'exec', 'eval', '__import__']
    if any(danger in code for danger in dangerous):
        return "Error: Potentially unsafe code detected"
    
    # Use the built-in safe python executor
    from verifiers.utils.tools import python
    return python(code)
```

## MCP Server Integration

For heavy computational resources, we recommend hosting tools as standalone servers:

```python
import requests

def mcp_calculator(expression: str) -> str:
    """Calculator via MCP server."""
    try:
        response = requests.post(
            "http://localhost:8080/calculate",
            json={"expression": expression},
            timeout=30
        )
        return response.json()["result"]
    except Exception as e:
        return f"Error: {str(e)}"

# Use lightweight wrapper in ToolEnv
vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[mcp_calculator],  # Lightweight wrapper
    max_turns=5
)
```

## XMLParser Tool Integration

For custom tool calling formats:

```python
# See environments/xml_tool_env for complete example
parser = vf.XMLParser(fields=["reasoning", "tool_call", "answer"])

def tool_reward_func(parser, completion) -> float:
    """Reward proper tool call formatting."""
    parsed = parser.parse(completion)
    if hasattr(parsed, 'tool_call') and parsed.tool_call:
        try:
            import json
            json.loads(parsed.tool_call)  # Valid JSON
            return 1.0
        except:
            return 0.0
    return 1.0  # No tool call needed
```

## Best Practices

1. **Error Handling**: Always return descriptive error messages as strings
2. **Timeouts**: Include timeout protection for long-running operations
3. **Type Conversion**: Convert all outputs to strings for LLM compatibility
4. **Documentation**: Write clear docstrings with examples
5. **Security**: Validate inputs and restrict dangerous operations
6. **Parallel Support**: Design tools to work independently for parallel execution
7. **Resource Management**: Use lightweight wrappers for heavy computational tools

## TODO Sections

TODO: Add documentation for:
- Advanced tool composition patterns
- Integration with different MCP servers
- Custom tool calling parsers for non-XML formats
- Performance optimization for tool-heavy workflows