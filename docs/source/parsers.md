# Parsers

Parsers extract structured information from model outputs. The framework provides convenience parsers for common use cases, but **for nontrivial environments, users will want to write their own parsers** to handle specific output formats and requirements.

## Parser Hierarchy

```
Parser (base class)
├── XMLParser      # Convenience: XML-tagged field extraction
├── ThinkParser    # Convenience: Extract content after </think> tags
└── SmolaParser    # Specialized for Smolagents tool format
```

## When to Use Built-in Parsers

The built-in parsers (`XMLParser`, `ThinkParser`) are designed for common use cases:

- **XMLParser**: When you need structured output with multiple fields
- **ThinkParser**: When you want step-by-step reasoning with a final answer
- **Base Parser**: When you just need raw text extraction

For more complex requirements, extend the base `Parser` class.

## Base Parser

The base `Parser` class provides the foundation for all parsers:

```python
from verifiers.parsers import Parser

class Parser:
    """Base parser class for extracting structured information from model outputs."""
    
    def parse(self, text: str) -> Any:
        """Parse text and return structured data. Default: return text as-is."""
        return text
    
    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the final answer from a completion."""
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            return self.parse(completion[-1]["content"])
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks format compliance."""
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0  # Default: always return 1.0
        return format_reward_func
```

## XMLParser: Structured Field Extraction

`XMLParser` is a convenience parser for extracting structured fields from XML-tagged output:

```python
from verifiers.parsers import XMLParser

# Define expected fields
parser = XMLParser(fields=["reasoning", "answer"])

# Parse model output
output = """
<reasoning>
To solve 2+2, I need to add two and two together.
Two plus two equals four.
</reasoning>
<answer>
4
</answer>
"""

parsed = parser.parse(output)
print(parsed.reasoning)  # "To solve 2+2, I need to add..."
print(parsed.answer)     # "4"
```

### Alternative Field Names

Support multiple names for the same field:

```python
# Accept either "reasoning" or "thinking" tags
parser = XMLParser(fields=[("reasoning", "thinking"), "answer"])

# Both formats work
output1 = "<thinking>...</thinking><answer>42</answer>"
output2 = "<reasoning>...</reasoning><answer>42</answer>"

parsed1 = parser.parse(output1)
parsed2 = parser.parse(output2)

# Access using the first name in the tuple
print(parsed1.reasoning)  # Works
print(parsed2.reasoning)  # Also works
```

### Format Enforcement

XMLParser provides a format reward function to encourage proper formatting:

```python
parser = XMLParser(fields=["reasoning", "answer"])

# Get the format reward function
format_func = parser.get_format_reward_func()

# Well-formatted output gets high reward
good_output = "<reasoning>Step by step...</reasoning><answer>42</answer>"
reward = format_func(good_output)  # 1.0

# Missing fields get lower reward
bad_output = "The answer is 42"
reward = format_func(bad_output)  # 0.0
```

## ThinkParser: Step-by-Step Reasoning

`ThinkParser` is a convenience parser for extracting content after `</think>` tags:

```python
from verifiers.parsers import ThinkParser

parser = ThinkParser()

# Model output: "<think>Let me calculate...</think>The answer is 4"
# parser.parse_answer(output) returns: "The answer is 4"

# With custom extraction function
def extract_boxed_answer(text):
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else text

parser = ThinkParser(extract_fn=extract_boxed_answer)
# Now extracts boxed answers: "The answer is 4" -> "4"
```

### Format Rewards

ThinkParser provides format rewards for proper `<think>` tag usage:

```python
parser = ThinkParser()
format_func = parser.get_format_reward_func()

# Proper format
good_output = "<think>Reasoning here</think>Final answer"
reward = format_func(good_output)  # 1.0

# Missing think tags
bad_output = "Just the answer"
reward = format_func(bad_output)  # 0.0
```

## Custom Parser Implementation

For nontrivial environments, you'll want to write your own parser by extending the base `Parser` class:

```python
from verifiers.parsers import Parser
import json
import re

class JSONParser(Parser):
    """Parse JSON-formatted responses."""
    
    def __init__(self, required_fields=None):
        super().__init__()
        self.required_fields = required_fields or []
    
    def parse(self, text: str) -> dict:
        """Extract JSON from text and validate required fields."""
        # Find JSON in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return {}
        
        try:
            parsed = json.loads(json_match.group())
            
            # Validate required fields
            for field in self.required_fields:
                if field not in parsed:
                    parsed[field] = None
                    
            return parsed
        except json.JSONDecodeError:
            return {}
    
    def get_format_reward_func(self):
        """Reward function for proper JSON formatting."""
        def format_reward_func(completion, **kwargs):
            try:
                # Check if completion contains valid JSON
                if isinstance(completion, str):
                    json_match = re.search(r'\{.*\}', completion, re.DOTALL)
                    if json_match:
                        json.loads(json_match.group())
                        return 1.0
                return 0.0
            except:
                return 0.0
        return format_reward_func

class CodeParser(Parser):
    """Parse code blocks from markdown."""
    
    def parse(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        code_match = re.search(r'```(?:\w+)?\n(.*?)\n```', text, re.DOTALL)
        return code_match.group(1) if code_match else text
    
    def get_format_reward_func(self):
        """Reward function for proper code block formatting."""
        def format_reward_func(completion, **kwargs):
            if isinstance(completion, str):
                return 1.0 if '```' in completion else 0.0
            return 0.0
        return format_reward_func
```

## Parser Integration with Rubrics

Always include format rewards from your parser in the rubric:

```python
from verifiers.rubrics import Rubric

parser = XMLParser(fields=["reasoning", "answer"])

def correct_answer(completion, answer, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer == answer else 0.0

rubric = Rubric(
    funcs=[
        correct_answer,                    # Task-specific correctness
        parser.get_format_reward_func()    # Format compliance
    ],
    weights=[0.8, 0.2],  # Format is important but not primary
    parser=parser
)
```

## Key Gotchas

1. **Format Rewards**: Always include `parser.get_format_reward_func()` in your rubric
2. **Error Handling**: Parsers should gracefully handle malformed input
3. **Answer Extraction**: Use `parse_answer()` for final answer extraction, `parse()` for full parsing
4. **Message Lists**: Parsers handle both strings and OpenAI message formats
5. **Custom Parsers**: For complex requirements, extend the base `Parser` class rather than trying to force built-in parsers

## Best Practices

1. **Start Simple**: Use built-in parsers for common cases
2. **Write Custom**: Create custom parsers for specific output formats
3. **Include Format Rewards**: Always include format compliance in evaluation
4. **Test Thoroughly**: Test parsers with various input formats
5. **Document Format**: Clearly document expected output format in system prompts