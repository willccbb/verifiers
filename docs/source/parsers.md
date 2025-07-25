# Parsers

Parsers extract structured information from model outputs. The framework provides convenience parsers for common use cases, but **for nontrivial environments, users will want to write their own parsers** to handle specific output formats and requirements.

## Parser Hierarchy

```
Parser (base class)
├── XMLParser      # Structured XML field extraction
└── ThinkParser    # Extract content after </think> tags
```

## When to Use Built-in Parsers

The built-in parsers are designed for common use cases:

- **Parser**: Returns text as-is (default for simple cases)
- **XMLParser**: When you need structured output with multiple fields
- **ThinkParser**: When you want step-by-step reasoning with a final answer

For more complex requirements, extend the base `Parser` class.

## Base Parser

The base `Parser` class provides the foundation for all parsers:

```python
import verifiers as vf
from typing import Any, List, Dict, Callable, Union

class Parser:
    """Base parser class for extracting structured information from model outputs."""
    
    def parse(self, text: str) -> Any:
        """Parse text and return structured data. Default: return text as-is."""
        return text
    
    def parse_answer(self, completion: Union[str, List[Dict[str, str]]]) -> str | None:
        """Extract the final answer from a completion.
        
        Args:
            completion: Either a string (completion format) or list of messages (chat format)
            
        Returns:
            Extracted answer string or None if not found
        """
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            # For chat format, parse the last message's content
            return self.parse(completion[-1]["content"])
    
    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks format compliance."""
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0  # Default: always return 1.0
        return format_reward_func
    
    def get_assistant_messages(self, completion: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'assistant']

# Basic usage
parser = vf.Parser()
result = parser.parse("Hello, world!")  # Returns: "Hello, world!"
```

## XMLParser: Structured Field Extraction

`XMLParser` is a convenience parser for extracting structured fields from XML-tagged output:

```python
import verifiers as vf
from typing import List, Union, Tuple

# Define expected fields
parser = vf.XMLParser(fields=["reasoning", "answer"])

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
parser = vf.XMLParser(fields=[("reasoning", "thinking"), "answer"])

# Both formats work:
# <reasoning>...</reasoning> OR <thinking>...</thinking>
# Parsed as: parsed.reasoning and parsed.thinking (both accessible)
```

### Answer Field Specification

Specify which field contains the final answer:

```python
parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")

# parse_answer() will extract from the "answer" field
final_answer = parser.parse_answer(completion)
```

### Format Reward Function

XMLParser provides built-in format checking:

```python
parser = vf.XMLParser(fields=["reasoning", "answer"])

# Get format reward function
format_reward = parser.get_format_reward_func()

# Use in rubric
rubric = vf.Rubric(funcs=[
    correct_answer_func,
    format_reward  # Rewards proper XML formatting
], weights=[1.0, 0.2])
```

### Formatting Output

Generate properly formatted XML:

```python
parser = vf.XMLParser(fields=["reasoning", "answer"])

# Format structured data into XML
formatted = parser.format(
    reasoning="I need to calculate 2+2",
    answer="4"
)
print(formatted)
# <reasoning>
# I need to calculate 2+2
# </reasoning>
# <answer>
# 4
# </answer>
```

## ThinkParser: Step-by-Step Reasoning

`ThinkParser` extracts content after `</think>` tags for step-by-step reasoning:

```python
import verifiers as vf

parser = vf.ThinkParser()

# Model output with thinking
output = """
<think>
Let me work through this step by step.
2 + 2 = 4
So the answer is 4.
</think>
The answer is 4.
"""

# Extract final answer (content after </think>)
result = parser.parse(output)
print(result)  # "The answer is 4."

# parse_answer does the same thing
answer = parser.parse_answer(output)
print(answer)  # "The answer is 4."
```

### With Custom Extraction

Apply custom extraction functions:

```python
from verifiers.utils.data_utils import extract_boxed_answer

# Extract boxed answers from think parser output  
parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

output = """
<think>
2 + 2 = 4
</think>
The answer is \\boxed{4}.
"""

result = parser.parse(output)  # "4" (extracted from \\boxed{})
```

### Format Reward Function

ThinkParser validates the `<think>...</think>` format:

```python
parser = vf.ThinkParser()

# Get format reward function
format_reward = parser.get_format_reward_func()

# This checks that each assistant message:
# 1. Starts with <think>
# 2. Has exactly one <think> and one </think>
# 3. Has content after </think>

# Use in rubric
rubric = vf.Rubric(funcs=[
    correct_answer_func,
    format_reward  # Rewards proper <think> formatting
], weights=[1.0, 0.2])
```

## Custom Parser Patterns

### Custom Field Extraction

```python
import re
import verifiers as vf

class CodeParser(vf.Parser):
    """Extract code blocks from model output."""
    
    def parse(self, text: str) -> dict:
        """Extract all code blocks."""
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for language, code in matches:
            code_blocks.append({
                'language': language or 'text',
                'code': code.strip()
            })
        
        return {
            'code_blocks': code_blocks,
            'num_blocks': len(code_blocks)
        }
    
    def parse_answer(self, completion):
        """Extract the last code block as the answer."""  
        parsed = self.parse(self.get_text_content(completion))
        if parsed['code_blocks']:
            return parsed['code_blocks'][-1]['code']
        return None
    
    def get_text_content(self, completion):
        """Helper to extract text from completion."""
        if isinstance(completion, str):
            return completion
        else:
            return completion[-1]["content"]

# Usage
parser = CodeParser()
result = parser.parse("Here's the code:\n```python\nprint('hello')\n```")
print(result['code_blocks'][0]['code'])  # "print('hello')"
```

### Multi-Step Parser

```python
import verifiers as vf

class MathStepParser(vf.Parser):
    """Parse mathematical step-by-step solutions."""
    
    def parse(self, text: str) -> dict:
        """Extract numbered steps and final answer."""
        import re
        
        # Extract steps (lines starting with numbers)
        step_pattern = r'^\d+\.\s*(.*?)$'
        steps = re.findall(step_pattern, text, re.MULTILINE)
        
        # Extract final answer (content in \\boxed{})
        answer_pattern = r'\\boxed\{([^}]+)\}'
        answer_match = re.search(answer_pattern, text)
        final_answer = answer_match.group(1) if answer_match else None
        
        return {
            'steps': steps,
            'num_steps': len(steps),
            'final_answer': final_answer,
            'has_final_answer': final_answer is not None
        }
    
    def parse_answer(self, completion):
        """Extract the final answer."""
        parsed = self.parse(self.get_text_content(completion))
        return parsed['final_answer']
    
    def get_format_reward_func(self):
        """Reward proper step format and final answer."""
        def format_reward(completion, **kwargs):
            text = self.get_text_content(completion)
            parsed = self.parse(text)
            
            # Reward having steps and final answer
            has_steps = parsed['num_steps'] > 0
            has_answer = parsed['has_final_answer']
            
            if has_steps and has_answer:
                return 1.0
            elif has_steps or has_answer:
                return 0.5
            else:
                return 0.0
        
        return format_reward
    
    def get_text_content(self, completion):
        """Helper to extract text from completion."""
        if isinstance(completion, str):
            return completion
        else:
            return completion[-1]["content"]

# Usage with environment
parser = MathStepParser()
rubric = vf.Rubric(funcs=[
    correct_answer_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.3])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

### JSON Parser

```python
import json
import verifiers as vf

class JSONParser(vf.Parser):
    """Parse JSON output from models."""
    
    def __init__(self, required_fields=None):
        super().__init__()
        self.required_fields = required_fields or []
    
    def parse(self, text: str) -> dict:
        """Extract and parse JSON from text."""
        import re
        
        # Find JSON blocks
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON-like content
            json_text = text.strip()
        
        try:
            parsed_json = json.loads(json_text)
            return {
                'valid': True,
                'data': parsed_json,
                'error': None
            }
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'data': None,
                'error': str(e)
            }
    
    def parse_answer(self, completion):
        """Extract specific field as answer."""
        parsed = self.parse(self.get_text_content(completion))
        if parsed['valid'] and 'answer' in parsed['data']:
            return str(parsed['data']['answer'])
        return None
    
    def get_format_reward_func(self):
        """Reward valid JSON with required fields."""
        def format_reward(completion, **kwargs):
            text = self.get_text_content(completion)
            parsed = self.parse(text)
            
            if not parsed['valid']:
                return 0.0
            
            # Check required fields
            data = parsed['data']
            missing_fields = [f for f in self.required_fields if f not in data]
            
            if missing_fields:
                return 0.5  # Partial credit for valid JSON
            else:
                return 1.0  # Full credit for valid JSON with all fields
        
        return format_reward
    
    def get_text_content(self, completion):
        """Helper to extract text from completion."""
        if isinstance(completion, str):
            return completion
        else:
            return completion[-1]["content"]

# Usage
parser = JSONParser(required_fields=['reasoning', 'answer', 'confidence'])

# Expects output like:
# ```json
# {
#   "reasoning": "I need to calculate 2+2",
#   "answer": "4", 
#   "confidence": 0.95
# }
# ```
```

## Parser Integration with Environments

### Using Parsers in Environment Modules

```python
# math_environment.py
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

def load_environment(**kwargs):
    dataset = vf.load_example_dataset("math", split="train")
    
    # Choose parser based on requirements
    parser = vf.Parser(extract_fn=extract_boxed_answer)  # Simple extraction
    # OR
    parser = vf.XMLParser(fields=["reasoning", "answer"])  # Structured output
    # OR  
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)  # Step-by-step
    
    def correct_answer_func(parser, completion, answer) -> float:
        response = parser.parse_answer(completion) or ''
        return 1.0 if response.strip() == answer.strip() else 0.0
    
    rubric = vf.Rubric(
        funcs=[correct_answer_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser
    )
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
```

### Parser Selection Strategy

```python
def get_parser_for_task(task_type: str):
    """Select appropriate parser based on task."""
    if task_type == "math":
        return vf.ThinkParser(extract_fn=extract_boxed_answer)
    elif task_type == "reasoning":
        return vf.XMLParser(fields=["reasoning", "answer"])
    elif task_type == "code":
        return CodeParser()  # Custom parser
    else:
        return vf.Parser()  # Default parser
```

## Common Parser Patterns

### Multi-Turn Parser

```python
class ConversationParser(vf.Parser):
    """Parse multi-turn conversations."""
    
    def parse_conversation(self, messages):
        """Extract conversation structure."""
        conversation = {
            'turns': len([m for m in messages if m['role'] == 'assistant']),
            'total_length': sum(len(m['content']) for m in messages),
            'last_response': messages[-1]['content'] if messages else ""
        }
        return conversation
    
    def parse_answer(self, completion):
        """Get the final response."""
        if isinstance(completion, list) and completion:
            return completion[-1]['content']
        return str(completion) if completion else None
```

### Error-Resilient Parser

```python
class RobustParser(vf.Parser):
    """Parser that handles malformed input gracefully."""
    
    def parse(self, text: str) -> dict:
        """Parse with multiple fallback strategies."""
        
        # Try primary parsing strategy
        try:
            return self.primary_parse(text)
        except Exception as e1:
            # Try secondary strategy
            try:
                return self.fallback_parse(text)
            except Exception as e2:
                # Return safe default
                return {
                    'valid': False,
                    'raw_text': text,
                    'errors': [str(e1), str(e2)]
                }
    
    def primary_parse(self, text: str) -> dict:
        """Primary parsing logic."""
        # Implementation...
        pass
    
    def fallback_parse(self, text: str) -> dict:
        """Fallback parsing logic."""
        # Implementation...
        pass
```

## Best Practices

1. **Always Include Format Rewards**: Use `parser.get_format_reward_func()` in your rubrics
2. **Handle Edge Cases**: Design parsers to gracefully handle malformed input
3. **Clear Field Names**: Use descriptive field names in XMLParser
4. **Consistent Return Types**: Always return strings from `parse_answer()`
5. **Test Thoroughly**: Test parsers with various model outputs
6. **Document Expected Format**: Include format requirements in system prompts

## Parser Testing

```python
def test_parser():
    """Test parser with various inputs."""
    parser = vf.XMLParser(fields=["reasoning", "answer"])
    
    # Test valid input
    valid_input = "<reasoning>Test</reasoning><answer>42</answer>"
    result = parser.parse(valid_input)
    assert result.reasoning == "Test"
    assert result.answer == "42"
    
    # Test malformed input
    malformed_input = "<reasoning>Test</reasoning><answer>42"
    result = parser.parse(malformed_input)
    assert result.reasoning == "Test"
    assert result.answer is None  # Missing closing tag
    
    # Test format reward
    format_reward = parser.get_format_reward_func()
    completion = [{"role": "assistant", "content": valid_input}]
    reward = format_reward(completion)
    assert reward > 0.8  # Should reward valid format

test_parser()
```

## TODO Sections

TODO: Add documentation for:
- Advanced XML schema validation patterns
- Integration with different model chat templates
- Performance optimization for large-scale parsing
- Custom format reward function patterns