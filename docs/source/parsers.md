# Parsers

Parsers extract structured information from model outputs. While the framework supports custom parsers, **we strongly recommend using XMLParser** for its reliability, built-in validation, and format rewards.

## Parser Hierarchy

```
Parser (base class)
├── XMLParser      # Recommended: XML-tagged field extraction
└── SmolaParser    # Specialized for Smolagents tool format
```

## Why XMLParser?

XMLParser provides several advantages over plain text parsing:

1. **Unambiguous Structure**: Clear field boundaries with XML tags
2. **Format Validation**: Built-in reward functions for compliance
3. **Flexible Fields**: Support for optional and alternative field names
4. **Error Recovery**: Graceful handling of malformed outputs
5. **Multi-format Support**: Works with both strings and message lists

## Basic XMLParser Usage

### Simple Fields

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

### Optional Fields

Handle fields that may not always be present:

```python
parser = XMLParser(fields=["reasoning", "answer", "confidence"])

output = """
<reasoning>Simple addition</reasoning>
<answer>4</answer>
"""

parsed = parser.parse(output)
print(parsed.reasoning)    # "Simple addition"
print(parsed.answer)       # "4"
print(parsed.confidence)   # "" (empty string for missing fields)
```

## Format Enforcement

### Built-in Format Rewards

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

# Partial format gets partial reward
partial_output = "<answer>42</answer>"
reward = format_func(partial_output)  # 0.5 (has answer but missing reasoning)
```

### Integrating Format Rewards

Always include format rewards in your rubric:

```python
from verifiers.rubrics import Rubric

parser = XMLParser(fields=["reasoning", "answer"])

rubric = Rubric(
    funcs=[
        correct_answer_func,           # Weight: 0.8
        parser.get_format_reward_func()  # Weight: 0.2
    ],
    weights=[0.8, 0.2],  # Format is important but not primary
    parser=parser
)
```

## Advanced Parsing Features

### Parsing from Message Lists

XMLParser handles both strings and OpenAI message formats:

```python
# String input
string_output = "<reasoning>...</reasoning><answer>42</answer>"
parsed = parser.parse(string_output)

# Message list input (from chat models)
messages = [
    {"role": "assistant", "content": "<reasoning>...</reasoning>"},
    {"role": "assistant", "content": "<answer>42</answer>"}
]
parsed = parser.parse(messages)  # Automatically concatenates
```

### Answer Extraction

Special method for common answer-only parsing:

```python
parser = XMLParser(fields=["answer"])

# Works with various formats
outputs = [
    "<answer>42</answer>",
    "The answer is <answer>42</answer>.",
    "<reasoning>...</reasoning>\n<answer>42</answer>",
    [{"role": "assistant", "content": "<answer>42</answer>"}]
]

for output in outputs:
    answer = parser.parse_answer(output)
    print(answer)  # Always "42"
```

### Raw String Access

Access unparsed content when needed:

```python
output = "<reasoning>Complex\nMulti-line\nReasoning</reasoning>"
parsed = parser.parse(output)

# Parsed attribute (cleaned up)
print(parsed.reasoning)  # "Complex Multi-line Reasoning"

# Raw extraction
raw = parser._extract_tag_content(output, "reasoning")
print(raw)  # "Complex\nMulti-line\nReasoning"
```

## Custom Parser Implementation

If you need custom parsing logic, extend the base Parser:

```python
from verifiers.parsers import Parser
import json

class JSONParser(Parser):
    """Parse JSON-formatted responses."""
    
    def __init__(self, schema=None):
        super().__init__()
        self.schema = schema
    
    def parse(self, output):
        """Extract JSON from model output."""
        # Handle message lists
        if isinstance(output, list):
            output = " ".join(msg["content"] for msg in output)
        
        # Find JSON block
        try:
            # Handle markdown code blocks
            if "```json" in output:
                start = output.find("```json") + 7
                end = output.find("```", start)
                json_str = output[start:end].strip()
            else:
                # Try to parse the whole output
                json_str = output
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate against schema if provided
            if self.schema:
                self.validate_schema(data)
            
            # Return as object with attribute access
            return type('ParsedJSON', (), data)()
            
        except Exception as e:
            # Return empty object on failure
            return type('ParsedJSON', (), {})()
    
    def get_format_reward_func(self):
        """Reward valid JSON format."""
        def json_format_reward(completion, **kwargs):
            try:
                parsed = self.parse(completion)
                # Check if parsing succeeded
                return 1.0 if hasattr(parsed, '__dict__') else 0.0
            except:
                return 0.0
        
        return json_format_reward
```

## Parser Design Patterns

### 1. Hierarchical Parsing

For complex nested structures:

```python
# Main parser for outer structure
main_parser = XMLParser(fields=["analysis", "solution"])

# Sub-parser for inner structure  
step_parser = XMLParser(fields=["description", "calculation", "result"])

def parse_solution(output):
    # First level parsing
    parsed = main_parser.parse(output)
    
    # Parse steps within solution
    steps = []
    for step_match in re.finditer(r'<step>.*?</step>', parsed.solution, re.DOTALL):
        step_parsed = step_parser.parse(step_match.group())
        steps.append(step_parsed)
    
    return parsed, steps
```

### 2. Fallback Parsing

Handle multiple possible formats:

```python
class FlexibleParser(Parser):
    def __init__(self):
        self.xml_parser = XMLParser(["answer"])
        self.json_parser = JSONParser()
    
    def parse(self, output):
        # Try XML first
        xml_result = self.xml_parser.parse(output)
        if xml_result.answer:
            return xml_result
        
        # Fallback to JSON
        json_result = self.json_parser.parse(output)
        if hasattr(json_result, 'answer'):
            return json_result
        
        # Last resort: regex
        match = re.search(r'answer[:\s]+(\S+)', output, re.I)
        if match:
            result = type('Parsed', (), {'answer': match.group(1)})()
            return result
        
        # Return empty
        return type('Parsed', (), {'answer': ''})()
```

### 3. Validation Parsing

Add validation to parsing:

```python
class ValidatedXMLParser(XMLParser):
    def __init__(self, fields, validators=None):
        super().__init__(fields)
        self.validators = validators or {}
    
    def parse(self, output):
        parsed = super().parse(output)
        
        # Run validators
        for field, validator in self.validators.items():
            if hasattr(parsed, field):
                value = getattr(parsed, field)
                if not validator(value):
                    # Replace with empty on validation failure
                    setattr(parsed, field, "")
        
        return parsed

# Usage
parser = ValidatedXMLParser(
    fields=["reasoning", "answer"],
    validators={
        "answer": lambda x: x.isdigit(),  # Answer must be numeric
        "reasoning": lambda x: len(x) > 10  # Reasoning must be substantial
    }
)
```

## System Prompt Integration

Always communicate format expectations clearly:

```python
SYSTEM_PROMPT = """You are a helpful assistant. Format ALL responses as:

<reasoning>
Explain your step-by-step thought process here.
Show your work and justify your approach.
</reasoning>

<answer>
Your final answer here
</answer>

This exact format is required. Do not include any text outside these tags."""

env = SingleTurnEnv(
    parser=XMLParser(["reasoning", "answer"]),
    system_prompt=SYSTEM_PROMPT
)
```

## Common Pitfalls and Solutions

### 1. Whitespace Handling

XMLParser strips whitespace by default:

```python
output = """
<answer>
    42
</answer>
"""
parsed = parser.parse(output)
print(repr(parsed.answer))  # "42" not "    42    "
```

### 2. Special Characters

XML special characters are handled automatically:

```python
output = '<answer>x < 5 && y > 3</answer>'
parsed = parser.parse(output)
print(parsed.answer)  # "x < 5 && y > 3"
```

### 3. Multiple Tag Instances

Parser returns first occurrence by default:

```python
output = """
<answer>Wrong: 41</answer>
Actually, let me recalculate...
<answer>Correct: 42</answer>
"""
parsed = parser.parse(output)
print(parsed.answer)  # "Wrong: 41" (first occurrence)
```

## Best Practices

### 1. Field Naming
- Use clear, descriptive field names
- Provide alternatives for common variations
- Keep field names consistent across your project

### 2. Format Rewards
- Always include format rewards (10-20% weight typically)
- Use higher weights during initial training
- Reduce weight once models learn the format

### 3. Error Handling
- Always provide fallback values
- Log parsing failures for debugging
- Consider partial credit for partial formatting

### 4. Documentation
- Document expected format in system prompts
- Provide examples in few-shot prompts
- Keep format simple and consistent

## Performance Considerations

### Efficient Parsing

```python
# Compile regex patterns once
parser = XMLParser(fields=["reasoning", "answer"])

# Reuse parser instance across multiple parses
for output in outputs:
    parsed = parser.parse(output)  # Reuses compiled patterns
```

### Batch Processing

```python
# Parse multiple outputs efficiently
outputs = ["<answer>1</answer>", "<answer>2</answer>", ...]
parsed_results = [parser.parse(output) for output in outputs]

# Extract specific fields in bulk
answers = [parser.parse_answer(output) for output in outputs]
```

## Integration with Training

Parsers play a crucial role in the training pipeline:

1. **During Generation**: Parse outputs to extract answers
2. **In Reward Calculation**: Validate format compliance
3. **For Data Filtering**: Remove malformed examples
4. **In Evaluation**: Ensure consistent answer extraction

The XMLParser's reliability and built-in validation make it the ideal choice for production reinforcement learning pipelines.