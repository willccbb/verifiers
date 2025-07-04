# MockAsyncOpenAI Guide: Input-Output Mapping

## Overview

The `MockAsyncOpenAI` provides a sophisticated mock OpenAI client that maps specific inputs to specific outputs, making tests more predictable and order-independent.

## Key Features

### 1. Input-Output Mapping
- **Chat completions**: Maps conversation history to specific responses
- **Text completions**: Maps prompts to specific responses
- **Order independence**: Responses are consistent regardless of call order
- **Default responses**: Fallback responses for unmapped inputs

### 2. Smart Hashing
- Conversations are converted to hashable keys for consistent lookup
- Message order and content are preserved in the mapping
- System prompts are included in the mapping for realistic testing

## Basic Usage

### Setting Up Chat Response Mappings

```python
from tests.conftest import MockAsyncOpenAI

client = MockAsyncOpenAI()

# Add specific chat response mapping
client.add_chat_response(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    response="The answer is 4",
    finish_reason="stop"
)

# Use the client
response = await client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
)
assert response.choices[0].message.content == "The answer is 4"
```

### Setting Up Text Completion Mappings

```python
# Add text completion mapping
client.add_text_response(
    prompt="Calculate 2+2:",
    response="4",
    finish_reason="stop"
)

# Use the client
response = client.completions.create(prompt="Calculate 2+2:")
assert response.choices[0].text == "4"
```

### Custom Default Responses

```python
# Set custom default responses
client.set_default_responses(
    chat_response="I don't know that",
    text_response="No answer available"
)

# Unmapped requests will use these defaults
response = client.completions.create(prompt="Unknown question")
assert response.choices[0].text == "No answer available"
```

## Advanced Usage

### Testing Order Independence

```python
# Set up multiple mappings
client.add_chat_response(
    messages=[{"role": "user", "content": "Question A"}],
    response="Answer A"
)
client.add_chat_response(
    messages=[{"role": "user", "content": "Question B"}],
    response="Answer B"
)

# Test in different orders
response_a = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Question A"}]
)
response_b = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Question B"}]
)

# Test reverse order
response_b2 = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Question B"}]
)
response_a2 = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Question A"}]
)

# All responses are consistent
assert response_a.choices[0].message.content == response_a2.choices[0].message.content
assert response_b.choices[0].message.content == response_b2.choices[0].message.content
```

### Using with Environments

```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric

# Set up client with specific mappings
client = MockAsyncOpenAI()
client.add_chat_response(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    response="The answer is 4"
)

# Create environment
env = SingleTurnEnv(
    client=client,
    model="test-model",
    dataset=your_dataset,
    system_prompt="You are a helpful assistant.",
    parser=Parser(),
    rubric=Rubric()
)

# Run rollouts - responses will be consistent
completion, state = await env.rollout(
    client=client,
    model="test-model",
    prompt=[{"role": "user", "content": "What is 2+2?"}],
    answer="4"
)
```

## Comparison with Alternative Approaches

### Order-Dependent Mocking
```python
@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.chat.completions.create.side_effect = [
        mock_response_1,
        mock_response_2,
        mock_response_3
    ]
    return client
```

### Input-Output Mapping
```python
@pytest.fixture
def mock_client():
    client = MockAsyncOpenAI()
    client.add_chat_response(messages=conversation_1, response="Response 1")
    client.add_chat_response(messages=conversation_2, response="Response 2")
    client.add_chat_response(messages=conversation_3, response="Response 3")
    return client
```

## Benefits

1. **Predictable Testing**: Responses are always the same for the same inputs
2. **Order Independence**: Tests don't break when execution order changes
3. **Realistic Simulation**: Mimics how a real API would respond to specific inputs
4. **Easier Debugging**: Clear mapping between inputs and outputs
5. **Flexible Defaults**: Handle unmapped cases gracefully

## Implementation Details

### Message Hashing
- Messages are converted to tuples of `"role:content"` strings
- This creates a consistent, hashable key for lookup
- Preserves message order and content exactly

### Response Structure
- Chat responses include `message.content` and `finish_reason`
- Text responses include `text` and `finish_reason`
- Response objects mirror OpenAI's API structure

### Error Handling
- Unmapped inputs return default responses (never fail)
- Invalid inputs log warnings but don't crash tests
- Graceful fallbacks for edge cases

## Best Practices

1. **Set up mappings early**: Add all expected input-output pairs before running tests
2. **Use realistic inputs**: Include system prompts and full conversation context
3. **Test edge cases**: Include unmapped inputs to verify default behavior
4. **Keep mappings simple**: One mapping per distinct conversation or prompt
5. **Group related tests**: Use fixtures to share common mappings across test methods