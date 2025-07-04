# Multi-Turn Dependencies in MockAsyncOpenAI

## Overview

**Yes!** The `MockAsyncOpenAI` fully supports multi-turn dependencies. The key insight is that `add_chat_response()` maps the **entire conversation history** to responses, enabling sophisticated context-dependent testing.

## How It Works

### Core Mechanism
```python
client.add_chat_response(
    messages=[entire_conversation_history],  # Full context
    response="Context-dependent response"
)
```

The conversation history is converted to a hashable key:
```python
# Internal representation
("user:Hello", "assistant:Hi!", "user:How are you?") → response
```

### Multi-Turn Interface

```python
client = MockAsyncOpenAI()

# Turn 1: Initial conversation
client.add_chat_response(
    messages=[
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": "I need help with math"}
    ],
    response="What specific topic?"
)

# Turn 2: Conversation builds on previous context  
client.add_chat_response(
    messages=[
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": "I need help with math"},
        {"role": "assistant", "content": "What specific topic?"},
        {"role": "user", "content": "Quadratic equations"}
    ],
    response="Let's start with the quadratic formula!"
)

# Turn 3: Even more context-dependent
client.add_chat_response(
    messages=[
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": "I need help with math"},
        {"role": "assistant", "content": "What specific topic?"},
        {"role": "user", "content": "Quadratic equations"},
        {"role": "assistant", "content": "Let's start with the quadratic formula!"},
        {"role": "user", "content": "Can you show an example?"}
    ],
    response="Sure! Let's solve x² - 5x + 6 = 0"
)
```

## Real Test Usage Examples

### 1. Context-Dependent Responses
The same question gets different answers based on conversation history:

```python
@pytest.mark.asyncio
async def test_context_dependent_math_help():
    client = MockAsyncOpenAI()
    
    # Math context
    client.add_chat_response(
        messages=[
            {"role": "user", "content": "I'm studying calculus"},
            {"role": "assistant", "content": "Great! Calculus is fascinating."},
            {"role": "user", "content": "What should I focus on?"}
        ],
        response="Focus on limits and derivatives first."
    )
    
    # Cooking context - same question, different answer  
    client.add_chat_response(
        messages=[
            {"role": "user", "content": "I'm learning to cook"},
            {"role": "assistant", "content": "Cooking is wonderful!"},
            {"role": "user", "content": "What should I focus on?"}
        ],
        response="Start with knife skills and heat control."
    )
    
    # Test both contexts
    math_response = await client.chat.completions.create(messages=[...])
    cooking_response = await client.chat.completions.create(messages=[...])
    
    assert math_response.choices[0].message.content == "Focus on limits and derivatives first."
    assert cooking_response.choices[0].message.content == "Start with knife skills and heat control."
```

### 2. Branching Conversations
Handle conversation trees with different paths:

```python
@pytest.mark.asyncio 
async def test_branching_travel_assistant():
    client = MockAsyncOpenAI()
    
    start_conversation = [
        {"role": "system", "content": "You are a travel assistant."},
        {"role": "user", "content": "I want to plan a vacation"},
        {"role": "assistant", "content": "What type interests you?"}
    ]
    
    # Branch A: Beach vacation
    client.add_chat_response(
        messages=start_conversation + [
            {"role": "user", "content": "Beach vacation"}
        ],
        response="Consider Hawaii, Caribbean, or Mediterranean!"
    )
    
    # Branch B: Mountain vacation
    client.add_chat_response(
        messages=start_conversation + [
            {"role": "user", "content": "Mountain vacation"}
        ],
        response="Look into Alps, Rockies, or Himalayas!"
    )
    
    # Branch A continued
    client.add_chat_response(
        messages=start_conversation + [
            {"role": "user", "content": "Beach vacation"},
            {"role": "assistant", "content": "Consider Hawaii, Caribbean, or Mediterranean!"},
            {"role": "user", "content": "Budget options?"}
        ],
        response="Try Mexico, Thailand, or Florida."
    )
```

### 3. MultiTurnEnv Integration
How this works with actual environment testing:

```python
@pytest.mark.asyncio
async def test_multiturn_environment_flow():
    client = MockAsyncOpenAI()
    
    # Set up the exact conversation flow that MultiTurnEnv produces
    
    # Turn 1: Initial user message
    client.add_chat_response(
        messages=[{"role": "user", "content": "Start conversation"}],
        response="First response"
    )
    
    # Turn 2: After environment adds its response
    client.add_chat_response(
        messages=[
            {"role": "user", "content": "Start conversation"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Continue (turn 1)"}  # Environment response
        ],
        response="Second response" 
    )
    
    # Turn 3: Environment encourages completion
    client.add_chat_response(
        messages=[
            {"role": "user", "content": "Start conversation"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Continue (turn 1)"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Please finish with DONE"}
        ],
        response="Final response DONE"  # Triggers completion
    )
    
    # Test the full conversation flow
    env = MultiTurnEnv(client=client, ...)
    completion, state = await env.rollout(...)
    
    assert len(completion) == 5  # All conversation turns
    assert "DONE" in completion[-1]["content"]  # Completion triggered
```

## Key Benefits

### 1. **True Context Awareness**
- Same inputs with different conversation history → different outputs
- Realistic simulation of how LLMs actually behave
- Perfect for testing context-dependent logic

### 2. **Complex Conversation Testing**
- Multi-turn dialogues with state dependencies
- Branching conversation trees
- Environment-driven conversation flows

### 3. **Deterministic Testing**
- Same conversation history always produces same response
- No flaky tests due to order dependencies
- Reproducible across test runs

### 4. **Easy Debugging**
- Clear mapping: conversation history → expected response
- Easy to trace where conversation went wrong
- Simple to add new test cases

## Comparison with Alternatives

### ❌ Old Approach (Order-Dependent)
```python
# Brittle - depends on call order
client.side_effect = [response1, response2, response3]

# Problems:
# - Breaks if tests run in different order
# - No context awareness  
# - Hard to debug failures
# - Can't handle branching conversations
```

### ✅ Current Approach (Context-Dependent)
```python
# Robust - depends on conversation content
client.add_chat_response(messages=[full_context], response="...")

# Benefits:
# - Order independent
# - Context aware
# - Easy to debug
# - Handles complex conversation trees
```

## Best Practices

### 1. **Include Full Context**
```python
# Good: Include entire conversation history
client.add_chat_response(
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    response="Context-aware response"
)
```

### 2. **Map Exact Expected Flows**
```python
# Map the exact conversation your environment will produce
# Study the environment's env_response() method to predict the flow
```

### 3. **Use Default Responses for Fallbacks**
```python
# Handle unexpected conversation paths gracefully
client.set_default_responses(chat_response="Unexpected conversation path")
```

### 4. **Test Conversation Branches**
```python
# Test different conversation paths separately
# Each branch gets its own mapping
```

## Summary

The `MockAsyncOpenAI` provides **full multi-turn dependency support** through conversation history mapping. This enables:

- **Context-dependent responses**: Same question → different answers based on history
- **Complex conversation trees**: Branching dialogues with state dependencies  
- **Environment integration**: Perfect for MultiTurnEnv testing
- **Deterministic testing**: Reliable, reproducible test results

The interface is simple but powerful: map full conversation histories to expected responses, and the mock client handles the rest!