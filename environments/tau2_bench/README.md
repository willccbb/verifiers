# Tau2Bench Environment

Implementation of [Tau-Bench](https://github.com/sierra-research/tau2-bench) for the verifiers framework.

## Status

⚠️ **Work in Progress**: This implementation is actively being developed to achieve full parity with the original Tau2-Bench logic.

**Current Performance (Retail Domain)**:
- Model: gpt-4.1-mini
- Pass@1: 42.1%
- User model: gpt-4.1
- Note: The original benchmark reports >60% for gpt-4.1-mini


## Overview

Tau2Bench evaluates language models on complex, multi-turn conversations requiring tool use across three domains:
- **Retail**: E-commerce customer service scenarios
- **Airline**: Flight booking and management tasks  
- **Telecom**: Telecommunications service support (includes user-side tool calls)

## Installation

```bash
vf-install tau2_bench
```

This will automatically install the tau2-bench package and its dependencies.

## Usage

```bash
# Evaluate on retail domain (default)
vf-eval tau2_bench -n 20

# Evaluate on specific domain
vf-eval tau2_bench -n 20 --env-args '{"domain": "airline"}'

# Evaluate with different user simulator model
vf-eval tau2_bench -n 20 --env-args '{"user_llm": "gpt-4"}'
```

## Environment Arguments

- `domain`: Domain to evaluate on ("retail", "airline", "telecom"). Default: "retail"
- `user_llm`: Model to use for user simulation. Default: "gpt-4.1" 
- `solo_mode`: Enable solo mode for telecom domain. Default: False
- `max_steps`: Maximum conversation steps. Default: 200
- `max_errors`: Maximum allowed errors before termination. Default: 10

## Implementation Details

This environment wraps the original tau2-bench implementation, providing:
- Native tau2 tool execution through `tau2_env.get_response()`
- Proper state management and database isolation between rollouts
- User simulation using tau2's UserSimulator
- Original tau2 evaluation metrics via `evaluate_simulation()`

The implementation aims to exactly mirror tau2's behavior while conforming to the verifiers framework patterns.

## Evaluation

The environment uses tau2's native evaluation, which checks:
1. **Task Completion**: Whether the agent successfully completed the requested task
2. **Communication Score**: How well the agent's actions matched expected behavior
3. **Database State**: Whether the final database state matches expectations

## Troubleshooting

If you encounter import errors, ensure tau2-bench is properly installed:
```bash
vf-install tau2-bench
```

For evaluation discrepancies, check that:
- The model has access to all required tools
- The system prompt includes domain-specific policies
- Tool responses are properly formatted

## Contributing

When improving this implementation:
1. Ensure all changes maintain compatibility with tau2's evaluation logic
2. Test against multiple domains and edge cases
3. Verify database state isolation between parallel rollouts
4. Keep tool execution logic aligned with tau2's original implementation