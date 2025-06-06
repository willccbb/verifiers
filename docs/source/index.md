# Verifiers

Verifiers is a Python package for reinforcement learning with LLMs, providing parsers, rubrics, and environments for training and evaluation.

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents:

testing
development
```

## Quick Start

### Installation

```bash
# Install the package
uv add verifiers

# Install with test dependencies
uv add verifiers --optional tests
```

### Basic Usage

```python
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.envs import SingleTurnEnv

# Create components
parser = XMLParser(["reasoning", "answer"])
rubric = Rubric()
env = SingleTurnEnv(dataset=dataset, client=client)
```

## Components

### Parsers
- **XMLParser** - Parse structured XML responses
- **SmolaParser** - Enhanced XML parsing with tool support
- **Parser** - Base parser class

### Rubrics  
- **Rubric** - Base reward function aggregation
- **ToolRubric** - Tool execution and code evaluation
- **RubricGroup** - Multiple rubric aggregation

### Environments
- **SingleTurnEnv** - Single interaction environments
- **MultiTurnEnv** - Multi-turn conversation environments  
- **Environment** - Base environment class

## Testing

The package includes a comprehensive test suite with 133+ tests. See the [Testing Guide](testing.md) for details on running tests, coverage, and contributing.

## Development

### Running Tests

```bash
# Install test dependencies
uv add --optional tests

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=verifiers
```

### Contributing

1. Fork the repository
2. Install development dependencies: `uv add --optional tests`
3. Make your changes
4. Run tests: `uv run pytest tests/`
5. Submit a pull request