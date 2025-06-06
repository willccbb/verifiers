# Testing Guide

The verifiers package includes a comprehensive test suite with 133+ tests covering Parser, Rubric, and Environment classes. This guide explains how to install test dependencies, run tests, and understand the test structure.

## Installation

### Install Test Dependencies

Test dependencies are defined as an optional dependency group. Install them using:

```bash
# Install test dependencies
uv add --optional tests

# Or install specific test packages
uv add pytest pytest-asyncio pytest-cov
```

### Dependencies

The test suite requires:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support  
- `pytest-cov` - Coverage reporting

## Running Tests

### Quick Start

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_parsers.py
```

### Using the Test Runner

The test suite includes a convenient test runner script:

```bash
# Run all tests with coverage
python tests/run_tests.py all

# Run specific test categories
python tests/run_tests.py parsers
python tests/run_tests.py rubrics  
python tests/run_tests.py environments

# Run quick tests without coverage
python tests/run_tests.py quick

# Run specific test file or pattern
python tests/run_tests.py test_parsers.py::TestXMLParser
```

### Test Categories

#### Parser Tests (51 tests)
Test XML parsing, format validation, and edge cases:

```bash
uv run pytest tests/test_parsers.py -v
```

#### Rubric Tests (44 tests)  
Test reward functions, async scoring, and tool evaluation:

```bash
uv run pytest tests/test_rubrics.py -v
```

#### Environment Tests (38 tests)
Test single/multi-turn scenarios, dataset processing, and rollouts:

```bash
uv run pytest tests/test_environments.py -v
```

## Coverage

### Generate Coverage Reports

```bash
# HTML coverage report
uv run pytest tests/ --cov=verifiers --cov-report=html

# Terminal coverage report
uv run pytest tests/ --cov=verifiers --cov-report=term-missing

# Both HTML and terminal
uv run pytest tests/ --cov=verifiers --cov-report=html --cov-report=term-missing
```

Coverage reports are generated in `htmlcov/` directory.

### Coverage Goals

The test suite maintains:
- **90%+ line coverage** across all modules
- **100% coverage** of public APIs
- **Edge case coverage** for error conditions

## Test Structure

### Test Organization

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── mock_openai_client.py    # Mock OpenAI client for testing
├── test_parsers.py          # Parser class tests
├── test_rubrics.py          # Rubric class tests  
├── test_environments.py     # Environment class tests
├── run_tests.py            # Test runner script
└── README.md               # Detailed test documentation
```

### Mock Infrastructure

The test suite uses comprehensive mocks to avoid external dependencies:

- **MockOpenAIClient** - Simulates OpenAI API responses
- **Pattern-based responses** - Different responses for different inputs
- **Error simulation** - Tests API failures and edge cases
- **Specialized clients** - Math and tool-specific response patterns

### Test Types

#### Unit Tests
Test individual methods and functions:

```bash
# Test specific parser method
uv run pytest tests/test_parsers.py::TestXMLParser::test_parse_simple_xml
```

#### Integration Tests  
Test component interactions:

```bash
# Test rubric inheritance
uv run pytest tests/test_rubrics.py::TestRubricIntegration
```

#### Async Tests
Test asynchronous functionality:

```bash
# Run async tests
uv run pytest tests/ -k "async"
```

#### Edge Case Tests
Test boundary conditions and error handling:

```bash
# Test edge cases
uv run pytest tests/ -k "edge"
```

## Advanced Usage

### Filtering Tests

```bash
# Run tests matching pattern
uv run pytest tests/ -k "xml_parser"

# Run tests with specific markers
uv run pytest tests/ -m "not slow"

# Run specific test class
uv run pytest tests/test_parsers.py::TestXMLParser
```

### Debugging Tests

```bash
# Stop on first failure
uv run pytest tests/ -x

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Verbose output with full tracebacks
uv run pytest tests/ -vvv --tb=long
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
uv add pytest-xdist
uv run pytest tests/ -n auto
```

## Continuous Integration

The test suite is designed for CI environments:

- **Fast execution** - Most tests complete in seconds
- **No external dependencies** - Uses comprehensive mocking
- **Clear failure messages** - Easy debugging in CI
- **Coverage reporting** - Automated coverage tracking

### Example CI Configuration

```yaml
# GitHub Actions example
- name: Install test dependencies
  run: uv add --optional tests

- name: Run tests with coverage
  run: uv run pytest tests/ --cov=verifiers --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Configuration

### pytest.ini

The test suite uses these pytest configurations:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --strict-markers --tb=short --disable-warnings
asyncio_mode = auto
```

### Markers

Available test markers:

- `slow` - Tests that take longer to run
- `integration` - Integration tests
- `unit` - Unit tests  
- `asyncio` - Async tests

## Writing Tests

### Guidelines

When adding new tests:

1. **Use descriptive names** - `test_xml_parser_with_missing_fields`
2. **Test success and failure cases** - Cover both paths
3. **Use appropriate fixtures** - Leverage existing test infrastructure
4. **Add docstrings** - Explain complex test scenarios
5. **Group related tests** - Use test classes for organization

### Test Structure Example

```python
class TestMyFeature:
    """Test the MyFeature functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature operation."""
        # Arrange
        feature = MyFeature()
        
        # Act
        result = feature.process("input")
        
        # Assert
        assert result == "expected_output"
    
    def test_error_handling(self):
        """Test feature handles errors gracefully."""
        feature = MyFeature()
        
        with pytest.raises(ValueError):
            feature.process(None)
```

### Using Fixtures

```python
def test_with_mock_client(mock_client):
    """Test using the mock OpenAI client fixture."""
    env = SingleTurnEnv(client=mock_client)
    result = env.rollout(mock_client, "model", "prompt", "answer")
    assert result is not None
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/verifiers

# Install in development mode
uv pip install -e .
```

#### Async Test Issues  
```bash
# Install nest-asyncio if needed
uv add nest-asyncio
```

#### Coverage Issues
```bash
# Ensure verifiers package is installed
uv pip install -e .

# Run with explicit source
uv run pytest tests/ --cov=./verifiers
```

### Getting Help

- Check test logs: `uv run pytest tests/ -v -s`
- Run individual tests: `uv run pytest tests/test_file.py::test_name -v`
- Review test documentation: `tests/README.md`

## Performance

The test suite is optimized for speed:

- **Parallel execution** - Tests run concurrently where possible
- **Efficient mocking** - No network calls or file I/O
- **Smart fixtures** - Reusable test components
- **Fast startup** - Minimal import overhead

Typical execution times:
- **Full suite**: ~1-2 seconds
- **Single file**: ~0.1-0.3 seconds  
- **Individual test**: ~0.01-0.05 seconds