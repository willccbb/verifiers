# How to Run Tests

This guide shows how to run tests using standard pytest commands.

## Prerequisites

```bash
pip install pytest pytest-asyncio pytest-cov
```

## Running All Tests

```bash
# Basic run
pytest

# Verbose output
pytest -v

# With coverage
pytest --cov=verifiers --cov-report=term-missing
```

## Running Specific Test Files

```bash
# Run parser tests
pytest tests/test_parser.py tests/test_xml_parser.py tests/test_think_parser.py

# Run environment tests  
pytest tests/test_environment.py tests/test_singleturn_env.py tests/test_multiturn_env.py tests/test_env_group.py

# Run rubric tests
pytest tests/test_rubric.py tests/test_rubric_group.py
```

## Running Specific Tests

```bash
# Run a specific test class
pytest tests/test_parser.py::TestParser

# Run a specific test method
pytest tests/test_parser.py::TestParser::test_parse_returns_text_as_is

# Run tests matching a pattern
pytest -k "test_parse"
```

## Test Markers

```bash
# Run only slow tests
pytest -m slow

# Run all except slow tests
pytest -m "not slow"

# Run async tests
pytest -m asyncio
```

## Coverage Reports

```bash
# Terminal coverage report
pytest --cov=verifiers --cov-report=term-missing

# HTML coverage report
pytest --cov=verifiers --cov-report=html
# Then open htmlcov/index.html in a browser

# XML coverage report (for CI)
pytest --cov=verifiers --cov-report=xml
```

## Useful Options

```bash
# Stop on first failure
pytest -x

# Run failed tests from last run
pytest --lf

# Show local variables on failure
pytest -l

# Disable output capturing (show print statements)
pytest -s

# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

## Quick Commands

```bash
# Quick test run (quiet mode)
pytest -q

# Detailed test run with coverage
pytest -v --cov=verifiers --cov-report=term-missing

# Test a specific component
pytest tests/test_parser.py -v

# Debug a failing test
pytest tests/test_parser.py::TestParser::test_parse_returns_text_as_is -vvs
```

## Configuration

Test configuration is defined in `pytest.ini`. The default settings include:
- Automatic test discovery in the `tests/` directory
- Short traceback format
- Strict marker checking
- Filtered warnings
- Automatic async test handling