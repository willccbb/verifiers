# Verifiers Test Suite

This directory contains the test suite for the verifiers package.

## Setup

**Prerequisites:**
- Install `uv` package manager (https://docs.astral.sh/uv/getting-started/installation/)
- Ensure Python 3.11+ is available

Install test dependencies:

```bash
uv sync --extra tests
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run specific test files:

```bash
uv run pytest tests/test_parser.py
uv run pytest tests/test_xml_parser.py
uv run pytest tests/test_think_parser.py
```

Run with coverage:

```bash
uv run pytest --cov=verifiers
```

Run only unit tests:

```bash
uv run pytest -m unit
```

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_parser.py` - Tests for the base Parser class
- `test_xml_parser.py` - Tests for the XMLParser class
- `test_think_parser.py` - Tests for the ThinkParser class

## Test Markers

- `unit` - Fast unit tests (default for all current tests)
- `integration` - Integration tests
- `slow` - Slow-running tests
- `asyncio` - Async tests

## Adding New Tests

1. Create test files following the `test_*.py` naming convention
2. Use the fixtures from `conftest.py` for common parser instances
3. Add appropriate test markers
4. Follow the existing test structure and naming conventions