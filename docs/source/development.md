# Development Guide

This guide covers development setup, testing, and contributing to the verifiers package.

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/verifiers.git
cd verifiers

# Install in development mode with all dependencies
uv sync --all-extras

# Or install just test dependencies
uv add --optional tests
```

## Project Structure

```
verifiers/
├── verifiers/          # Main package
│   ├── envs/           # Environment classes
│   ├── parsers/        # Parser classes  
│   ├── rubrics/        # Rubric classes
│   ├── tools/          # Tool implementations
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
└── pyproject.toml      # Project configuration
```

## Testing

### Quick Commands

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=verifiers --cov-report=html

# Run specific test category
uv run pytest tests/test_parsers.py
uv run pytest tests/test_rubrics.py  
uv run pytest tests/test_environments.py

# Run with verbose output
uv run pytest tests/ -v
```

### Test Runner Script

```bash
# Use the convenience script
python tests/run_tests.py all      # All tests with coverage
python tests/run_tests.py quick    # Quick run without coverage
python tests/run_tests.py parsers  # Just parser tests
```

### Coverage Goals

- Maintain **90%+ line coverage**
- **100% coverage** of public APIs
- Cover edge cases and error conditions

## Code Style

### Rules (from CLAUDE.md)

- Follow existing code conventions
- Check imports and dependencies before using libraries
- Maintain security best practices  
- No comments unless requested
- Comments should reflect global behavior and be "timeless"
- Never reference old vs. new code state in comments

### Linting

```bash
# Check for linting tools in the project
uv run ruff check verifiers/
uv run mypy verifiers/
```

## Testing Guidelines

### Writing Tests

1. **Descriptive test names**
   ```python
   def test_xml_parser_with_missing_fields(self):
   ```

2. **Test both success and failure cases**
   ```python
   def test_valid_input(self):
       # Test normal operation
   
   def test_invalid_input_raises_error(self):
       # Test error handling
   ```

3. **Use fixtures effectively**
   ```python
   def test_with_mock_client(mock_client):
       # Use provided fixtures
   ```

4. **Organize with test classes**
   ```python
   class TestXMLParser:
       """Test XML parsing functionality."""
   ```

### Test Categories

- **Unit tests** - Test individual methods
- **Integration tests** - Test component interactions  
- **Edge case tests** - Test boundary conditions
- **Async tests** - Test asynchronous operations

### Mock Usage

The test suite provides comprehensive mocks:

```python
from tests.mock_openai_client import MockOpenAIClient

# Basic mock client
client = MockOpenAIClient()

# Math-specific responses
client = create_mock_math_client()

# Tool-specific responses  
client = create_mock_tool_client()
```

## CI/CD

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled runs

### Local CI Simulation

```bash
# Run the same checks as CI
uv run pytest tests/ --cov=verifiers --cov-report=xml
```

## Documentation

### Building Docs

```bash
cd docs/
make html
```

### Documentation Files

- `docs/source/index.md` - Main documentation
- `docs/source/testing.md` - Testing guide
- `docs/source/development.md` - This file
- `tests/README.md` - Detailed test documentation

## Contributing

### Workflow

1. **Fork and clone** the repository
2. **Create a branch** for your feature/fix
3. **Install dependencies** with `uv add --optional tests`
4. **Make your changes** following code style guidelines
5. **Add tests** for new functionality
6. **Run tests** to ensure everything passes
7. **Update documentation** if needed
8. **Submit a pull request**

### Pull Request Checklist

- [ ] Tests pass: `uv run pytest tests/`
- [ ] Coverage maintained: Check coverage report
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)

### Review Process

- All tests must pass
- Maintain or improve code coverage
- Follow existing patterns and conventions
- Clear commit messages and PR description

## Debugging

### Test Debugging

```bash
# Stop on first failure
uv run pytest tests/ -x

# Enter debugger on failure
uv run pytest tests/ --pdb

# Verbose output
uv run pytest tests/ -vvv

# Run specific failing test
uv run pytest tests/test_file.py::TestClass::test_method -v
```

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
uv pip install -e .
```

#### Test Dependencies
```bash
# Install all test dependencies
uv add --optional tests
```

#### Async Issues
```bash
# If async tests fail, ensure nest-asyncio is available
uv add nest-asyncio
```

## Release Process

### Version Bumping

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"
```

### Building

```bash
# Build distribution packages
uv build

# Check build contents
tar -tzf dist/verifiers-*.tar.gz
```

### Publishing

```bash
# Upload to PyPI (maintainers only)
uv publish
```

## Performance

### Test Performance

The test suite is optimized for speed:
- Uses mocks instead of real API calls
- Parallel execution where possible
- Efficient fixtures and setup

Typical benchmark:
- Full test suite: ~1-2 seconds
- Individual test file: ~0.1-0.3 seconds

### Profiling

```bash
# Profile test execution
uv run pytest tests/ --profile

# Profile specific functionality
python -m cProfile -o profile.stats script.py
```

## Troubleshooting

### Common Development Issues

1. **Tests fail after changes**
   - Run tests individually to isolate issues
   - Check for import or dependency problems
   - Verify mock behavior matches expectations

2. **Coverage drops**
   - Check which lines are not covered
   - Add tests for missing branches
   - Verify test fixtures are working

3. **CI failures**
   - Run same commands locally
   - Check for environment-specific issues
   - Verify all dependencies are specified

### Getting Help

- Check existing issues and documentation
- Review test output carefully
- Use debugging tools: `--pdb`, `-vvv`, `-x`
- Ask questions in issues or discussions