# Verifiers Documentation

This directory contains the comprehensive documentation for the verifiers framework.

## Building the Documentation

### Prerequisites

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Or using uv
uv add sphinx sphinx-rtd-theme myst-parser
```

### Build Commands

```bash
# Build HTML documentation
cd docs/
make html

# Or using uv from project root
cd docs/
uv run make html

# View the documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

## Documentation Structure

### Getting Started
- **overview.md** - Core concepts and architecture
- **examples.md** - Walkthrough of example implementations

### Core Components
- **environments.md** - Environment types and patterns
- **parsers.md** - Output parsing and validation
- **rubrics.md** - Evaluation and reward functions
- **tools.md** - Tool integration and safety

### Advanced Topics
- **training.md** - GRPO training and optimization
- **advanced.md** - Advanced patterns and techniques
- **api_reference.md** - Complete API documentation

### Development
- **testing.md** - Testing guide and infrastructure
- **development.md** - Contributing and development setup

## Key Design Principles

The documentation emphasizes:

1. **Conceptual Understanding** - Start with the big picture
2. **Practical Examples** - Learn by doing with real code
3. **Progressive Complexity** - Simple to advanced patterns
4. **Opinionated Guidance** - Best practices and recommendations

## Documentation Philosophy

- **XMLParser is recommended** - While flexible, we guide users to reliable patterns
- **Modular composition** - Build complex behaviors from simple parts
- **Production focus** - Patterns that work at scale
- **Clear examples** - Every concept illustrated with code

## Contributing to Docs

When adding documentation:

1. Use clear, concise language
2. Include code examples for every concept
3. Show both simple and advanced usage
4. Explain the "why" not just the "how"
5. Keep consistency with existing style

## Deployment

The documentation is designed to work with:
- GitHub Pages
- ReadTheDocs
- Any static site host

For ReadTheDocs, the existing configuration should work out of the box.