# Verifiers Documentation

Welcome to Verifiers! This library provides a flexible framework for creating RL environments with custom multi-turn interaction protocols between LLMs and environments.

```{toctree}
:maxdepth: 2
:hidden:

overview
environments
components
training
development
api_reference
```

## What is Verifiers?

Verifiers enables you to:
- Define custom interaction protocols between models and environments
- Build multi-turn conversations, tool-augmented reasoning, and interactive games
- Create reusable evaluation environments with multi-criteria reward functions
- Train models using GRPO or integrate with other RL frameworks

Key features:
- **Flexible multi-turn interactions** via `MultiTurnEnv` 
- **Native tool calling** support with `ToolEnv`
- **Modular reward functions** through rubrics
- **Environment modules** for easy sharing and reuse

## Installation

### Basic Installation

For evaluation and API model usage:
```bash
uv add verifiers
```

### Training Support

For GPU training with `vf.GRPOTrainer`:
```bash
uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation
```

### Latest Development Version

To use the latest `main` branch:
```bash
uv add verifiers @ git+https://github.com/willccbb/verifiers.git
```

### Development Setup

For contributing to verifiers:
```bash
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync --all-extras && uv pip install flash-attn --no-build-isolation
uv run pre-commit install
```

### Integration with prime-rl

For large-scale FSDP training, see [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Documentation

### Getting Started

**[Overview](overview.md)** — Core concepts and architecture. Start here if you're new to Verifiers to understand how environments orchestrate interactions.

**[Environments](environments.md)** — Creating custom interaction protocols with `MultiTurnEnv`, `ToolEnv`, and basic rubrics.

### Advanced Usage

**[Components](components.md)** — Advanced rubrics, tools, parsers, with practical examples. Covers judge rubrics, tool design, and complex workflows.

**[Training](training.md)** — GRPO training and hyperparameter tuning. Read this when you're ready to train models with your environments.

### Reference

**[Development](development.md)** — Contributing to verifiers

**[Type Reference](api_reference.md)** — Understanding data structures
