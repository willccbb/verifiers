# Luanti Code Generation Environment

**First Luanti (Minetest) code generation environment for the verifiers framework**

## Overview

This environment evaluates Large Language Models on their ability to generate, fix, and modify Lua code for the Luanti (formerly Minetest) game engine. It covers four main task categories:

- **Scaffold**: Create new nodes, tools, entities, and recipes from scratch
- **Repair**: Fix broken or incomplete existing code  
- **Refactor**: Modify existing code (change properties, add features)
- **Documentation**: Use and explain Minetest API features

## Installation

```bash
# Install the environment  
uv add environments/luanti

# Or install in development mode
uv pip install -e environments/luanti
```

## Usage

```bash
# Quick evaluation with sample data
vf-eval luanti --num-examples 10 --rollouts-per-example 3

# Evaluation with custom dataset
vf-eval luanti --num-examples 60 --rollouts-per-example 5 \
  --sampling-args '{"temperature":0.2,"max_tokens":300}'

# View results
vf-tui
```

## Environment Details

### Task Categories

1. **Scaffold Tasks** (Create new code)
   ```
   Create a Luanti node called 'glowing_crystal' that emits light level 12
   ```

2. **Repair Tasks** (Fix broken code)
   ```
   Fix this broken code: minetest.register_node('test', {description = 'Test'})
   ```

3. **Refactor Tasks** (Modify existing code)
   ```  
   Change the light level from 8 to 14 in this lamp node
   ```

4. **Documentation Tasks** (API usage and explanation)
   ```
   Create a node using paramtype = 'light' and explain what it does
   ```

### Scoring Rubric

The environment uses a comprehensive scoring system:

- **Syntax (30%)**: Valid Lua, balanced braces, no syntax errors
- **API Usage (40%)**: Correct `minetest.register_*` calls, proper properties  
- **Task Completion (30%)**: Addresses prompt requirements correctly

### Performance Benchmarks

**Baseline Results** (Standard LLMs):
- GPT-4: ~28.33% pass@5
- Other models: Significantly lower

**Fine-tuned Results** (Our specialized model):
- Pass@1: 93.3%
- Pass@5: 100%  
- Improvement: +71.7pp vs baseline

## Dataset

The environment can work with:
- **Sample data** (3 embedded tasks for testing)
- **Full dataset** (60+ comprehensive evaluation items)
- **Custom datasets** (JSONL format with instruction/expected patterns)

### Dataset Format

```json
{
  "instruction": "Create a Luanti node that emits light level 12",
  "any_of": ["light_source = 12", "minetest.register_node"],
  "must_not_contain": ["error", "undefined"],
  "task_type": "scaffold"
}
```

## Integration

### With Prime-RL

This environment is compatible with Prime-RL for distributed training:

```toml
[environment] 
id = "luanti"
```

### With GRPO Training

Use with verifiers' GRPO trainer for reinforcement learning:

```python
import verifiers as vf

env = vf.load_environment("luanti")
trainer = vf.GRPOTrainer(env=env, model="your-model")
trainer.train()
```

## Technical Notes

### Luanti API Patterns

Common patterns the environment recognizes:
- `minetest.register_node()` - Block/node definitions
- `minetest.register_tool()` - Tool definitions  
- `minetest.register_craft()` - Crafting recipes
- `light_source` - Light emission (0-14)
- `groups` - Block behavior groupings
- `tiles` - Texture specifications

### Code Quality Checks

The environment validates:
- Proper Lua table syntax
- Balanced parentheses and braces
- Correct Minetest API usage
- Logical property combinations
- Code readability and structure

## Results

This environment has been used to demonstrate significant improvements in Luanti code generation through fine-tuning, achieving 100% pass@5 vs 28.33% baseline performance.

## Contributing

This environment serves as an example for:
- Game engine modding domain evaluation
- Lua code generation assessment  
- Multi-category task evaluation (scaffold/repair/refactor/docs)
- Integration between specialized domains and RL frameworks

## License

Apache 2.0 - Safe for commercial and research use.

## Citation

If you use this environment, please cite:

```
Luanti Code Generation Environment for Verifiers Framework
https://github.com/toddllm/luanti-fine-tune
First specialized environment for game engine modding evaluation
```