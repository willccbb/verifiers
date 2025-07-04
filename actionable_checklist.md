# Actionable Checklist for Documentation Fixes

## Phase 1: Critical Fixes (Must Do First)

### `docs/source/index.md`
- [ ] Change `pip install verifiers` to `uv add verifiers`
- [ ] Replace all `from verifiers.envs import` with `import verifiers as vf`
- [ ] Replace quick start example with working code from `verifiers/examples/gsm8k.py`
- [ ] Test that the quick start example actually runs

### `docs/source/overview.md`
- [ ] Change primary parser example from `XMLParser` to `ThinkParser`
- [ ] Replace complex constructor examples with simple `vf.SingleTurnEnv(...)` patterns
- [ ] Add `vf.load_example_dataset()` and `vf.get_model_and_tokenizer()` examples
- [ ] Test all code examples for syntax errors

### `docs/source/examples.md`
- [ ] Remove all fictional examples (MathEnv class, etc.)
- [ ] Copy actual working examples from `verifiers/examples/`
- [ ] Add these missing examples:
  - [ ] `ToolEnv` example (from math_python.py)
  - [ ] `TextArenaEnv` example (from wordle.py)
  - [ ] `ReasoningGymEnv` example (from arc_1d.py)
  - [ ] `SmolaToolEnv` example (from smola_math_tools.py)
  - [ ] `DoubleCheckEnv` example (from doublecheck.py)

## Phase 2: Core Content Updates

### `docs/source/environments.md`
- [ ] Add documentation for missing environment types:
  - [ ] `ToolEnv` - show constructor signature and usage
  - [ ] `TextArenaEnv` - show game integration pattern
  - [ ] `ReasoningGymEnv` - show reasoning gym integration
  - [ ] `SmolaToolEnv` - show smolagents integration
  - [ ] `DoubleCheckEnv` - show verification pattern
- [ ] Update environment hierarchy diagram
- [ ] Replace custom inheritance examples with direct usage examples
- [ ] Show automatic parser/rubric selection patterns

### `docs/source/training.md`
- [ ] Replace complex training setup with `vf.grpo_defaults()` pattern
- [ ] Add vLLM server setup instructions
- [ ] Add multi-GPU setup examples from actual training scripts
- [ ] Show `vf.lora_defaults()` usage for PEFT training
- [ ] Remove fictional training examples
- [ ] Add actual training commands with GPU environment variables

### `docs/source/parsers.md`
- [ ] Move `XMLParser` to advanced section
- [ ] Lead with `ThinkParser` as primary example
- [ ] Add `SmolaParser` documentation
- [ ] Show automatic parser selection by environments
- [ ] Remove overly complex parsing examples

### `docs/source/rubrics.md`
- [ ] Show how environments create rubrics automatically
- [ ] Add built-in rubric types: `MathRubric`, `ToolRubric`, `SmolaToolRubric`
- [ ] Simplify multi-criteria examples
- [ ] Focus on when custom rubrics are needed vs defaults

### `docs/source/tools.md`
- [ ] Add `verifiers.tools` module documentation
- [ ] Show built-in tools: `python`, `calculator`
- [ ] Add SmolaAgents integration examples
- [ ] Show tool schema inference
- [ ] Replace complex tool examples with actual usage

## Phase 3: Comprehensive Updates

### `docs/source/advanced.md`
- [ ] Move complex examples here (multi-criteria rubrics, custom environments)
- [ ] Add infrastructure patterns (vLLM, distributed training)
- [ ] Document async batch processing
- [ ] Show environment composition patterns

### `docs/source/api_reference.md`
- [ ] Update all API signatures to match actual code
- [ ] Add missing classes and methods
- [ ] Remove deprecated or fictional APIs
- [ ] Add proper type hints

### Global Changes
- [ ] Find and replace all import statements:
  - [ ] `from verifiers.envs import` → `import verifiers as vf`
  - [ ] `from verifiers.parsers import` → `import verifiers as vf`
  - [ ] `from verifiers.rubrics import` → `import verifiers as vf`
  - [ ] `from verifiers.trainers import` → `import verifiers as vf`
- [ ] Update all environment constructors to use `vf.` prefix
- [ ] Update all parser/rubric usage to use `vf.` prefix
- [ ] Update all training examples to use `vf.` prefix

## Validation Checklist

For each documentation file:
- [ ] All code examples are syntactically correct
- [ ] All imports work without errors
- [ ] All examples can be copy-pasted and run
- [ ] All referenced classes/functions exist in the codebase
- [ ] All examples match patterns used in `verifiers/examples/`

## Testing Strategy

1. **Create test script** that runs all documentation examples
2. **Set up clean environment** with uv
3. **Test installation** following documentation
4. **Run each code example** and verify it works
5. **Compare with actual examples** in `verifiers/examples/`

## Files to Delete or Significantly Reduce

Consider removing/reducing these sections if they can't be easily fixed:
- [ ] Complex multi-turn examples that don't match actual usage
- [ ] Fictional custom environment classes
- [ ] Overly detailed parser/rubric setup that's not needed
- [ ] Advanced patterns that aren't used in practice

## Success Criteria

- [ ] New user can follow quick start and get working result
- [ ] All environment types used in examples are documented
- [ ] All code examples in docs are runnable
- [ ] Zero import errors from following documentation
- [ ] Documentation matches actual API patterns

## Priority Order

1. **Start with `index.md`** - fix installation and quick start
2. **Fix `examples.md`** - replace with actual working examples
3. **Update `environments.md`** - add missing environment types
4. **Fix `training.md`** - update training patterns
5. **Simplify other sections** - remove complex examples, focus on basics

This checklist ensures that the most critical user-blocking issues are fixed first, followed by comprehensive content updates.