# Verifiers Documentation Improvement Plan

## Executive Summary

After analyzing the current documentation against the actual codebase and "golden" examples, I've identified significant discrepancies that make the documentation misleading and potentially unusable. This plan outlines the key issues and recommended fixes.

## Major Issues Identified

### 1. **Outdated API Patterns**
**Problem**: Documentation shows complex, verbose constructor patterns that don't match actual usage.

**Documentation shows**:
```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

parser = XMLParser(fields=["think", "answer"])
rubric = Rubric(funcs=[...], weights=[...])
env = SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, ...)
```

**Actual usage**:
```python
import verifiers as vf

vf_env = vf.SingleTurnEnv(dataset=dataset, system_prompt=system_prompt, ...)
# or
vf_env = vf.ToolEnv(dataset=dataset, tools=[python], ...)
```

**Impact**: Users following documentation will get import errors and verbose, unnecessary code.

### 2. **Missing Environment Types**
**Problem**: Documentation only covers basic environments, missing the specialized ones actually used.

**Documented**: `SingleTurnEnv`, `MultiTurnEnv`, basic `ToolEnv`
**Actually used**: `ToolEnv`, `TextArenaEnv`, `ReasoningGymEnv`, `SmolaToolEnv`, `DoubleCheckEnv`

**Impact**: Users can't find documentation for the environments they see in examples.

### 3. **Parser Misrepresentation**
**Problem**: Documentation focuses heavily on XMLParser, but actual examples use different parsers.

**Documentation emphasizes**: `XMLParser(fields=["reasoning", "answer"])`
**Actually used**: `ThinkParser`, `SmolaParser`, built-in parsers with defaults

**Impact**: Users implement unnecessary XML parsing when simpler solutions exist.

### 4. **Training API Mismatch**
**Problem**: Documentation shows complex training setup, but actual usage is much simpler.

**Documentation shows**:
```python
# Complex setup with manual configuration
training_args = TrainingArguments(...)
trainer = GRPOTrainer(model=model, args=training_args, ...)
```

**Actual usage**:
```python
args = vf.grpo_defaults(run_name="my-experiment")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
```

**Impact**: Users write unnecessarily complex training code.

### 5. **Installation and Import Issues**
**Problem**: Documentation shows pip-based installation, but project uses uv.

**Documentation shows**: `pip install verifiers`
**Project uses**: `uv` for package management

**Impact**: Users may have dependency management issues.

### 6. **Outdated Examples**
**Problem**: Documentation examples are elaborate but don't match actual working code.

**Examples in docs**: Complex multi-file implementations with custom classes
**Actual examples**: Simple, concise scripts using built-in functionality

**Impact**: Users implement overly complex solutions.

### 7. **Missing Key Concepts**
**Problem**: Documentation misses crucial concepts used in actual examples.

**Missing**:
- `load_example_dataset()` usage
- `vf.get_model_and_tokenizer()` 
- vLLM server integration
- Accelerate/multi-GPU setup
- `vf.lora_defaults()` for PEFT

**Impact**: Users can't reproduce working examples.

## Recommended Actions

### Phase 1: Critical API Fixes (High Priority)

1. **Fix Installation Instructions**
   - Remove pip references
   - Add uv installation instructions
   - Update quick start to use uv

2. **Update Import Patterns**
   - Change all examples to use `import verifiers as vf`
   - Remove verbose individual imports
   - Show the actual import pattern used in examples

3. **Simplify Quick Start**
   - Replace complex example with actual working code from examples
   - Use `vf.grpo_defaults()` and simple APIs
   - Focus on getting something working quickly

### Phase 2: Environment Documentation (High Priority)

1. **Add Missing Environment Types**
   - Document `ToolEnv`, `TextArenaEnv`, `ReasoningGymEnv`, `SmolaToolEnv`, `DoubleCheckEnv`
   - Show actual constructor signatures with defaults
   - Remove or significantly reduce `SingleTurnEnv` documentation

2. **Update Environment Examples**
   - Replace complex custom implementations with actual usage
   - Show how environments handle parsers and rubrics automatically
   - Demonstrate built-in dataset loading

### Phase 3: Parser and Rubric Simplification (Medium Priority)

1. **Downplay XMLParser**
   - Move XMLParser to advanced section
   - Lead with `ThinkParser` and automatic parser selection
   - Show how environments choose parsers automatically

2. **Simplify Rubric Documentation**
   - Show how rubrics are created automatically by environments
   - Focus on when you need custom rubrics vs. using defaults
   - Remove complex multi-criteria examples unless necessary

### Phase 4: Training Documentation (Medium Priority)

1. **Update Training Examples**
   - Use `vf.grpo_defaults()` as the primary pattern
   - Show vLLM server setup
   - Document accelerate/multi-GPU patterns from examples
   - Add PEFT/LoRA examples

2. **Add Infrastructure Documentation**
   - vLLM server setup and usage
   - Multi-GPU training patterns
   - Model loading best practices

### Phase 5: Example Alignment (Low Priority)

1. **Replace Verbose Examples**
   - Replace documentation examples with actual working code from examples/
   - Ensure every example can be run as-is
   - Focus on practical, common use cases

2. **Add Real-World Patterns**
   - Show actual training commands with GPU settings
   - Document common model/dataset combinations
   - Include evaluation and dataset creation patterns

## Files to Update

### Critical (Phase 1)
- `docs/source/index.md` - Fix installation, quick start
- `docs/source/overview.md` - Update API patterns
- `docs/source/examples.md` - Replace with actual working examples

### High Priority (Phase 2)
- `docs/source/environments.md` - Add missing environment types
- `docs/source/training.md` - Update training patterns

### Medium Priority (Phase 3-4)
- `docs/source/parsers.md` - Simplify parser documentation
- `docs/source/rubrics.md` - Focus on automatic rubric creation
- `docs/source/tools.md` - Update tool integration patterns

### Low Priority (Phase 5)
- `docs/source/advanced.md` - Move complex examples here
- `docs/source/api_reference.md` - Update API signatures

## Validation Strategy

1. **Test All Examples**: Every code example in documentation should be runnable
2. **Cross-reference with Examples**: Ensure documentation matches `verifiers/examples/`
3. **User Testing**: Documentation should enable users to reproduce example results
4. **Incremental Updates**: Update in phases to avoid breaking existing users

## Success Metrics

- Documentation examples match actual working code
- Users can follow quick start and get working results
- Zero import errors from documentation examples
- Environment documentation covers all actually-used environments
- Training examples work with current infrastructure

## Next Steps

1. Start with Phase 1 (Critical API Fixes)
2. Update one section at a time
3. Test each update against working examples
4. Remove outdated content rather than trying to fix everything
5. Focus on what users actually need vs. comprehensive coverage

This plan prioritizes fixing the most critical issues that prevent users from getting started, then systematically addresses the broader documentation-reality gap.