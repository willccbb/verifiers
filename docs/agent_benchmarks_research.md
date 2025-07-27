# Agent Benchmarks Research Report for 2025

## Executive Summary

Based on comprehensive research of popular LLM benchmarks in 2025, I've identified three leading candidates that emphasize agent-related tasks rather than simple QA. These benchmarks focus on tool use, multi-turn interactions, and complex reasoning in realistic scenarios.

## Top 3 Agent Benchmark Candidates

### 1. **τ-bench (TAU-bench)** - Tool-Agent-User Interaction
**Priority: HIGHEST**

- **Description**: Measures agents' ability to interact with simulated users and tools while following domain-specific policies
- **Key Features**:
  - Multi-turn conversations with dynamic user simulation
  - Complex tool use with APIs and databases
  - Policy compliance testing
  - Reliability metric (pass^k) for consistency
- **Domains**: Retail (115 tasks) and Airline (50 tasks)
- **GitHub**: https://github.com/sierra-research/tau-bench
- **Why Important**: 
  - Most realistic simulation of customer service scenarios
  - Already adopted by major AI labs (Anthropic, OpenAI)
  - Tests both reasoning AND multi-turn interaction
  - Has extension τ²-bench with telecom domain

### 2. **GAIA** - General AI Assistants
**Priority: HIGH**

- **Description**: Tests fundamental abilities like reasoning, multi-modality, web browsing, and tool proficiency
- **Key Features**:
  - 466 questions requiring multi-step reasoning
  - Three difficulty levels (1-3)
  - Unambiguous, factual answers
  - Tool use including web browsing, image analysis
- **GitHub**: https://github.com/UKGovernmentBEIS/inspect_evals (implementation)
- **Dataset**: HuggingFace with public validation set
- **Why Important**:
  - Widely adopted standard (used by OpenAI, Anthropic, H2O.ai)
  - Tests complex multi-hop reasoning
  - Simple evaluation framework
  - Human baseline: 92% vs best AI: ~74%

### 3. **ToolSandbox** - Stateful Tool Use
**Priority: MEDIUM**

- **Description**: Evaluates conversational agents with stateful tool execution and dependencies
- **Key Features**:
  - Stateful tool execution
  - Implicit state dependencies between tools
  - Built-in user simulator
  - Dynamic evaluation over trajectories
- **Paper**: ACL 2025 Findings
- **Why Important**:
  - Tests complex state management
  - More advanced than simple tool calling
  - Conversational evaluation

## Recommendation: Start with τ-bench

**Reasons**:
1. **Most Realistic**: Simulates actual customer service scenarios with policies
2. **Well-Documented**: Clear GitHub repo with examples
3. **Industry Adoption**: Used by leading AI companies
4. **Fits Verifiers Framework**: 
   - Uses tools similar to ToolEnv
   - Multi-turn like MultiTurnEnv
   - Has clear evaluation metrics
5. **Extensible**: Can add new domains beyond retail/airline

## Implementation Plan for τ-bench

### Phase 1: Core Infrastructure
```python
# Example structure for tau_bench environment
def load_environment(**kwargs) -> vf.Environment:
    # Load domain (retail/airline)
    domain = kwargs.get("domain", "retail")
    
    # Create tools based on domain
    tools = create_domain_tools(domain)
    
    # Load policy documents
    policies = load_policies(domain)
    
    # Create rubric with policy compliance
    rubric = create_tau_rubric(policies)
    
    return vf.ToolEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tools,
        max_turns=20
    )
```

### Phase 2: Key Components
1. **User Simulator**: LLM-based user that follows scenarios
2. **Policy Checker**: Ensures agent follows domain rules
3. **State Verifier**: Checks database state after completion
4. **Reliability Evaluator**: Implements pass^k metric

### Phase 3: Integration Points
- Leverage existing ToolEnv for tool interactions
- Use MultiTurnEnv patterns for conversation flow
- Adapt evaluation metrics to verifiers format

## Next Steps

1. **Download and analyze** τ-bench official implementation
2. **Map components** to verifiers framework
3. **Create prototype** with retail domain first
4. **Test with GPT-4** to establish baseline
5. **Iterate and expand** to other domains

## Additional Notes

- GAIA would be excellent as a second benchmark - simpler to implement
- Consider creating a unified evaluation suite with multiple benchmarks
- Focus on verbatim translation of logic, not reimplementation