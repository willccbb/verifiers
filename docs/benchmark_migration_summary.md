# LLM Agent Benchmark Migration Summary

## Executive Summary

After researching popular LLM benchmarks for 2025 focusing on agent-related tasks, I've identified **τ-bench (TAU-bench)** as the best candidate for migration to the verifiers infrastructure. This benchmark emphasizes real-world agent capabilities including multi-turn interactions, tool usage, and policy compliance.

## Selected Benchmark: τ-bench

### Why τ-bench?

1. **Real-world relevance**: Simulates actual customer service scenarios
2. **Industry adoption**: Used by Anthropic, OpenAI, and other major AI labs
3. **Comprehensive testing**: Covers tool use, user interaction, and policy compliance
4. **Clear evaluation**: Objective database state verification
5. **Extensible design**: Easy to add new domains and tasks

### Key Features

- **Domains**: Retail (115 tasks) and Airline (50 tasks)
- **Components**: 
  - 17 tools for retail domain
  - Dynamic user simulation
  - Policy compliance checking
  - Reliability metric (pass^k)
- **Evaluation**: State-based verification, not just conversation quality

## Implementation Approach

### Phase 1: Core Infrastructure (Week 1)
1. Create environment directory structure
2. Implement base MultiTurnEnv for τ-bench
3. Port 5-10 core tools (find_user, get_order, etc.)
4. Set up basic dataset loading

### Phase 2: User Simulation (Week 2)
1. Implement LLM-based user simulator
2. Create scenario-based response generation
3. Add state tracking for conversations
4. Test with simple interactions

### Phase 3: Full Implementation (Week 3-4)
1. Port all 17 retail tools
2. Implement policy compliance checking
3. Add pass^k reliability metric
4. Create comprehensive test suite

## Technical Architecture

```
environments/tau_bench_retail/
├── __init__.py
├── tau_bench_retail.py     # Main environment class
├── tools.py                # All tool implementations
├── user_simulator.py       # User response generation
├── policies.py             # Policy compliance logic
├── data/
│   ├── tasks.json         # Task definitions
│   ├── policies.json      # Domain policies
│   └── database.json      # Mock database
└── pyproject.toml
```

## Key Implementation Details

1. **Environment**: Extend `MultiTurnEnv` with user simulation
2. **Tools**: Convert from τ-bench format to `BaseTool`
3. **Evaluation**: State-based checking with policy compliance
4. **Dataset**: Convert tasks to verifiers dataset format

## Success Metrics

- [ ] All 17 retail tools implemented
- [ ] User simulator generates realistic responses
- [ ] Policy compliance detection works
- [ ] Pass^k metric matches original benchmark
- [ ] GPT-4 achieves ~45% on retail domain

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complex user simulation | Start with template-based, enhance with LLM |
| State management complexity | Use structured state dict with clear schema |
| Policy parsing challenges | Begin with simple rules, expand gradually |
| Performance differences | Compare with original implementation regularly |

## Next Immediate Steps

1. **Today**: Set up environment structure
2. **Tomorrow**: Implement first 3 tools
3. **This week**: Get basic task running end-to-end
4. **Next week**: Add user simulation

## Alternative Benchmarks

If τ-bench proves too complex initially:

1. **GAIA**: Simpler single-turn tasks with tool use
2. **ToolSandbox**: Focuses on stateful tool execution

## Recommendation

**Proceed with τ-bench implementation** as it provides the most comprehensive test of agent capabilities and aligns well with verifiers' focus on multi-turn, tool-using environments. The verbatim translation approach ensures we maintain benchmark integrity while leveraging verifiers' infrastructure.

## Resources

- Original repo: https://github.com/sierra-research/tau-bench
- Paper: https://arxiv.org/abs/2406.12045
- Leaderboard: https://hal.cs.princeton.edu/taubench_retail

## Contact

For questions about this migration, refer to:
- `docs/agent_benchmarks_research.md` - Full research
- `docs/tau_bench_implementation_guide.md` - Detailed implementation
- Original τ-bench team: research@sierra.ai