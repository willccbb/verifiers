# Test Plan

This document outlines initial testing coverage for the project. The focus is on
`envs`, `parsers`, and `rubrics` as used in example/eval workflows. Trainer and
inference code will be covered later since they require GPUs.

## Environments

- **Initialization tests**
  - Verify that each environment class can be constructed with default
    arguments.
  - Check that each environment is registered in `verifiers.envs` so they can be instantiated by name.
  - Ensure optional parameters propagate correctly to the base `Environment`
    class.

- **Rollout tests**
  - Use a mock `OpenAI` client to simulate API responses.
  - Confirm that `SingleTurnEnv.rollout` and other environment rollout methods
    return the expected completion format and state.
  - Validate that environment `evaluate` helpers (e.g. `ReasoningGymEnv`) return
    dictionaries with the correct keys (`prompt`, `completion`, `answer`,
    `reward`, etc.).

- **Dataset generation**
  - For environments that support `make_dataset`, verify that the resulting
    `datasets.Dataset` contains all expected columns and sample counts.
  - Confirm that all dataset columns match the schema expected by `DatasetInfo`.

## Parsers

- **Parser behavior**
  - Test the base `Parser` class methods (`parse`, `parse_answer`) using simple
    text and message inputs.
  - For `XMLParser`, validate parsing of well-formed XML snippets and handling
    of missing fields or multiple tag alternatives.
  - Ensure `get_format_str` produces the expected format description.
  - Validate error handling when required tags are missing.
  - Test the reward function returned by `get_format_reward_func` on properly
    and improperly formatted messages.

## Rubrics

- **Reward function invocation**
  - Verify that `Rubric` correctly stores reward functions and weights.
  - Test `_call_reward_func` with functions requiring different argument
    subsets to ensure selective argument passing works.
  - Confirm weighted sums combine individual reward functions correctly.

- **Scoring helpers**
  - Use simple synchronous reward functions to test `score_rollout` and
    `score_rollouts` without needing async OpenAI calls.
  - Confirm that aggregated reward dictionaries contain per-function scores and a summed `reward` key.
  - Ensure metrics returned by `score_rollouts` match expectations for aggregated scores.

## Examples/Evals Integration

- Create small dummy prompts/completions mirroring those in `verifiers/examples/eval`.
- Test end-to-end evaluation using mocks so that environments, parsers and rubrics interact together as they would during example runs.
- Verify that final scores match expected results for these mini-evals.

