"""Tests for the Rubric class."""

from typing import cast

import pytest

from verifiers import Parser, Rubric
from verifiers.types import RewardFunc


class TestRubric:
    """Test cases for the Rubric class."""

    def test_rubric_initialization_empty(self):
        """Test Rubric initialization with no parameters."""
        rubric = Rubric()

        assert rubric.reward_funcs == []
        assert rubric.reward_weights == []
        assert isinstance(rubric.parser, Parser)

    def test_rubric_initialization_with_functions(self):
        """Test Rubric initialization with reward functions."""

        def reward_func1(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def reward_func2(completion, **kwargs):
            return len(completion) * 0.1

        funcs = cast(list[RewardFunc], [reward_func1, reward_func2])
        weights = [1.0, 0.5]

        rubric = Rubric(funcs=funcs, weights=weights)

        assert rubric.reward_funcs == funcs
        assert rubric.reward_weights == weights
        assert len(rubric.get_reward_func_names()) == 2
        assert rubric.get_reward_func_names() == ["reward_func1", "reward_func2"]

    def test_rubric_initialization_functions_without_weights(self):
        """Test Rubric initialization with functions but no explicit weights."""

        def reward_func1(completion, **kwargs) -> float:
            return 1.0

        def reward_func2(completion, **kwargs) -> float:
            return 0.5

        funcs = cast(list[RewardFunc], [reward_func1, reward_func2])

        rubric = Rubric(funcs=funcs)

        assert rubric.reward_funcs == funcs
        assert rubric.reward_weights == [1.0, 1.0]  # Default weights

    def test_rubric_initialization_with_kwargs(self):
        """Test Rubric initialization with additional kwargs."""
        rubric = Rubric(custom_param="test_value", another_param=42)

        assert rubric.custom_param == "test_value"  # type: ignore
        assert rubric.another_param == 42  # type: ignore

    def test_add_reward_func(self):
        """Test adding reward functions."""
        rubric = Rubric(funcs=[], weights=[])

        def test_func(completion, **kwargs):
            return 1.0

        rubric.add_reward_func(test_func, weight=0.8)

        assert len(rubric.reward_funcs) == 1
        assert rubric.reward_funcs[0] == test_func
        assert rubric.reward_weights == [0.8]
        assert rubric.get_reward_func_names() == ["test_func"]

    def test_add_multiple_reward_funcs(self):
        """Test adding multiple reward functions."""
        # Create fresh rubric to avoid test isolation issues
        rubric = Rubric(funcs=[], weights=[])

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric.add_reward_func(func1, weight=1.0)
        rubric.add_reward_func(func2, weight=0.3)

        assert len(rubric.reward_funcs) == 2
        assert rubric.get_reward_func_names() == ["func1", "func2"]
        assert rubric.reward_weights == [1.0, 0.3]

    def test_add_reward_func_default_weight(self):
        """Test adding reward function with default weight."""
        rubric = Rubric(funcs=[], weights=[])

        def test_func(completion, **kwargs):
            return 1.0

        rubric.add_reward_func(test_func)

        assert rubric.reward_weights == [1.0]

    def test_get_methods(self):
        """Test getter methods."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric = Rubric(funcs=[func1, func2], weights=[0.8, 0.2])

        assert rubric.get_reward_funcs() == [func1, func2]
        assert rubric.get_reward_weights() == [0.8, 0.2]
        assert rubric.get_reward_func_names() == ["func1", "func2"]

    @pytest.mark.asyncio
    async def test_call_reward_func_with_all_args(self):
        """Test calling reward function with all possible arguments."""

        def comprehensive_func(prompt, completion, answer, state, task, info, **kwargs):
            return len(completion) + len(answer) + len(task)

        rubric = Rubric(funcs=[], weights=[])

        result = await rubric.call_reward_func(
            func=comprehensive_func,
            parser=Parser(),
            prompt="test prompt",
            completion="test completion",
            answer="test answer",
            state={"key": "value"},
            task="test task",
            info={"info_key": "info_value"},
        )

        # len("test completion") + len("test answer") + len("test task")
        expected = len("test completion") + len("test answer") + len("test task")
        assert result == expected

    @pytest.mark.asyncio
    async def test_call_reward_func_with_subset_args(self):
        """Test calling reward function that only uses some arguments."""

        def simple_func(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        rubric = Rubric(funcs=[], weights=[])

        result = await rubric.call_reward_func(
            func=simple_func,
            parser=Parser(),
            prompt="irrelevant",
            completion="same",
            answer="same",
            state={},
            task="irrelevant",
            info={},
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_call_reward_func_with_var_kwargs(self):
        """Test calling reward function that accepts **kwargs."""

        def kwargs_func(completion, **kwargs):
            return len(kwargs)

        rubric = Rubric(funcs=[], weights=[])

        result = await rubric.call_reward_func(
            func=kwargs_func,
            parser=Parser(),
            prompt="test",
            completion="test",
            answer="test",
            state={},
            task="test",
            info={},
        )

        # Should receive parser, prompt, answer, state, task, info (completion used directly)
        assert result == 6

    @pytest.mark.asyncio
    async def test_call_reward_func_error_handling(self):
        """Test error handling in reward function calls."""

        def error_func(completion, **kwargs):
            raise ValueError("Test error")

        rubric = Rubric(funcs=[], weights=[])

        result = await rubric.call_reward_func(
            func=error_func,
            parser=Parser(),
            prompt="test",
            completion="test",
            answer="test",
            state={},
            task="test",
            info={},
        )

        assert result == 0.0  # Should return 0.0 on error

    @pytest.mark.asyncio
    async def test_score_rollout_single(self):
        """Test scoring a single rollout."""

        def func1(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def func2(completion, **kwargs):
            return len(completion) * 0.1

        rubric = Rubric(funcs=[func1, func2], weights=[1.0, 0.5])

        result = await rubric.score_rollout(
            prompt="test prompt",
            completion="test",
            answer="test",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="test_task",
            info={},
        )

        assert "func1" in result.metrics
        assert "func2" in result.metrics
        assert hasattr(result, "reward")
        assert result.metrics["func1"] == 1.0  # completion == answer
        assert result.metrics["func2"] == 0.4  # len("test") * 0.1
        assert result.reward == 1.0 * 1.0 + 0.4 * 0.5  # Weighted sum

    @pytest.mark.asyncio
    async def test_score_rollout_with_list_completion(self):
        """Test scoring rollout with list-type completion."""

        def list_func(completion, **kwargs):
            return len(completion) if isinstance(completion, list) else 0.0

        rubric = Rubric(funcs=[list_func])

        completion = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await rubric.score_rollout(
            prompt="test",
            completion=completion,  # type: ignore
            answer="test",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="test",
            info={},
        )

        assert result.metrics["list_func"] == 2.0  # Length of completion list
        assert result.reward == 2.0

    @pytest.mark.asyncio
    async def test_score_rollouts_multiple(self):
        """Test scoring multiple rollouts."""

        def accuracy_func(completion, answer, **kwargs):
            return 1.0 if completion == answer else 0.0

        def length_func(completion, **kwargs):
            return len(str(completion))

        rubric = Rubric(funcs=[accuracy_func, length_func], weights=[1.0, 0.1])

        prompts = ["prompt1", "prompt2", "prompt3"]
        completions = ["answer1", "answer2", "wrong"]
        answers = ["answer1", "answer2", "answer3"]
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}},
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}},
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}},
        ]
        tasks = ["task1", "task2", "task3"]
        infos = [{}, {}, {}]

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
        )

        assert "accuracy_func" in results.metrics
        assert "length_func" in results.metrics
        assert hasattr(results, "reward")
        assert len(results.metrics["accuracy_func"]) == 3
        assert results.metrics["accuracy_func"] == [
            1.0,
            1.0,
            0.0,
        ]  # First two match, third doesn't
        assert results.metrics["length_func"] == [
            7.0,
            7.0,
            5.0,
        ]  # Lengths of completions

    @pytest.mark.asyncio
    async def test_score_rollouts_with_weights(self):
        """Test scoring rollouts applies reward function weights correctly."""

        def func1(completion, **kwargs):
            return 1.0

        def func2(completion, **kwargs):
            return 0.5

        rubric = Rubric(funcs=[func1, func2], weights=[2.0, 3.0])

        prompts = ["test"]
        completions = ["test"]
        answers = ["test"]
        tasks = ["test"]
        infos = [{}]

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=[
                {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
            ],
            tasks=tasks,
            infos=infos,
        )

        # Weighted sum: 1.0 * 2.0 + 0.5 * 3.0 = 2.0 + 1.5 = 3.5
        assert results.reward[0] == 3.5
        # Individual metrics should not be weighted
        assert results.metrics["func1"][0] == 1.0
        assert results.metrics["func2"][0] == 0.5

    @pytest.mark.asyncio
    async def test_score_rollouts_empty(self):
        """Test scoring empty list of rollouts."""

        def test_func(completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[test_func], weights=[1.0])

        # Should handle empty rollouts gracefully
        results = await rubric.score_rollouts(
            prompts=[], completions=[], answers=[], states=[], tasks=[], infos=[]
        )

        # Should return empty lists for each function
        assert "test_func" in results.metrics
        assert hasattr(results, "reward")
        assert results.metrics["test_func"] == []
        assert results.reward == []

    @pytest.mark.asyncio
    async def test_score_rollouts_with_default_infos(self):
        """Test scoring rollouts with default empty infos."""

        def simple_func(completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[simple_func], weights=[1.0])

        results = await rubric.score_rollouts(
            prompts=["test"],
            completions=["test"],
            answers=["test"],
            states=[
                {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
            ],
            tasks=["test"],
            infos=[{}],  # Explicitly provide infos to match other lists
        )

        assert "simple_func" in results.metrics
        assert results.metrics["simple_func"] == [1.0]

    def test_rubric_with_custom_parser(self):
        """Test Rubric with custom parser."""
        custom_parser = Parser()
        rubric = Rubric(funcs=[], weights=[], parser=custom_parser)

        assert rubric.parser is custom_parser

    @pytest.mark.asyncio
    async def test_score_rollouts_with_mixed_return_types(self):
        """Test scoring when reward functions return different types."""

        def scalar_func(completion, **kwargs):
            return 0.5

        def list_func(completion, **kwargs):
            # This should not happen, but test robustness
            return [0.1, 0.2]  # Wrong return type

        rubric = Rubric(funcs=[scalar_func], weights=[1.0])

        results = await rubric.score_rollouts(
            prompts=["test"],
            completions=["test"],
            answers=["test"],
            states=[
                {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
            ],
            tasks=["test"],
            infos=[{}],
        )

        assert results.metrics["scalar_func"] == [0.5]
        assert results.reward == [0.5]

    @pytest.mark.asyncio
    async def test_call_reward_func_kwargs_filtering(self):
        """Test that functions without **kwargs get filtered kwargs."""

        def f_no_kwargs(completion, answer):
            return 0.5

        def f_with_kwargs(completion, **kwargs):
            assert kwargs.get("extra") == 123
            return 1.0

        rubric = Rubric(funcs=[f_no_kwargs, f_with_kwargs], weights=[1.0, 2.0])

        result = await rubric.score_rollout(
            prompt=[{"role": "user", "content": "q"}],
            completion=[{"role": "assistant", "content": "a"}],
            answer="ans",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="default",
            info={},
            extra=123,
        )

        # Weighted sum: 0.5*1 + 1.0*2 = 2.5
        assert result.reward == pytest.approx(2.5)
        assert set(result.metrics.keys()) == {"f_no_kwargs", "f_with_kwargs"}

    @pytest.mark.asyncio
    async def test_score_rollout_serial_execution_order(self):
        """Test that serial mode respects execution order."""
        calls = []

        def g1(**kwargs):
            calls.append("g1")
            return 0.2

        def g2(**kwargs):
            calls.append("g2")
            return 0.3

        rubric = Rubric(funcs=[g1, g2], weights=[1.0, 1.0], parallelize_scoring=False)

        result = await rubric.score_rollout(
            prompt="q",
            completion="a",
            answer="ans",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="default",
        )

        assert result.reward == pytest.approx(0.5)
        assert calls == ["g1", "g2"]  # serial order respected

    @pytest.mark.asyncio
    async def test_call_reward_func_error_handling_both_paths(self):
        """Test error handling for both kwargs and no-kwargs functions."""

        def error_func_no_kwargs(completion, answer):
            raise ValueError("Test error without kwargs")

        def error_func_with_kwargs(completion, **kwargs):
            raise RuntimeError("Test error with kwargs")

        rubric = Rubric()

        # Test both error paths return 0.0
        result1 = await rubric.call_reward_func(
            func=error_func_no_kwargs,
            parser=rubric.parser,
            prompt="test",
            completion="test",
            answer="test",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="test",
            info={},
        )

        result2 = await rubric.call_reward_func(
            func=error_func_with_kwargs,
            parser=rubric.parser,
            prompt="test",
            completion="test",
            answer="test",
            state={
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}
            },
            task="test",
            info={},
        )

        assert result1 == 0.0
        assert result2 == 0.0

    @pytest.mark.asyncio
    async def test_group_transformation_standardize(self):
        """Test standardize_groups transformation."""

        def simple_reward(completion, **kwargs):
            return float(completion)

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        # Create rollouts with 2 groups of 3 rollouts each
        # Group 1: [1.0, 2.0, 3.0] -> mean=2.0, std=0.816
        # Group 2: [4.0, 5.0, 6.0] -> mean=5.0, std=0.816
        prompts = ["p1", "p1", "p1", "p2", "p2", "p2"]
        completions = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]
        answers = [""] * 6
        tasks = ["test"] * 6
        infos = [{}] * 6
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 6

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform="standardize",
        )

        # After standardization, each group should have mean≈0 and std≈1
        rewards = results.reward
        group1 = rewards[:3]
        group2 = rewards[3:]

        # Check that means are close to 0
        assert abs(sum(group1) / 3) < 1e-10
        assert abs(sum(group2) / 3) < 1e-10

        # Check relative ordering is preserved
        assert group1[0] < group1[1] < group1[2]
        assert group2[0] < group2[1] < group2[2]

    @pytest.mark.asyncio
    async def test_group_transformation_normalize(self):
        """Test normalize_groups transformation (min-max normalization)."""

        def simple_reward(completion, **kwargs):
            return float(completion)

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        # Create rollouts with 2 groups
        # Group 1: [1.0, 2.0, 3.0] -> normalized to [0.0, 0.5, 1.0]
        # Group 2: [10.0, 20.0, 30.0] -> normalized to [0.0, 0.5, 1.0]
        prompts = ["p1", "p1", "p1", "p2", "p2", "p2"]
        completions = ["1.0", "2.0", "3.0", "10.0", "20.0", "30.0"]
        answers = [""] * 6
        tasks = ["test"] * 6
        infos = [{}] * 6
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 6

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform="normalize",
        )

        rewards = results.reward

        # Both groups should be normalized to [0.0, 0.5, 1.0]
        assert pytest.approx(rewards[0], abs=1e-6) == 0.0
        assert pytest.approx(rewards[1], abs=1e-6) == 0.5
        assert pytest.approx(rewards[2], abs=1e-6) == 1.0
        assert pytest.approx(rewards[3], abs=1e-6) == 0.0
        assert pytest.approx(rewards[4], abs=1e-6) == 0.5
        assert pytest.approx(rewards[5], abs=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_group_transformation_rank(self):
        """Test rank_groups transformation."""

        def simple_reward(completion, **kwargs):
            return float(completion)

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        # Create rollouts with varying scores
        prompts = ["p1", "p1", "p1", "p2", "p2", "p2"]
        completions = ["3.0", "1.0", "2.0", "30.0", "10.0", "20.0"]
        answers = [""] * 6
        tasks = ["test"] * 6
        infos = [{}] * 6
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 6

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform="rank",
        )

        rewards = results.reward

        # Ranks should be 0, 1, 2 for each group (0=lowest, 2=highest)
        # Group 1: [3.0, 1.0, 2.0] -> ranks [2, 0, 1]
        assert rewards[0] == 2.0  # 3.0 is highest
        assert rewards[1] == 0.0  # 1.0 is lowest
        assert rewards[2] == 1.0  # 2.0 is middle

        # Group 2: [30.0, 10.0, 20.0] -> ranks [2, 0, 1]
        assert rewards[3] == 2.0  # 30.0 is highest
        assert rewards[4] == 0.0  # 10.0 is lowest
        assert rewards[5] == 1.0  # 20.0 is middle

    @pytest.mark.asyncio
    async def test_group_transformation_custom_function(self):
        """Test custom group transformation function."""

        def simple_reward(completion, **kwargs):
            return float(completion)

        def custom_transform(rewards: list[float]) -> list[float]:
            """Custom transform: subtract min from each value in group."""
            min_val = min(rewards)
            return [r - min_val for r in rewards]

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        prompts = ["p1", "p1", "p1"]
        completions = ["5.0", "7.0", "9.0"]
        answers = [""] * 3
        tasks = ["test"] * 3
        infos = [{}] * 3
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 3

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform=custom_transform,
        )

        rewards = results.reward

        # Should subtract 5.0 from each: [0.0, 2.0, 4.0]
        assert rewards[0] == 0.0
        assert rewards[1] == 2.0
        assert rewards[2] == 4.0

    @pytest.mark.asyncio
    async def test_score_rollouts_without_grouping_unchanged(self):
        """Test that score_rollouts without group_size behaves as before."""

        def simple_reward(completion, **kwargs):
            return float(completion)

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        prompts = ["p1", "p1", "p1"]
        completions = ["1.0", "2.0", "3.0"]
        answers = [""] * 3
        tasks = ["test"] * 3
        infos = [{}] * 3
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 3

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
        )

        # Should return original rewards without any transformation
        assert results.reward == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_group_size_validation(self):
        """Test that group_size must divide total rollouts evenly."""

        def simple_reward(completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[simple_reward], weights=[1.0])

        prompts = ["p1", "p1", "p1", "p2", "p2"]  # 5 rollouts
        completions = ["a", "b", "c", "d", "e"]
        answers = [""] * 5
        tasks = ["test"] * 5
        infos = [{}] * 5
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 5

        # group_size=2 should fail because 5 is not divisible by 2
        with pytest.raises(
            ValueError, match="Number of rollouts .* must be divisible by group_size"
        ):
            await rubric.score_rollouts(
                prompts=prompts,  # type: ignore
                completions=completions,  # type: ignore
                answers=answers,
                states=states,
                tasks=tasks,
                infos=infos,
                group_size=2,
            )

    @pytest.mark.asyncio
    async def test_group_transformation_with_metrics(self):
        """Test that group transformations apply to both reward and metrics."""

        def reward1(completion, **kwargs):
            return float(completion)

        def reward2(completion, **kwargs):
            return float(completion) * 2

        rubric = Rubric(funcs=[reward1, reward2], weights=[1.0, 0.5])

        prompts = ["p1", "p1", "p1"]
        completions = ["1.0", "2.0", "3.0"]
        answers = [""] * 3
        tasks = ["test"] * 3
        infos = [{}] * 3
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 3

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform="normalize",
        )

        # Both metrics should be normalized
        assert pytest.approx(results.metrics["reward1"][0], abs=1e-6) == 0.0
        assert pytest.approx(results.metrics["reward1"][1], abs=1e-6) == 0.5
        assert pytest.approx(results.metrics["reward1"][2], abs=1e-6) == 1.0

        assert pytest.approx(results.metrics["reward2"][0], abs=1e-6) == 0.0
        assert pytest.approx(results.metrics["reward2"][1], abs=1e-6) == 0.5
        assert pytest.approx(results.metrics["reward2"][2], abs=1e-6) == 1.0

    @pytest.mark.asyncio
    async def test_group_transformation_constant_rewards(self):
        """Test standardize handles constant rewards in a group gracefully."""

        def constant_reward(completion, **kwargs):
            return 5.0  # All rewards are the same

        rubric = Rubric(funcs=[constant_reward], weights=[1.0])

        prompts = ["p1", "p1", "p1"]
        completions = ["a", "b", "c"]
        answers = [""] * 3
        tasks = ["test"] * 3
        infos = [{}] * 3
        states = [
            {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}
        ] * 3

        results = await rubric.score_rollouts(
            prompts=prompts,  # type: ignore
            completions=completions,  # type: ignore
            answers=answers,
            states=states,
            tasks=tasks,
            infos=infos,
            group_size=3,
            group_transform="standardize",
        )

        # When std=0, standardized values should be 0
        assert all(r == 0.0 for r in results.reward)
