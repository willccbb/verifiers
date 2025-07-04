"""
Comprehensive test suite for Rubric classes.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from verifiers.rubrics.rubric import Rubric
from verifiers.rubrics.tool_rubric import ToolRubric
from verifiers.rubrics.rubric_group import RubricGroup
from verifiers.parsers import Parser, XMLParser


class TestRubric:
    """Test the base Rubric class."""
    
    def test_init_default(self):
        rubric = Rubric()
        assert rubric.parser is not None
        assert isinstance(rubric.parser, Parser)
        assert rubric.reward_funcs == []
        assert rubric.reward_weights == []
        assert hasattr(rubric, 'logger')
    
    def test_init_with_params(self):
        parser = XMLParser(["reasoning", "answer"])
        
        def test_func(completion, **kwargs):
            return 0.5
        
        rubric = Rubric(
            funcs=[test_func],
            weights=[0.8],
            parser=parser,
            custom_attr="test"
        )
        
        assert rubric.parser == parser
        assert len(rubric.reward_funcs) == 1
        assert rubric.reward_weights == [0.8]
        assert rubric.custom_attr == "test"
    
    def test_init_auto_weights(self):
        def test_func1(completion, **kwargs):
            return 0.5
        
        def test_func2(completion, **kwargs):
            return 0.7
        
        rubric = Rubric(funcs=[test_func1, test_func2])
        assert rubric.reward_weights == [1.0, 1.0]
    
    def test_get_reward_func_names(self):
        def test_func1(completion, **kwargs):
            return 0.5
        
        def test_func2(completion, **kwargs):
            return 0.7
        
        test_func1.__name__ = "func1"
        test_func2.__name__ = "func2"
        
        rubric = Rubric(funcs=[test_func1, test_func2])
        names = rubric.get_reward_func_names()
        assert names == ["func1", "func2"]
    
    def test_get_reward_funcs(self):
        def test_func(completion, **kwargs):
            return 0.5
        
        rubric = Rubric(funcs=[test_func])
        funcs = rubric.get_reward_funcs()
        assert len(funcs) == 1
        assert funcs[0] == test_func
    
    def test_get_reward_weights(self):
        def test_func(completion, **kwargs):
            return 0.5
        
        rubric = Rubric(funcs=[test_func], weights=[0.8])
        weights = rubric.get_reward_weights()
        assert weights == [0.8]
    
    def test_add_reward_func(self):
        rubric = Rubric()
        
        def new_func(completion, **kwargs):
            return 0.6
        
        rubric.add_reward_func(new_func, weight=0.9)
        assert len(rubric.reward_funcs) == 1
        assert rubric.reward_weights == [0.9]
    
    def test_add_reward_func_default_weight(self):
        rubric = Rubric()
        initial_weights = len(rubric.reward_weights)
        
        def new_func(completion, **kwargs):
            return 0.6
        
        rubric.add_reward_func(new_func)
        assert len(rubric.reward_weights) == initial_weights + 1
        assert rubric.reward_weights[-1] == 1.0
    
    def test_call_reward_func_with_kwargs(self):
        rubric = Rubric()
        
        def test_func(completion, answer, **kwargs):
            return float(len(completion) + len(answer))
        
        result = rubric._call_reward_func(
            test_func,
            prompt="test prompt",
            completion="test completion", 
            answer="test answer",
            state={},
            task="test"
        )
        assert result == 26.0  # len("test completion") + len("test answer")
    
    def test_call_reward_func_selective_params(self):
        rubric = Rubric()
        
        def test_func(completion, answer):
            return float(len(completion))
        
        result = rubric._call_reward_func(
            test_func,
            prompt="test prompt",
            completion="test",
            answer="test answer",
            state={},
            task="test",
            extra_param="ignored"
        )
        assert result == 4.0
    
    def test_call_reward_func_error_handling(self):
        rubric = Rubric()
        
        def error_func(completion, **kwargs):
            raise ValueError("Test error")
        
        result = rubric._call_reward_func(
            error_func,
            prompt="test",
            completion="test",
            answer="test",
            state={},
            task="test"
        )
        assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_score_rollout(self):
        def test_func1(completion, **kwargs):
            return 0.5
        
        def test_func2(completion, **kwargs):
            return 0.8
        
        test_func1.__name__ = "func1"
        test_func2.__name__ = "func2"
        
        rubric = Rubric(funcs=[test_func1, test_func2], weights=[1.0, 2.0])
        
        result = await rubric.score_rollout(
            prompt="test prompt",
            completion="test completion",
            answer="test answer",
            state={}
        )
        
        assert "func1" in result
        assert "func2" in result
        assert "reward" in result
        assert result["func1"] == 0.5
        assert result["func2"] == 0.8
        assert result["reward"] == 0.5 * 1.0 + 0.8 * 2.0
    
    def test_score_rollouts_sync(self):
        def test_func(completion, **kwargs):
            return 0.5
        
        test_func.__name__ = "test_func"
        rubric = Rubric(funcs=[test_func])
        
        prompts = ["prompt1", "prompt2"]
        completions = ["completion1", "completion2"]
        answers = ["answer1", "answer2"]
        states = [{}, {}]
        tasks = ["task1", "task2"]
        
        result = rubric.score_rollouts(
            prompts, completions, answers, states, tasks
        )
        
        assert "test_func" in result
        assert "reward" in result
        assert len(result["test_func"]) == 2
        assert all(score == 0.5 for score in result["test_func"])



class TestToolRubric:
    """Test the ToolRubric class."""
    
    def test_init_default(self):
        rubric = ToolRubric()
        assert isinstance(rubric.parser, XMLParser)
        assert isinstance(rubric.env_parser, XMLParser)
        assert len(rubric.tools) == 0
        assert len(rubric.reward_funcs) == 3  # correct_answer, tool_execution, format
    
    def test_init_with_tools(self):
        def calculator_tool():
            pass
        
        def search_tool():
            pass
        
        calculator_tool.__name__ = "calculator"
        search_tool.__name__ = "search"
        
        rubric = ToolRubric(tools=[calculator_tool, search_tool])
        assert "calculator" in rubric.tools
        assert "search" in rubric.tools
        assert len(rubric.reward_funcs) == 9  # 3 base + 3 per tool * 2 tools
    
    def test_evaluate_code_success(self):
        rubric = ToolRubric()
        code = "print('Hello')"
        answer = '{"test_cases": [{"input": "", "output": "Hello"}]}'
        
        with patch('sys.stdin'), patch('builtins.exec') as mock_exec:
            mock_exec.return_value = None
            with patch('sys.stdout.write'):
                reward = rubric.evaluate_code(code, answer)
                # This is a simplified test - actual implementation may vary
                assert isinstance(reward, float)
    
    def test_evaluate_code_invalid_answer(self):
        rubric = ToolRubric()
        code = "print('Hello')"
        answer = "invalid json"
        
        reward = rubric.evaluate_code(code, answer)
        assert reward == 0.0
    
    def test_evaluate_code_strips_markdown(self):
        rubric = ToolRubric()
        code = "```python\nprint('Hello')\n```"
        answer = '{"test_cases": [{"input": "", "output": "Hello"}]}'
        
        with patch('sys.stdin'), patch('builtins.exec') as mock_exec:
            mock_exec.return_value = None
            with patch('sys.stdout.write'):
                reward = rubric.evaluate_code(code, answer)
                assert isinstance(reward, float)
    
    def test_correct_answer_reward_func_mc(self):
        rubric = ToolRubric()
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>A</answer>"}]
        answer = "A"
        task = "mc"
        
        reward = rubric.correct_answer_reward_func(completion, answer, task)
        assert reward == 1.0
    
    def test_correct_answer_reward_func_mc_wrong(self):
        rubric = ToolRubric()
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>B</answer>"}]
        answer = "A"
        task = "mc"
        
        reward = rubric.correct_answer_reward_func(completion, answer, task)
        assert reward == 0.0
    
    def test_correct_answer_reward_func_math(self):
        rubric = ToolRubric()
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>42</answer>"}]
        answer = "42"
        task = "math"
        
        reward = rubric.correct_answer_reward_func(completion, answer, task)
        assert reward == 1.0
    
    def test_correct_answer_reward_func_code(self):
        rubric = ToolRubric()
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>print('test')</answer>"}]
        answer = '{"test_cases": [{"input": "", "output": "test"}]}'
        task = "code"
        
        with patch.object(rubric, 'evaluate_code', return_value=0.8):
            reward = rubric.correct_answer_reward_func(completion, answer, task)
            assert reward == 0.8
    
    def test_correct_answer_reward_func_unknown_task(self):
        rubric = ToolRubric()
        completion = [{"role": "assistant", "content": "<answer>test</answer>"}]
        answer = "test"
        task = "unknown"
        
        reward = rubric.correct_answer_reward_func(completion, answer, task)
        assert reward == 0.0
    
    def test_tool_execution_reward_func_success(self):
        rubric = ToolRubric()
        completion = [
            {"role": "assistant", "content": '<reasoning>Calculate</reasoning>\n<tool>{"name": "calc"}</tool>'},
            {"role": "user", "content": "<result>4</result>"}
        ]
        
        reward = rubric.tool_execution_reward_func(completion)
        assert reward == 1.0
    
    def test_tool_execution_reward_func_error(self):
        rubric = ToolRubric()
        completion = [
            {"role": "assistant", "content": '<reasoning>Calculate</reasoning>\n<tool>{"name": "calc"}</tool>'},
            {"role": "user", "content": "<result>Error: invalid input</result>"}
        ]
        
        reward = rubric.tool_execution_reward_func(completion)
        assert reward == 0.0
    
    def test_tool_execution_reward_func_no_tools(self):
        rubric = ToolRubric()
        completion = [
            {"role": "assistant", "content": "<reasoning>Just thinking</reasoning>"}
        ]
        
        reward = rubric.tool_execution_reward_func(completion)
        assert reward == 0.0
    
    def test_tool_execution_reward_func_partial_success(self):
        rubric = ToolRubric()
        completion = [
            {"role": "assistant", "content": '<tool>{"name": "calc1"}</tool>'},
            {"role": "user", "content": "<result>4</result>"},
            {"role": "assistant", "content": '<tool>{"name": "calc2"}</tool>'},
            {"role": "user", "content": "<result>Error: failed</result>"}
        ]
        
        reward = rubric.tool_execution_reward_func(completion)
        assert reward == 0.5
    
    def test_get_named_tool_reward_func(self):
        def calculator_tool():
            pass
        
        calculator_tool.__name__ = "calculator"
        rubric = ToolRubric(tools=[calculator_tool])
        
        reward_func = rubric.get_named_tool_reward_func("calculator")
        assert callable(reward_func)
        assert reward_func.__name__ == "calculator_reward_func"
        
        # Test the function
        completion = [
            {"role": "assistant", "content": '<tool>{"name": "calculator", "args": {}}</tool>'},
            {"role": "user", "content": "<result>42</result>"}
        ]
        reward = reward_func(completion)
        assert reward == 1.0
    
    def test_get_named_tool_count_reward_func(self):
        def calculator_tool():
            pass
        
        calculator_tool.__name__ = "calculator"
        rubric = ToolRubric(tools=[calculator_tool])
        
        count_func = rubric.get_named_tool_count_reward_func("calculator")
        assert callable(count_func)
        assert count_func.__name__ == "calculator_count_reward_func"
        
        # Test the function
        completion = [
            {"role": "assistant", "content": '<tool>{"name": "calculator", "args": {}}</tool>'},
            {"role": "user", "content": "<result>42</result>"},
            {"role": "assistant", "content": '<tool>{"name": "calculator", "args": {}}</tool>'},
            {"role": "user", "content": "<result>24</result>"}
        ]
        count = count_func(completion)
        assert count == 2.0
    
    def test_get_named_tool_attempt_reward_func(self):
        def calculator_tool():
            pass
        
        calculator_tool.__name__ = "calculator"
        rubric = ToolRubric(tools=[calculator_tool])
        
        attempt_func = rubric.get_named_tool_attempt_reward_func("calculator")
        assert callable(attempt_func)
        assert attempt_func.__name__ == "calculator_attempt_reward_func"
        
        # Test the function with successful and failed attempts
        completion = [
            {"role": "assistant", "content": '<tool>{"name": "calculator", "args": {}}</tool>'},
            {"role": "user", "content": "<result>Error: failed</result>"},
            {"role": "assistant", "content": '<tool>{"name": "calculator", "args": {}}</tool>'}
        ]
        attempts = attempt_func(completion)
        assert attempts == 2.0


class TestRubricGroup:
    """Test the RubricGroup class."""
    
    def test_init(self):
        rubric1 = Rubric()
        rubric2 = Rubric()
        group = RubricGroup([rubric1, rubric2])
        
        assert len(group.rubrics) == 2
        assert group.rubrics[0] == rubric1
        assert group.rubrics[1] == rubric2
    
    def test_init_empty_error(self):
        with pytest.raises(AssertionError, match="RubricGroup must have at least one rubric"):
            RubricGroup([])
    
    def test_get_reward_func_names(self):
        def func1(completion, **kwargs):
            return 0.5
        
        def func2(completion, **kwargs):
            return 0.7
        
        func1.__name__ = "func1"
        func2.__name__ = "func2"
        
        rubric1 = Rubric(funcs=[func1])
        rubric2 = Rubric(funcs=[func2])
        group = RubricGroup([rubric1, rubric2])
        
        names = group.get_reward_func_names()
        assert "func1" in names
        assert "func2" in names
    
    def test_get_reward_funcs(self):
        def func1(completion, **kwargs):
            return 0.5
        
        def func2(completion, **kwargs):
            return 0.7
        
        rubric1 = Rubric(funcs=[func1])
        rubric2 = Rubric(funcs=[func2])
        group = RubricGroup([rubric1, rubric2])
        
        funcs = group.get_reward_funcs()
        assert func1 in funcs
        assert func2 in funcs
    
    def test_get_reward_weights(self):
        def func1(completion, **kwargs):
            return 0.5
        
        def func2(completion, **kwargs):
            return 0.7
        
        rubric1 = Rubric(funcs=[func1], weights=[0.8])
        rubric2 = Rubric(funcs=[func2], weights=[0.9])
        group = RubricGroup([rubric1, rubric2])
        
        weights = group.get_reward_weights()
        assert 0.8 in weights
        assert 0.9 in weights
    
    def test_add_reward_func(self):
        # Create fresh rubrics with no functions to avoid state issues
        rubric1 = Rubric(funcs=[], weights=[], parser=None)
        rubric2 = Rubric(funcs=[], weights=[], parser=None)
        group = RubricGroup([rubric1, rubric2])
        
        def new_func(completion, **kwargs):
            return 0.6
        
        group.add_reward_func(new_func, weight=0.5)
        
        # Should be added to the first rubric
        assert len(rubric1.reward_funcs) == 1
        assert len(rubric2.reward_funcs) == 0
    
    def test_score_rollouts(self):
        def func1(completion, **kwargs):
            return 0.5
        
        def func2(completion, **kwargs):
            return 0.7
        
        func1.__name__ = "func1"
        func2.__name__ = "shared_func"
        
        rubric1 = Rubric(funcs=[func1])
        rubric2 = Rubric(funcs=[func2])
        group = RubricGroup([rubric1, rubric2])
        
        prompts = ["prompt1"]
        completions = ["completion1"]
        answers = ["answer1"]
        states = [{}]
        tasks = ["task1"]
        
        result = group.score_rollouts(prompts, completions, answers, states, tasks)
        
        assert "func1" in result
        assert "shared_func" in result
        assert "reward" in result
        assert len(result["func1"]) == 1
        assert result["func1"][0] == 0.5
    
    def test_score_rollouts_same_name_aggregation(self):
        def func1(completion, **kwargs):
            return 0.3
        
        def func2(completion, **kwargs):
            return 0.7
        
        func1.__name__ = "shared_func"
        func2.__name__ = "shared_func"
        
        rubric1 = Rubric(funcs=[func1])
        rubric2 = Rubric(funcs=[func2])
        group = RubricGroup([rubric1, rubric2])
        
        prompts = ["prompt1"]
        completions = ["completion1"]
        answers = ["answer1"]
        states = [{}]
        tasks = ["task1"]
        
        result = group.score_rollouts(prompts, completions, answers, states, tasks)
        
        # Should sum the scores from functions with the same name
        assert result["shared_func"][0] == 1.0  # 0.3 + 0.7


class TestRubricIntegration:
    """Integration tests across different rubric types."""
    
    def test_rubric_inheritance(self):
        """Test that specialized rubrics inherit from base Rubric."""
        tool_rubric = ToolRubric()
        group = RubricGroup([tool_rubric])
        
        assert isinstance(tool_rubric, Rubric)
        assert isinstance(group, Rubric)
    
    def test_rubric_polymorphism(self):
        """Test that rubrics can be used polymorphically."""
        rubrics = [
            Rubric(),
            ToolRubric()
        ]
        
        for rubric in rubrics:
            funcs = rubric.get_reward_funcs()
            weights = rubric.get_reward_weights()
            names = rubric.get_reward_func_names()
            
            assert isinstance(funcs, list)
            assert isinstance(weights, list)
            assert isinstance(names, list)
    
    @pytest.mark.asyncio
    async def test_all_rubrics_score_rollout(self):
        """Test that all rubrics can score rollouts asynchronously."""
        rubrics = [
            Rubric(),
            ToolRubric()
        ]
        
        for rubric in rubrics:
            result = await rubric.score_rollout(
                prompt="test prompt",
                completion="test completion",
                answer="test answer",
                state={}
            )
            
            assert isinstance(result, dict)
            assert "reward" in result
            assert isinstance(result["reward"], (int, float))
    
    def test_all_rubrics_score_rollout_sync(self):
        """Test that all rubrics can score rollouts synchronously."""
        rubrics = [
            Rubric(),
            ToolRubric()
        ]
        
        for rubric in rubrics:
            result = rubric.score_rollouts(
                prompts=["test prompt"],
                completions=["test completion"],
                answers=["test answer"],
                states=[{}],
                tasks=["test"]
            )
            
            assert isinstance(result, dict)
            assert "reward" in result
            assert isinstance(result["reward"], list)


class TestRubricEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_rubric_with_no_functions(self):
        # Create a fresh rubric with no functions
        rubric = Rubric(funcs=[], weights=[], parser=None)
        
        prompts = ["test"]
        completions = ["test"]
        answers = ["test"]
        states = [{}]
        tasks = ["test"]
        
        result = rubric.score_rollouts(prompts, completions, answers, states, tasks)
        
        # Should still work with empty function list
        assert "reward" in result
        assert result["reward"] == [0.0]
    
    def test_tool_rubric_with_invalid_tool_json(self):
        rubric = ToolRubric()
        completion = [
            {"role": "assistant", "content": '<tool>invalid json</tool>'},
            {"role": "user", "content": "<result>success</result>"}
        ]
        
        reward = rubric.tool_execution_reward_func(completion)
        # XML parsing succeeds even with invalid JSON content, so tool execution reward is 1.0
        assert reward == 1.0
    
    def test_rubric_group_warning_on_add_func(self, caplog):
        # Create fresh rubrics with no functions
        rubric1 = Rubric(funcs=[], weights=[], parser=None)
        rubric2 = Rubric(funcs=[], weights=[], parser=None)
        group = RubricGroup([rubric1, rubric2])
        
        def new_func(completion, **kwargs):
            return 0.5
        
        group.add_reward_func(new_func)
        
        # Should log a warning - check that function was added to first rubric
        assert len(rubric1.reward_funcs) == 1  # Function was added
        assert len(rubric2.reward_funcs) == 0  # Second rubric unchanged