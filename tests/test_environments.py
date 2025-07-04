"""
Comprehensive test suite for Environment classes.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from datasets import Dataset

from verifiers.envs.environment import Environment
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import Parser, XMLParser
from verifiers.rubrics import Rubric
from tests.mock_openai_client import MockOpenAIClient, create_mock_math_client


class ConcreteMultiTurnEnv(MultiTurnEnv):
    """Concrete implementation of MultiTurnEnv for testing."""
    
    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs) -> bool:
        # Complete if we have an answer or reached max turns
        return any("final answer" in msg.get("content", "").lower() for msg in messages)
    
    def env_response(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs) -> tuple:
        # Simple echo response
        last_msg = messages[-1].get("content", "")
        if "calculate" in last_msg.lower():
            response = {"role": "user", "content": "Calculation result: 42"}
        else:
            response = {"role": "user", "content": "Environment response"}
        return response, state


class TestEnvironment:
    """Test the base Environment class."""
    
    def test_init_minimal(self):
        # Create a minimal dataset for testing
        dataset = Dataset.from_dict({
            "question": ["What is 2+2?"],
            "answer": ["4"]
        })
        
        with pytest.raises(TypeError):
            # Environment is abstract, can't instantiate directly
            Environment(dataset=dataset)
    
    def test_format_prompt(self):
        # Use SingleTurnEnv as concrete implementation
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient()
        )
        
        prompt = "What is 2+2?"
        system_prompt = "You are a math tutor"
        few_shot = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"}
        ]
        
        result = env.format_prompt(prompt, system_prompt, few_shot)
        
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_prompt
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is 1+1?"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "2"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == prompt
    
    def test_format_prompt_minimal(self):
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient()
        )
        
        prompt = "Test prompt"
        result = env.format_prompt(prompt)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == prompt
    
    def test_format_dataset(self):
        dataset = Dataset.from_dict({
            "question": ["What is 2+2?", "What is 3+3?"],
            "answer": ["4", "6"]
        })
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=MockOpenAIClient(),
            system_prompt="You are helpful"
        )
        
        # Dataset should be automatically formatted during init
        formatted = env.dataset
        assert "prompt" in formatted.column_names
        assert len(formatted) == 2
        
        # Check first prompt format
        first_prompt = formatted[0]["prompt"]
        assert isinstance(first_prompt, list)
        assert first_prompt[0]["role"] == "system"
        assert first_prompt[1]["role"] == "user"
        assert first_prompt[1]["content"] == "What is 2+2?"
    
    def test_format_dataset_custom_keys(self):
        dataset = Dataset.from_dict({
            "input": ["Question 1"],
            "output": ["Answer 1"]
        })
        
        # Create environment without dataset to avoid auto-formatting
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["dummy"], "answer": ["dummy"]}),
            client=MockOpenAIClient()
        )
        
        # Test manual formatting with custom keys
        formatted = env.format_dataset(dataset, question_key="input", answer_key="output")
        assert "prompt" in formatted.column_names
        assert "answer" in formatted.column_names
    
    def test_get_dataset(self):
        dataset = Dataset.from_dict({
            "question": [f"Question {i}" for i in range(10)],
            "answer": [f"Answer {i}" for i in range(10)]
        })
        
        env = SingleTurnEnv(dataset=dataset, client=MockOpenAIClient())
        
        # Get all data
        all_data = env.get_dataset()
        assert len(all_data) == 10
        
        # Get limited data
        limited_data = env.get_dataset(n=5, seed=42)
        assert len(limited_data) == 5
    
    def test_get_eval_dataset(self):
        train_dataset = Dataset.from_dict({
            "question": ["Train Q"], "answer": ["Train A"]
        })
        eval_dataset = Dataset.from_dict({
            "question": ["Eval Q"], "answer": ["Eval A"]
        })
        
        env = SingleTurnEnv(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            client=MockOpenAIClient()
        )
        
        eval_data = env.get_eval_dataset()
        assert len(eval_data) == 1
        assert eval_data[0]["prompt"][0]["content"] == "Eval Q"
    
    def test_sanitize_sampling_args_localhost(self):
        client = MockOpenAIClient(base_url="http://localhost")  # No port to match netloc check
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client
        )
        
        sampling_args = {"temperature": 0.7, "extra_body": {"custom": "value"}}
        result = env.sanitize_sampling_args(client, sampling_args)
        
        # Should preserve extra_body for localhost
        assert "extra_body" in result
        assert result["temperature"] == 0.7
    
    def test_sanitize_sampling_args_remote(self):
        client = MockOpenAIClient(base_url="https://api.openai.com")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client
        )
        
        sampling_args = {"temperature": 0.7, "extra_body": {"custom": "value"}}
        result = env.sanitize_sampling_args(client, sampling_args)
        
        # Should remove extra_body for remote URLs
        assert "extra_body" not in result
        assert result["temperature"] == 0.7
    
    def test_get_model_response_chat(self):
        client = MockOpenAIClient(default_chat_response="Test response")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client
        )
        
        prompt = [{"role": "user", "content": "Test question"}]
        response = env.get_model_response(prompt, client, "test-model")
        
        assert response == "Test response"
        assert client.chat.completions.last_model == "test-model"
    
    def test_get_model_response_completion(self):
        client = MockOpenAIClient(default_completion_response="Completion response")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            message_type="completion"
        )
        
        response = env.get_model_response("Test prompt", client, "test-model", message_type="completion")
        
        assert response == "Completion response"
        assert client.completions.last_model == "test-model"
    
    def test_get_model_response_max_tokens_error(self):
        client = MockOpenAIClient()
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client
        )
        
        prompt = [{"role": "user", "content": "Test"}]
        response = env.get_model_response(prompt, client, "test-model", {"max_tokens": 1})
        
        assert response == "[ERROR] max_tokens_reached"
    
    def test_get_model_response_exception_handling(self):
        from tests.mock_openai_client import create_mock_error_client
        
        client = create_mock_error_client()
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client
        )
        
        prompt = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception, match="Generic error"):
            env.get_model_response(prompt, client, "test-model")
    
    def test_init_completion_mode_with_system_prompt_error(self):
        dataset = Dataset.from_dict({"prompt": ["Test"], "answer": ["Test"]})
        
        with pytest.raises(ValueError, match="not supported for completion tasks"):
            SingleTurnEnv(
                dataset=dataset,
                client=MockOpenAIClient(),
                message_type="completion",
                system_prompt="System prompt"
            )
    
    def test_init_no_dataset_error(self):
        with pytest.raises(ValueError, match="Either dataset or eval_dataset must be provided"):
            SingleTurnEnv(client=MockOpenAIClient())
    
    def test_process_chat_format(self):
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient()
        )
        
        # Mock tokenizer that returns different lengths for different inputs
        tokenizer = Mock()
        def mock_apply_chat_template(msgs, **kwargs):
            if len(msgs) == 1:  # Just prompt
                return "prompt_text"
            elif len(msgs) == 2:  # Prompt + completion
                return "prompt_text completion_text"
            else:
                return "formatted_text"
        
        def mock_encode(text):
            if text == "prompt_text":
                return [1, 2]
            elif text == "prompt_text completion_text":
                return [1, 2, 3, 4]  # Includes new tokens [3, 4]
            else:
                return [1, 2, 3, 4, 5, 6]
        
        tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
        tokenizer.encode.side_effect = mock_encode
        
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        
        prompt_ids, prompt_mask, completion_ids, completion_mask = env.process_chat_format(
            prompt, completion, tokenizer
        )
        
        assert prompt_ids == [1, 2]
        assert prompt_mask == [1, 1]
        assert completion_ids == [3, 4]
        assert completion_mask == [1, 1]
    
    def test_process_completion_format(self):
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient()
        )
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: [1, 2] if text == "prompt" else [3, 4]
        
        prompt_ids, prompt_mask, completion_ids, completion_mask = env.process_completion_format(
            "prompt", "completion", tokenizer
        )
        
        assert prompt_ids == [1, 2]
        assert prompt_mask == [1, 1]
        assert completion_ids == [3, 4]
        assert completion_mask == [1, 1]


class TestSingleTurnEnv:
    """Test the SingleTurnEnv class."""
    
    def test_init_chat_mode(self):
        dataset = Dataset.from_dict({
            "question": ["What is 2+2?"],
            "answer": ["4"]
        })
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=MockOpenAIClient(),
            message_type="chat"
        )
        
        assert env.message_type == "chat"
        assert env.dataset is not None
    
    def test_init_completion_mode(self):
        dataset = Dataset.from_dict({
            "prompt": ["What is 2+2?"],
            "answer": ["4"]
        })
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=MockOpenAIClient(),
            message_type="completion"
        )
        
        assert env.message_type == "completion"
    
    def test_rollout_chat_mode(self):
        client = MockOpenAIClient(default_chat_response="4")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            message_type="chat"
        )
        
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        completion, state = env.rollout(client, "test-model", prompt, "4")
        
        assert isinstance(completion, list)
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "4"
        assert state == {}
    
    def test_rollout_completion_mode(self):
        client = MockOpenAIClient(default_completion_response="4")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            client=client,
            message_type="completion"
        )
        
        completion, state = env.rollout(client, "test-model", "What is 2+2?", "4")
        
        assert isinstance(completion, str)
        assert completion == "4"
        assert state == {}


class TestMultiTurnEnv:
    """Test the MultiTurnEnv class."""
    
    def test_init(self):
        env = ConcreteMultiTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient(),
            max_turns=5
        )
        
        assert env.max_turns == 5
    
    def test_rollout_early_completion(self):
        client = MockOpenAIClient(default_chat_response="The final answer is 42")
        env = ConcreteMultiTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            max_turns=10
        )
        
        prompt = [{"role": "user", "content": "What is the answer?"}]
        completion, state = env.rollout(client, "test-model", prompt, "42")
        
        # Should complete early because response contains "final answer"
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"
        assert "final answer" in completion[0]["content"]
        assert state["answer"] == "42"
    
    def test_rollout_max_turns(self):
        client = MockOpenAIClient(default_chat_response="Still thinking...")
        env = ConcreteMultiTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            max_turns=2
        )
        
        prompt = [{"role": "user", "content": "What is the answer?"}]
        completion, state = env.rollout(client, "test-model", prompt, "42")
        
        # Should stop at max_turns
        # Each turn generates: assistant response + env response (except last)
        assert len(completion) == 3  # assistant + env + assistant (stopped at max_turns)
    
    def test_rollout_with_error(self):
        client = MockOpenAIClient(default_chat_response="[ERROR] max_tokens_reached")
        env = ConcreteMultiTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            max_turns=10
        )
        
        prompt = [{"role": "user", "content": "What is the answer?"}]
        completion, state = env.rollout(client, "test-model", prompt, "42")
        
        # Should stop immediately on error
        assert len(completion) == 1
        assert "[ERROR]" in completion[0]["content"]
    
    def test_rollout_with_env_interaction(self):
        # Mock client that responds differently based on content
        responses = {
            "calculate": "I need to calculate 2+2",
            "Environment response": "The final answer is 4"
        }
        client = MockOpenAIClient(chat_responses=responses, default_chat_response="The final answer is 4")
        
        env = ConcreteMultiTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            max_turns=10
        )
        
        prompt = [{"role": "user", "content": "Please calculate 2+2"}]
        completion, state = env.rollout(client, "test-model", prompt, "4")
        
        # Should have: assistant response + env response + assistant response
        assert len(completion) >= 2
        assert any("calculate" in msg.get("content", "") for msg in completion)


class TestEnvironmentIntegration:
    """Integration tests for Environment classes."""
    
    def test_generate_method(self):
        client = create_mock_math_client()
        dataset = Dataset.from_dict({
            "question": ["What is 2+2?"],
            "answer": ["4"]
        })
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=client,
            model="test-model",
            parser=XMLParser(["think", "answer"]),
            rubric=Rubric()
        )
        
        inputs = {"prompt": env.dataset["prompt"], "answer": env.dataset["answer"]}
        results = env.generate(inputs)
        
        assert "completion" in results
        assert "state" in results
        assert "reward" in results
        assert len(results["completion"]) == 1
    
    def test_evaluate_method(self):
        client = create_mock_math_client()
        train_dataset = Dataset.from_dict({
            "question": ["What is 2+2?"],
            "answer": ["4"]
        })
        eval_dataset = Dataset.from_dict({
            "question": ["What is 5*3?"],
            "answer": ["15"]
        })
        
        env = SingleTurnEnv(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            client=client,
            model="test-model"
        )
        
        results = env.evaluate(num_samples=1)
        
        assert "completion" in results
        assert "reward" in results
        assert len(results["completion"]) == 1
    
    def test_make_dataset_method(self):
        client = create_mock_math_client()
        dataset = Dataset.from_dict({
            "question": ["What is 2+2?"],
            "answer": ["4"]
        })
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=client,
            model="test-model"
        )
        
        # Pass client and model explicitly since make_dataset needs them
        new_dataset = env.make_dataset(client=client, model="test-model", num_samples=1)
        
        assert isinstance(new_dataset, Dataset)
        assert "prompt" in new_dataset.column_names
        assert "completion" in new_dataset.column_names
        assert "answer" in new_dataset.column_names
        assert "reward" in new_dataset.column_names
    
    def test_environment_with_custom_rubric(self):
        def custom_reward(completion, answer, **kwargs):
            return 0.5 if "test" in str(completion) else 0.0
        
        custom_reward.__name__ = "custom_reward"
        rubric = Rubric(funcs=[custom_reward])
        
        client = MockOpenAIClient(default_chat_response="test response")
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            model="test-model",
            rubric=rubric
        )
        
        inputs = {"prompt": env.dataset["prompt"], "answer": env.dataset["answer"]}
        results = env.generate(inputs)
        
        assert results["custom_reward"][0] == 0.5
    
    def test_async_functionality(self):
        """Test that async operations work correctly."""
        client = MockOpenAIClient()
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=client,
            model="test-model"
        )
        
        # Test multiple rollouts with concurrency
        prompts = [env.dataset["prompt"][0]] * 3
        answers = [env.dataset["answer"][0]] * 3
        
        rollouts = env.run_rollouts(client, "test-model", prompts, answers, max_concurrent=2)
        
        assert len(rollouts) == 3
        assert all(isinstance(rollout, tuple) for rollout in rollouts)
        assert all(len(rollout) == 2 for rollout in rollouts)  # (completion, state)


class TestEnvironmentEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset_handling(self):
        dataset = Dataset.from_dict({"question": [], "answer": []})
        
        env = SingleTurnEnv(
            dataset=dataset,
            client=MockOpenAIClient()
        )
        
        # Should handle empty dataset gracefully
        assert len(env.dataset) == 0
    
    def test_missing_columns_in_dataset(self):
        # Dataset missing required columns
        dataset = Dataset.from_dict({"wrong_column": ["test"]})
        
        with pytest.raises(KeyError):
            env = SingleTurnEnv(
                dataset=dataset,
                client=MockOpenAIClient()
            )
            # This should fail when trying to access "question" column
            env.format_dataset(dataset)
    
    def test_process_env_results_chat_format(self):
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient()
        )
        
        # Mock tokenizer with proper chat format behavior
        tokenizer = Mock()
        call_count = 0
        def mock_apply_chat_template(msgs, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call for prompt only
                return "prompt_formatted"
            else:  # Second call for prompt + completion
                return "prompt_formatted completion_formatted"
        
        def mock_encode(text):
            if text == "prompt_formatted":
                return [1, 2]
            elif text == "prompt_formatted completion_formatted":
                return [1, 2, 3, 4]  # Prompt + completion tokens
            else:
                return [1, 2, 3]
        
        tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
        tokenizer.encode.side_effect = mock_encode
        
        prompts = [[{"role": "user", "content": "test"}]]
        completions = [[{"role": "assistant", "content": "response"}]]
        states = [{}]
        rewards = [1.0]
        
        result = env.process_env_results(
            prompts, completions, states, rewards, tokenizer
        )
        
        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert "rewards" in result
        assert len(result["prompt_ids"]) == 1
    
    def test_process_env_results_completion_format(self):
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient(),
            message_type="completion"
        )
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: [1, 2] if "prompt" in text else [3, 4]
        
        prompts = ["test prompt"]
        completions = ["test completion"]
        states = [{}]
        rewards = [1.0]
        
        result = env.process_env_results(
            prompts, completions, states, rewards, tokenizer
        )
        
        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert len(result["prompt_ids"]) == 1
    
    def test_multiturn_env_abstract_methods(self):
        """Test that MultiTurnEnv enforces abstract method implementation."""
        
        class IncompleteMultiTurnEnv(MultiTurnEnv):
            # Missing is_completed and env_response implementations
            pass
        
        with pytest.raises(TypeError):
            IncompleteMultiTurnEnv(
                dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
                client=MockOpenAIClient()
            )
    
    def test_environment_kwargs_handling(self):
        """Test that environments handle arbitrary kwargs correctly."""
        env = SingleTurnEnv(
            dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            client=MockOpenAIClient(),
            custom_param="test_value",
            another_param=42
        )
        
        assert env.custom_param == "test_value"
        assert env.another_param == 42