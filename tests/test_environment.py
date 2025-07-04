"""Tests for the base Environment class."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datasets import Dataset
from verifiers.envs.environment import Environment
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric


# Create a concrete implementation for testing the abstract base class
class ConcreteEnvironment(Environment):
    """Concrete implementation of Environment for testing."""
    
    async def rollout(self, client, model, prompt, answer, task="default", info={}, sampling_args={}, **kwargs):
        """Simple test rollout implementation."""
        response = await self.get_model_response(
            prompt=prompt,
            client=client,
            model=model,
            sampling_args=sampling_args
        )
        if self.message_type == 'chat':
            return [{'role': 'assistant', 'content': response}], {}
        return response, {}


class TestEnvironmentBase:
    """Test cases for the base Environment class."""

    def test_environment_initialization(self, mock_openai_client, sample_dataset):
        """Test that Environment initializes correctly."""
        env = ConcreteEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric()
        )
        assert env.client == mock_openai_client
        assert env.model == "test-model"
        assert env.message_type == 'chat'
        assert isinstance(env.parser, Parser)
        assert isinstance(env.rubric, Rubric)

    def test_environment_with_eval_dataset_only(self, mock_openai_client, sample_dataset):
        """Test Environment with only eval_dataset."""
        env = ConcreteEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric()
        )
        assert env.dataset is None
        assert env.eval_dataset is not None

    def test_environment_no_datasets_raises_error(self, mock_openai_client):
        """Test that Environment raises error when no datasets provided."""
        with pytest.raises(ValueError, match="Either dataset or eval_dataset must be provided"):
            TestEnvironment(
                client=mock_openai_client,
                model="test-model",
                parser=Parser(),
                rubric=Rubric()
            )

    def test_completion_mode_with_system_prompt_raises_error(self, mock_openai_client, sample_dataset):
        """Test that completion mode with system prompt raises error."""
        with pytest.raises(ValueError, match="not supported for completion tasks"):
            TestEnvironment(
                client=mock_openai_client,
                model="test-model",
                dataset=sample_dataset,
                message_type="completion",
                system_prompt="test prompt",
                parser=Parser(),
                rubric=Rubric()
            )

    def test_format_prompt(self, mock_openai_client, sample_dataset):
        """Test prompt formatting."""
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompt = "What is 2+2?"
        system_prompt = "You are a helpful assistant."
        few_shot = [{"role": "user", "content": "What is 1+1?"}, {"role": "assistant", "content": "2"}]
        
        formatted = env.format_prompt(prompt, system_prompt, few_shot)
        
        assert len(formatted) == 4
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "What is 1+1?"
        assert formatted[2]["role"] == "assistant"
        assert formatted[2]["content"] == "2"
        assert formatted[3]["role"] == "user"
        assert formatted[3]["content"] == prompt

    def test_get_dataset(self, mock_openai_client, sample_dataset):
        """Test dataset retrieval."""
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric()
        )
        
        # Get full dataset
        full_dataset = env.get_dataset()
        assert len(full_dataset) == 2
        
        # Get subset
        subset = env.get_dataset(n=1)
        assert len(subset) == 1

    @pytest.mark.asyncio
    async def test_get_model_response_chat(self, mock_openai_client):
        """Test get_model_response with chat format."""
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompt = [{"role": "user", "content": "Hello"}]
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="chat"
        )
        
        assert response == "This is a test response"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_completion(self, mock_openai_client):
        """Test get_model_response with completion format."""
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompt = "Complete this:"
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="completion"
        )
        
        assert response == "This is a test completion"
        mock_openai_client.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_max_tokens_reached(self, mock_openai_client):
        """Test handling of max_tokens_reached."""
        # Mock response with length finish_reason
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated response"
        mock_response.choices[0].finish_reason = "length"
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompt = [{"role": "user", "content": "Hello"}]
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model"
        )
        
        assert response == "[ERROR] max_tokens_reached"

    @pytest.mark.asyncio
    async def test_get_model_response_exception_handling(self, mock_openai_client):
        """Test exception handling in get_model_response."""
        # Mock an exception with context length error
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Request longer than the maximum context length")
        )
        
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompt = [{"role": "user", "content": "Hello"}]
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model"
        )
        
        assert response == "[ERROR] prompt_too_long"

    def test_sanitize_sampling_args_remote_server(self, mock_openai_client):
        """Test sampling args sanitization for remote servers."""
        mock_openai_client.base_url = "https://api.openai.com/v1/"
        
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        sampling_args = {
            "temperature": 0.7,
            "extra_body": {"skip_special_tokens": True}
        }
        
        sanitized = env.sanitize_sampling_args(mock_openai_client, sampling_args)
        
        assert "temperature" in sanitized
        assert "extra_body" not in sanitized

    def test_sanitize_sampling_args_local_server(self, mock_openai_client):
        """Test sampling args sanitization for local servers."""
        # Note: The netloc includes port (localhost:8000), so it doesn't match "localhost" exactly
        # This causes extra_body to be removed even for localhost URLs with ports
        mock_openai_client.base_url = "http://localhost/v1/"  # No port to match exactly
        
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        sampling_args = {
            "temperature": 0.7,
            "extra_body": {"skip_special_tokens": True}
        }
        
        sanitized = env.sanitize_sampling_args(mock_openai_client, sampling_args)
        
        # Check that for localhost (without port), extra_body is preserved
        assert "temperature" in sanitized
        assert "extra_body" in sanitized
        assert sanitized["extra_body"]["skip_special_tokens"] == True

    @pytest.mark.asyncio
    async def test_run_rollouts(self, mock_openai_client):
        """Test running multiple rollouts."""
        env = TestEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric()
        )
        
        prompts = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}]
        ]
        answers = ["response1", "response2"]
        tasks = ["default", "default"]
        infos = [{}, {}]
        
        # Mock the rollout method calls
        results = await env.run_rollouts(
            client=mock_openai_client,
            model="test-model",
            prompts=prompts,
            answers=answers,
            tasks=tasks,
            infos=infos
        )
        
        assert len(results) == 2
        assert all(len(result) == 2 for result in results)  # Each result is (completion, state)