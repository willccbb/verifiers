"""Tests for the MultiTurnEnv class."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datasets import Dataset
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric


class TestMultiTurnEnv:
    """Test cases for the MultiTurnEnv class."""

    def test_multiturn_env_initialization(self, mock_multiturn_env):
        """Test MultiTurnEnv initialization."""
        assert mock_multiturn_env.max_turns == 3
        assert mock_multiturn_env.message_type == 'chat'  # Default from parent

    def test_multiturn_env_default_max_turns(self, mock_openai_client, sample_chat_dataset):
        """Test MultiTurnEnv default max_turns value."""
        from tests.conftest import SimpleMultiTurnEnv
        env = SimpleMultiTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            parser=Parser(),
            rubric=Rubric()
        )
        assert env.max_turns == 10  # Default value

    @pytest.mark.asyncio
    async def test_basic_multiturn_rollout(self, mock_multiturn_env):
        """Test basic multi-turn conversation that completes normally."""
        # Configure mock to return responses that lead to completion
        prompt = [{"role": "user", "content": "Start conversation"}]
        
        # Set up responses for the conversation turns
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start conversation"}],
            response="First response"
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"}
            ],
            response="Second response"
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"},
                {"role": "assistant", "content": "Second response"},
                {"role": "user", "content": "Please finish with DONE"}
            ],
            response="Final response DONE"
        )
        
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="target_answer"
        )
        
        # Should have: assistant + user + assistant + user + assistant
        assert len(completion) == 5
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "First response"
        assert completion[1]["role"] == "user" 
        assert completion[2]["role"] == "assistant"
        assert completion[2]["content"] == "Second response"
        assert completion[4]["content"] == "Final response DONE"
        
        assert state["answer"] == "target_answer"

    @pytest.mark.asyncio
    async def test_max_turns_limiting(self, mock_multiturn_env_max_turns):
        """Test that rollout stops at max_turns."""
        # Set up responses that would continue indefinitely
        mock_multiturn_env_max_turns.client.set_default_responses(
            chat_response="Keep going"
        )
        
        prompt = [{"role": "user", "content": "Start conversation"}]
        completion, state = await mock_multiturn_env_max_turns.rollout(
            client=mock_multiturn_env_max_turns.client,
            model="test-model", 
            prompt=prompt,
            answer="target_answer"
        )
        
        # Should stop at max_turns=2: assistant + user + assistant (3 messages)
        assert len(completion) == 3
        assert completion[0]["role"] == "assistant"
        assert completion[1]["role"] == "user"
        assert completion[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_error_handling_stops_rollout(self, mock_multiturn_env):
        """Test that errors stop the rollout immediately."""
        # Set up the mock to return an error response for the expected conversation
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start conversation"}],
            response="[ERROR] Something went wrong"
        )
        
        prompt = [{"role": "user", "content": "Start conversation"}]
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="target_answer"
        )
        
        # Should stop immediately after error
        assert len(completion) == 1
        assert completion[0]["content"] == "[ERROR] Something went wrong"

    @pytest.mark.asyncio
    async def test_immediate_completion(self, mock_multiturn_env):
        """Test completion detection on first turn."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Quick question"}],
            response="Immediate DONE"
        )
        
        prompt = [{"role": "user", "content": "Quick question"}]
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="target_answer"
        )
        
        # Should complete immediately
        assert len(completion) == 1
        assert completion[0]["content"] == "Immediate DONE"

    @pytest.mark.asyncio
    async def test_env_response_integration(self, mock_multiturn_env):
        """Test that environment responses are properly integrated."""
        # Set up responses for the conversation turns
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start conversation"}],
            response="First response"
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"}
            ],
            response="Final response DONE"
        )
        
        prompt = [{"role": "user", "content": "Start conversation"}]
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="target_answer"
        )
        
        # Verify environment responses are included
        assert len(completion) >= 3
        user_messages = [msg for msg in completion if msg["role"] == "user"]
        assert len(user_messages) >= 1
        assert "Continue (turn 1)" in user_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_state_management(self, mock_multiturn_env):
        """Test that state is properly initialized and maintained."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Test state"}],
            response="Quick DONE"
        )
        
        prompt = [{"role": "user", "content": "Test state"}]
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="test_answer"
        )
        
        # State should contain the answer
        assert "answer" in state
        assert state["answer"] == "test_answer"

    @pytest.mark.asyncio
    async def test_prompt_copying(self, mock_multiturn_env):
        """Test that original prompt is not modified."""
        original_prompt = [{"role": "user", "content": "Original message"}]
        prompt_copy = [{"role": "user", "content": "Original message"}]
        
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Original message"}],
            response="Response DONE"
        )
        
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=original_prompt,
            answer="test_answer"
        )
        
        # Original prompt should be unchanged
        assert original_prompt == prompt_copy

    @pytest.mark.asyncio
    async def test_sampling_args_passed_through(self, mock_multiturn_env):
        """Test that sampling arguments are passed to model calls."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Test sampling"}],
            response="Quick DONE"
        )
        
        prompt = [{"role": "user", "content": "Test sampling"}]
        sampling_args = {"temperature": 0.8, "max_tokens": 50}
        
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="test_answer",
            sampling_args=sampling_args
        )
        
        # Verify sampling args were passed
        call_args = mock_multiturn_env.client.chat.completions.create.call_args
        assert "temperature" in call_args.kwargs
        assert "max_tokens" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_task_and_info_parameters(self, mock_multiturn_env):
        """Test rollout with task and info parameters."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Task question"}],
            response="Task DONE"
        )
        
        prompt = [{"role": "user", "content": "Task question"}]
        completion, state = await mock_multiturn_env.rollout(
            client=mock_multiturn_env.client,
            model="test-model",
            prompt=prompt,
            answer="task_answer",
            task="math",
            info={"difficulty": "hard"}
        )
        
        assert len(completion) >= 1
        assert state["answer"] == "task_answer"

    @pytest.mark.asyncio
    async def test_non_list_prompt_assertion(self, mock_multiturn_env):
        """Test that non-list prompts raise AssertionError."""
        with pytest.raises(AssertionError):
            await mock_multiturn_env.rollout(
                client=mock_multiturn_env.client,
                model="test-model",
                prompt="String prompt not allowed",  # Should be list
                answer="test_answer"
            )

    @pytest.mark.asyncio
    async def test_environment_response_state_modification(self, mock_openai_client, sample_chat_dataset):
        """Test that environment can modify state between turns."""
        class StatefulMultiTurnEnv(MultiTurnEnv):
            def is_completed(self, messages, state, **kwargs):
                return state.get("turn_count", 0) >= 2
            
            def env_response(self, messages, state, **kwargs):
                state["turn_count"] = state.get("turn_count", 0) + 1
                return {"role": "user", "content": f"Turn {state['turn_count']}"}, state
        
        env = StatefulMultiTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=5,
            parser=Parser(),
            rubric=Rubric()
        )
        
        env.client.set_default_responses(chat_response="Continue")
        
        prompt = [{"role": "user", "content": "Start"}]
        completion, state = await env.rollout(
            client=env.client,
            model="test-model",
            prompt=prompt,
            answer="test"
        )
        
        # Should complete when turn_count reaches 2
        assert state["turn_count"] == 2
        assert len(completion) >= 3  # Multiple turns with env responses

    def _create_mock_response(self, content):
        """Helper to create mock OpenAI response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "stop"
        return mock_response

    def test_abstract_methods_not_implemented(self):
        """Test that MultiTurnEnv cannot be instantiated directly (abstract class)."""
        # MultiTurnEnv is abstract and should not be instantiable without implementing abstract methods
        with pytest.raises(TypeError):
            # This should fail because MultiTurnEnv has abstract methods
            MultiTurnEnv(
                model="test-model",
                parser=Parser(),
                rubric=Rubric()
            )

    @pytest.mark.asyncio
    async def test_completion_detection_before_env_response(self, mock_openai_client, sample_chat_dataset):
        """Test completion detection works before env_response is called."""
        class ImmediateCompletionEnv(MultiTurnEnv):
            def is_completed(self, messages, state, **kwargs):
                # Complete if we have any assistant message
                return any(msg.get("role") == "assistant" for msg in messages)
            
            def env_response(self, messages, state, **kwargs):
                # This should never be called due to immediate completion
                return {"role": "user", "content": "Should not appear"}, state
        
        env = ImmediateCompletionEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=5,
            parser=Parser(),
            rubric=Rubric()
        )
        
        env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start"}],
            response="First response"
        )
        
        prompt = [{"role": "user", "content": "Start"}]
        completion, state = await env.rollout(
            client=env.client,
            model="test-model",
            prompt=prompt,
            answer="test"
        )
        
        # Should complete immediately after first assistant response
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "First response"