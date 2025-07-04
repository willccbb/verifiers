"""Pytest configuration and fixtures for verifiers tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datasets import Dataset
from verifiers.parsers import Parser, XMLParser, ThinkParser
from verifiers.envs import Environment, SingleTurnEnv, MultiTurnEnv
from verifiers.rubrics import Rubric


@pytest.fixture
def basic_parser():
    """Return a basic Parser instance."""
    return Parser()


@pytest.fixture
def xml_parser():
    """Return an XMLParser instance with common fields."""
    return XMLParser(
        fields=["reasoning", "answer"],
        answer_field="answer"
    )


@pytest.fixture
def xml_parser_with_alternatives():
    """Return an XMLParser instance with alternative field names."""
    return XMLParser(
        fields=["reasoning", ("code", "answer")],
        answer_field="answer"
    )


@pytest.fixture
def think_parser():
    """Return a ThinkParser instance."""
    return ThinkParser()


@pytest.fixture
def think_parser_with_extractor():
    """Return a ThinkParser instance with custom extraction function."""
    def extract_boxed(text):
        """Simple boxed answer extractor for testing."""
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        return match.group(1) if match else text
    
    return ThinkParser(extract_fn=extract_boxed)


# Async test fixtures for Environment testing

@pytest.fixture
def mock_openai_client():
    """Return a mocked AsyncOpenAI client."""
    client = AsyncMock()
    
    # Mock chat completions
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock()]
    mock_chat_response.choices[0].message.content = "This is a test response"
    mock_chat_response.choices[0].finish_reason = "stop"
    client.chat.completions.create = AsyncMock(return_value=mock_chat_response)
    
    # Mock regular completions - note: this is NOT async in the real OpenAI API
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [MagicMock()]
    mock_completion_response.choices[0].text = "This is a test completion"
    mock_completion_response.choices[0].finish_reason = "stop"
    client.completions.create = MagicMock(return_value=mock_completion_response)  # Not AsyncMock!
    
    return client


@pytest.fixture 
def sample_dataset():
    """Return a sample dataset for testing."""
    return Dataset.from_dict({
        "question": ["What is 2+2?", "What is the capital of France?"],
        "answer": ["4", "Paris"]
    })


@pytest.fixture
def sample_chat_dataset():
    """Return a sample dataset with chat format."""
    return Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is the capital of France?"}]
        ],
        "answer": ["4", "Paris"]
    })


@pytest.fixture
def mock_singleturn_env(mock_openai_client, sample_dataset):
    """Return a SingleTurnEnv with mocked client and dataset."""
    return SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_dataset,
        system_prompt="You are a helpful assistant.",
        parser=Parser(),
        rubric=Rubric()
    )


@pytest.fixture
def mock_singleturn_env_completion(mock_openai_client):
    """Return a SingleTurnEnv for completion format testing."""
    completion_dataset = Dataset.from_dict({
        "prompt": ["Calculate 2+2:", "Name the capital of France:"],
        "answer": ["4", "Paris"]
    })
    return SingleTurnEnv(
        client=mock_openai_client,
        model="test-model", 
        dataset=completion_dataset,
        message_type="completion",
        parser=Parser(),
        rubric=Rubric()
    )


# MultiTurnEnv test fixtures

class SimpleMultiTurnEnv(MultiTurnEnv):
    """Simple concrete implementation of MultiTurnEnv for testing."""
    
    def __init__(self, completion_condition="answer", **kwargs):
        super().__init__(**kwargs)
        self.completion_condition = completion_condition  # "answer", "max_turns", "error"
        self.env_response_count = 0
    
    def is_completed(self, messages, state, **kwargs):
        """Simple completion logic for testing."""
        if self.completion_condition == "answer":
            # Complete when assistant says "DONE"
            if messages and messages[-1].get("role") == "assistant":
                return "DONE" in messages[-1].get("content", "")
        elif self.completion_condition == "max_turns":
            # Never complete naturally (test max_turns)
            return False
        elif self.completion_condition == "error":
            # Complete on any error
            if messages and messages[-1].get("role") == "assistant":
                return messages[-1].get("content", "").startswith("[ERROR]")
        return False
    
    def env_response(self, messages, state, **kwargs):
        """Simple environment response for testing."""
        self.env_response_count += 1
        
        if self.completion_condition == "answer":
            # Encourage completion after a few turns
            if self.env_response_count >= 2:
                return {"role": "user", "content": "Please finish with DONE"}, state
            else:
                return {"role": "user", "content": f"Continue (turn {self.env_response_count})"}, state
        else:
            return {"role": "user", "content": f"Environment response {self.env_response_count}"}, state


@pytest.fixture
def mock_multiturn_env(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv for basic testing."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=3,
        completion_condition="answer",
        parser=Parser(),
        rubric=Rubric()
    )


@pytest.fixture  
def mock_multiturn_env_max_turns(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv that tests max_turns limiting."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=2,
        completion_condition="max_turns",  # Never complete naturally
        parser=Parser(),
        rubric=Rubric()
    )