"""Tests for the ToolEnv class."""

import pytest
from unittest.mock import MagicMock
from verifiers import XMLParser
from verifiers.envs.tool_env import ToolEnv


class TestToolEnv:
    """Test cases for the ToolEnv class."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        def add_tool(a: int, b: int) -> int:
            """Add two numbers together.
            
            Args:
                a: First number
                b: Second number
                
            Returns:
                int: Sum of a and b
            """
            return a + b

        return [add_tool]

    @pytest.fixture
    def tool_env(self, mock_tools, sample_dataset):
        """Create a ToolEnv instance with mock tools."""
        parser = XMLParser(fields=["think", ("tool", "answer")])
        return ToolEnv(
            tools=mock_tools,
            parser=parser,
            system_prompt="Test system prompt with {tool_descriptions}",
            max_turns=5,
            dataset=sample_dataset
        )

    def test_nested_tool_tags_not_detected(self, tool_env):
        """Test that tool tags inside think tags are not detected and executed."""
        messages = [
            {"role": "user", "content": "Calculate 2 + 3"},
            {"role": "assistant", "content": '''<think>
I need to add 2 and 3. Let me use the tool:
<tool>{"name": "add_tool", "args": {"a": 2, "b": 3}}</tool>
Actually, let me think about this more carefully.
</think>
This is a simple addition problem.'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should return error since no top-level tool tag was found
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_top_level_tool_tag_detected(self, tool_env):
        """Test that top-level tool tags are properly detected and executed."""
        messages = [
            {"role": "user", "content": "Calculate 2 + 3"},
            {"role": "assistant", "content": '''<think>I need to add 2 and 3</think>
<tool>{"name": "add_tool", "args": {"a": 2, "b": 3}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should execute the tool and return result
        assert response["role"] == "user"
        assert "5" in response["content"]  # Result of 2 + 3

    def test_mixed_nested_and_top_level_tool_tags(self, tool_env):
        """Test that only top-level tool tags are executed when both nested and top-level exist."""
        messages = [
            {"role": "user", "content": "Calculate some numbers"},
            {"role": "assistant", "content": '''<think>
Let me think about this problem. I could use:
<tool>{"name": "add_tool", "args": {"a": 1, "b": 1}}</tool>
But that's just in my thinking.
</think>
<tool>{"name": "add_tool", "args": {"a": 2, "b": 3}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should only execute the top-level tool (2 + 3 = 5), not the nested one (1 + 1 = 2)
        assert response["role"] == "user"
        assert "5" in response["content"]  # Result of top-level tool
        assert "2" not in response["content"]  # Nested tool should not execute

    def test_multiple_nested_tool_tags(self, tool_env):
        """Test that multiple nested tool tags in different sections are ignored."""
        messages = [
            {"role": "user", "content": "Calculate something"},
            {"role": "assistant", "content": '''<think>
First approach: <tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>
Second approach: <tool>{"name": "add_tool", "args": {"a": 3, "b": 4}}</tool>
</think>
<answer>
I considered using <tool>{"name": "add_tool", "args": {"a": 5, "b": 6}}</tool> in my answer too.
</answer>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should return error since no top-level tool tag exists
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_deeply_nested_tool_tags(self, tool_env):
        """Test that deeply nested tool tags are ignored."""
        messages = [
            {"role": "user", "content": "Solve this"},
            {"role": "assistant", "content": '''<think>
Let me think step by step:
<answer>
The solution involves: <tool>{"name": "add_tool", "args": {"a": 7, "b": 8}}</tool>
</answer>
</think>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should return error since the tool tag is nested inside answer which is inside think
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_multiple_top_level_tool_tags(self, tool_env):
        """Test that only the first top-level tool tag is executed."""
        messages = [
            {"role": "user", "content": "Calculate multiple things"},
            {"role": "assistant", "content": '''<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>
Some text in between
<tool>{"name": "add_tool", "args": {"a": 3, "b": 4}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should only execute the first tool
        assert response["role"] == "user"
        assert "3" in response["content"]  # Result of first tool (1 + 2)
        assert "7" not in response["content"]  # Second tool should not execute

    def test_tool_tag_nested_in_answer_at_top_level(self, tool_env):
        """Test that tool tags inside top-level answer tags are ignored."""
        messages = [
            {"role": "user", "content": "What's the answer?"},
            {"role": "assistant", "content": '''<answer>
The answer is to use this tool: <tool>{"name": "add_tool", "args": {"a": 10, "b": 20}}</tool>
</answer>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should return error since tool is nested inside answer
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_empty_tool_tag_content(self, tool_env):
        """Test handling of empty tool tag content."""
        messages = [
            {"role": "user", "content": "Empty tool"},
            {"role": "assistant", "content": '<tool></tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should handle empty tool tag gracefully
        assert response["role"] == "user"
        assert "Error:" in response["content"]

    def test_tool_tag_with_whitespace(self, tool_env):
        """Test tool tags with various whitespace patterns."""
        messages = [
            {"role": "user", "content": "Whitespace test"},
            {"role": "assistant", "content": '''<think>
    <tool>  {"name": "add_tool", "args": {"a": 1, "b": 1}}  </tool>
</think>
<tool>
    {"name": "add_tool", "args": {"a": 5, "b": 5}}
</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should execute the top-level tool with whitespace
        assert response["role"] == "user"
        assert "10" in response["content"]  # Result of 5 + 5

    def test_tool_tag_at_content_boundaries(self, tool_env):
        """Test tool tags at the very beginning and end of content."""
        # Test at beginning
        messages1 = [
            {"role": "user", "content": "At beginning"},
            {"role": "assistant", "content": '<tool>{"name": "add_tool", "args": {"a": 1, "b": 1}}</tool>'}
        ]
        state1 = {}
        
        response1, _ = tool_env.env_response(messages1, state1)
        assert "2" in response1["content"]
        
        # Test at end with other content before
        messages2 = [
            {"role": "user", "content": "At end"},
            {"role": "assistant", "content": 'Some preamble text<tool>{"name": "add_tool", "args": {"a": 2, "b": 2}}</tool>'}
        ]
        state2 = {}
        
        response2, _ = tool_env.env_response(messages2, state2)
        assert "4" in response2["content"]

    def test_malformed_nested_tags(self, tool_env):
        """Test behavior with malformed/unclosed nested tags."""
        messages = [
            {"role": "user", "content": "Malformed tags"},
            {"role": "assistant", "content": '''<think>
This has an unclosed tool tag: <tool>{"name": "add_tool", "args": {"a": 1, "b": 1}}
</think>
<tool>{"name": "add_tool", "args": {"a": 9, "b": 9}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # The unclosed tag inside think consumes the closing tag of the second tool,
        # so no valid top-level tool is found
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_properly_closed_nested_and_top_level(self, tool_env):
        """Test behavior with properly closed nested tags and a top-level tag."""
        messages = [
            {"role": "user", "content": "Properly closed tags"},
            {"role": "assistant", "content": '''<think>
This has a properly closed tool tag: <tool>{"name": "add_tool", "args": {"a": 1, "b": 1}}</tool>
</think>
<tool>{"name": "add_tool", "args": {"a": 9, "b": 9}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should execute the top-level tool only
        assert response["role"] == "user"
        assert "18" in response["content"]  # Result of 9 + 9
        assert "2" not in response["content"]  # Nested tool not executed