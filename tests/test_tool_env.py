"""Tests for the ToolEnv class with multiple tool call support."""

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

        def multiply_tool(x: int, y: int) -> int:
            """Multiply two numbers.
            
            Args:
                x: First number
                y: Second number
                
            Returns:
                int: Product of x and y
            """
            return x * y

        def greet_tool(name: str = "World") -> str:
            """Greet someone.
            
            Args:
                name: Name to greet
                
            Returns:
                str: Greeting message
            """
            return f"Hello, {name}!"

        return [add_tool, multiply_tool, greet_tool]

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

    def test_single_tool_call(self, tool_env):
        """Test calling a single tool."""
        messages = [
            {"role": "user", "content": "Calculate 2 + 3"},
            {"role": "assistant", "content": '<think>I need to add 2 and 3</think><tool>{"name": "add_tool", "args": {"a": 2, "b": 3}}</tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        assert "5" in response["content"]

    def test_multiple_tool_calls(self, tool_env):
        """Test calling multiple tools in one message."""
        messages = [
            {"role": "user", "content": "Calculate 2 + 3 and then multiply 4 * 5"},
            {"role": "assistant", "content": '''<think>I need to do two calculations</think>
<tool>{"name": "add_tool", "args": {"a": 2, "b": 3}}</tool>
<tool>{"name": "multiply_tool", "args": {"x": 4, "y": 5}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "multiply_tool result:" in content
        assert "5" in content  # Result of 2 + 3
        assert "20" in content  # Result of 4 * 5

    def test_three_tool_calls(self, tool_env):
        """Test calling three tools in one message."""
        messages = [
            {"role": "user", "content": "Do multiple operations"},
            {"role": "assistant", "content": '''<think>Multiple operations</think>
<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>
<tool>{"name": "multiply_tool", "args": {"x": 3, "y": 4}}</tool>
<tool>{"name": "greet_tool", "args": {"name": "Alice"}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "multiply_tool result:" in content
        assert "greet_tool result:" in content
        assert "3" in content  # Result of 1 + 2
        assert "12" in content  # Result of 3 * 4
        assert "Hello, Alice!" in content  # Result of greet_tool

    def test_mixed_valid_invalid_tools(self, tool_env):
        """Test mix of valid and invalid tool calls."""
        messages = [
            {"role": "user", "content": "Mix of valid and invalid"},
            {"role": "assistant", "content": '''<think>Testing mixed calls</think>
<tool>{"name": "add_tool", "args": {"a": 5, "b": 10}}</tool>
<tool>{"name": "invalid_tool", "args": {"x": 1}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "invalid_tool result:" in content  # Tool name is extracted from JSON even if tool doesn't exist
        assert "15" in content  # Valid result from add_tool
        assert "Error:" in content  # Error from invalid_tool

    def test_invalid_json_in_multiple_tools(self, tool_env):
        """Test handling of invalid JSON in multiple tool calls."""
        messages = [
            {"role": "user", "content": "Invalid JSON test"},
            {"role": "assistant", "content": '''<think>Testing invalid JSON</think>
<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>
<tool>{invalid json}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "unknown_tool result:" in content  # Invalid JSON gets "unknown_tool" label
        assert "3" in content  # Valid result
        assert "Error:" in content  # Error from invalid JSON

    def test_backward_compatibility_single_tool(self, tool_env):
        """Test that single tool calls still work (backward compatibility)."""
        messages = [
            {"role": "user", "content": "Single tool test"},
            {"role": "assistant", "content": '<tool>{"name": "greet_tool", "args": {}}</tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        # Should not have tool name prefix for single tool
        assert "greet_tool result:" not in response["content"]
        assert "Hello, World!" in response["content"]

    def test_no_tools_fallback(self, tool_env):
        """Test fallback when no tools are detected."""
        messages = [
            {"role": "user", "content": "No tools"},
            {"role": "assistant", "content": '<think>Just thinking, no tools</think>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        assert "Tool command not found" in response["content"]

    def test_empty_tool_results(self, tool_env):
        """Test handling of tools that return empty results."""
        # Mock a tool that returns empty string
        def empty_tool() -> str:
            """Tool that returns empty string."""
            return ""
        
        tool_env.tools["empty_tool"] = empty_tool
        tool_env.tool_schemas.append({
            "name": "empty_tool",
            "description": "Returns empty string",
            "args": {},
            "returns": "Empty string",
            "examples": []
        })
        
        messages = [
            {"role": "user", "content": "Empty tool test"},
            {"role": "assistant", "content": '''<tool>{"name": "add_tool", "args": {"a": 1, "b": 1}}</tool>
<tool>{"name": "empty_tool", "args": {}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "empty_tool result:" in content
        assert "2" in content  # Result from add_tool

    def test_tool_with_default_args(self, tool_env):
        """Test calling tools with default arguments in multiple calls."""
        messages = [
            {"role": "user", "content": "Test default args"},
            {"role": "assistant", "content": '''<think>Testing default arguments</think>
<tool>{"name": "greet_tool", "args": {}}</tool>
<tool>{"name": "greet_tool", "args": {"name": "Bob"}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "greet_tool result:" in content
        # Both calls use same tool name, so we'll see it twice
        assert "Hello, World!" in content  # Default name
        assert "Hello, Bob!" in content  # Specified name

    def test_tool_exception_handling(self, tool_env):
        """Test handling of tool exceptions in multiple calls."""
        def error_tool() -> str:
            """Tool that raises an exception."""
            raise ValueError("Tool error occurred")
        
        tool_env.tools["error_tool"] = error_tool
        
        messages = [
            {"role": "user", "content": "Test error handling"},
            {"role": "assistant", "content": '''<think>Testing error handling</think>
<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>
<tool>{"name": "error_tool", "args": {}}</tool>
<tool>{"name": "greet_tool", "args": {}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "add_tool result:" in content
        assert "error_tool result:" in content
        assert "greet_tool result:" in content
        assert "3" in content  # Successful result
        assert "Error:" in content  # Error from error_tool
        assert "Hello, World!" in content  # Successful result after error

    def test_empty_results_list_edge_case(self, tool_env):
        """Test the edge case where results list is somehow empty."""
        # This is a bit contrived, but tests the specific condition on line 179
        original_call_tool = tool_env.call_tool
        
        def mock_call_tool(tool_json):
            # Return None/empty to simulate empty results
            return None
        
        tool_env.call_tool = mock_call_tool
        
        messages = [
            {"role": "user", "content": "Empty results test"},
            {"role": "assistant", "content": '<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should handle the None result gracefully
        assert response["role"] == "user"
        
        # Restore original method
        tool_env.call_tool = original_call_tool

    def test_parse_all_exception_handling(self, tool_env):
        """Test exception handling in parse_all."""
        # Mock parse_all to raise an exception
        original_parse_all = tool_env.parser.parse_all
        
        def mock_parse_all(content):
            raise Exception("Parse error")
        
        tool_env.parser.parse_all = mock_parse_all
        
        messages = [
            {"role": "user", "content": "Parse error test"},
            {"role": "assistant", "content": '<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should fall back to error message
        assert response["role"] == "user"
        assert "Error:" in response["content"]
        
        # Restore original method
        tool_env.parser.parse_all = original_parse_all

    def test_missing_tool_attribute(self, tool_env):
        """Test when parsed_all doesn't have tool attribute."""
        # Mock parse_all to return object without tool attribute
        original_parse_all = tool_env.parser.parse_all
        
        def mock_parse_all(content):
            from types import SimpleNamespace
            return SimpleNamespace(other_field=[])
        
        tool_env.parser.parse_all = mock_parse_all
        
        messages = [
            {"role": "user", "content": "Missing tool attribute test"},
            {"role": "assistant", "content": '<tool>{"name": "add_tool", "args": {"a": 1, "b": 2}}</tool>'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        # Should fall back to single parse
        assert response["role"] == "user"
        
        # Restore original method
        tool_env.parser.parse_all = original_parse_all

    def test_tool_results_with_special_characters(self, tool_env):
        """Test tool results containing newlines and special characters."""
        def special_output_tool() -> str:
            """Tool that returns output with special characters."""
            return "Line 1\nLine 2\n\nTab:\tHere\nSpecial: !@#$%^&*()"
        
        tool_env.tools["special_output_tool"] = special_output_tool
        
        messages = [
            {"role": "user", "content": "Special characters test"},
            {"role": "assistant", "content": '''<think>Testing special characters</think>
<tool>{"name": "special_output_tool", "args": {}}</tool>
<tool>{"name": "greet_tool", "args": {}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "special_output_tool result:" in content
        assert "greet_tool result:" in content
        assert "Line 1\nLine 2" in content
        assert "Tab:\tHere" in content
        assert "Special: !@#$%^&*()" in content
        assert "Hello, World!" in content

    def test_many_tool_calls(self, tool_env):
        """Test performance with many tool calls (10+)."""
        # Create message with 15 tool calls
        tools_content = "\n".join([
            f'<tool>{{"name": "add_tool", "args": {{"a": {i}, "b": {i+1}}}}}</tool>' 
            for i in range(15)
        ])
        
        messages = [
            {"role": "user", "content": "Many tools test"},
            {"role": "assistant", "content": f'<think>Testing many tools</think>\n{tools_content}'}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        
        # Should have all 15 results with add_tool labels
        assert content.count("add_tool result:") == 15
        
        # Check some specific calculations
        assert str(0 + 1) in content  # First result: 0+1=1
        assert str(14 + 15) in content  # Last result: 14+15=29

    def test_max_chars_with_multiple_tools(self, tool_env):
        """Test max_chars truncation with multiple tool calls."""
        def long_output_tool() -> str:
            """Tool that returns very long output."""
            return "A" * 2000  # Very long string
        
        tool_env.tools["long_output_tool"] = long_output_tool
        
        messages = [
            {"role": "user", "content": "Long output test"},
            {"role": "assistant", "content": '''<think>Testing long output</think>
<tool>{"name": "long_output_tool", "args": {}}</tool>
<tool>{"name": "greet_tool", "args": {}}</tool>'''}
        ]
        state = {}
        
        response, new_state = tool_env.env_response(messages, state)
        
        assert response["role"] == "user"
        content = response["content"]
        assert "long_output_tool result:" in content
        assert "greet_tool result:" in content
        # Long output should be truncated if max_chars is set
        assert "Hello, World!" in content  # Second tool should still work