"""
Comprehensive test suite for Parser classes.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from verifiers.parsers.parser import Parser
from verifiers.parsers.xml_parser import XMLParser
from verifiers.parsers.smola_parser import SmolaParser


class TestParser:
    """Test the base Parser class."""
    
    def test_init_default(self):
        parser = Parser()
        assert hasattr(parser, 'logger')
        assert parser.logger.name == "verifiers.parsers.Parser"
    
    def test_init_with_kwargs(self):
        parser = Parser(custom_attr="test_value", another_attr=42)
        assert parser.custom_attr == "test_value"
        assert parser.another_attr == 42
    
    def test_parse_returns_text_as_is(self):
        parser = Parser()
        text = "This is test text"
        result = parser.parse(text)
        assert result == text
    
    def test_get_assistant_messages(self):
        parser = Parser()
        completion = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
        assistant_messages = parser.get_assistant_messages(completion)
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["content"] == "Hi there!"
        assert assistant_messages[1]["content"] == "I'm doing well!"
    
    def test_get_assistant_messages_empty(self):
        parser = Parser()
        completion = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        assistant_messages = parser.get_assistant_messages(completion)
        assert len(assistant_messages) == 0
    
    def test_parse_answer_with_string(self):
        parser = Parser()
        result = parser.parse_answer("test answer")
        assert result == "test answer"
    
    def test_parse_answer_with_completion_list(self):
        parser = Parser()
        completion = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]
        result = parser.parse_answer(completion)
        assert result == "Answer"
    
    def test_parse_answer_with_empty_completion(self):
        parser = Parser()
        completion = []
        with pytest.raises(IndexError):
            parser.parse_answer(completion)
    
    def test_get_format_reward_func(self):
        parser = Parser()
        reward_func = parser.get_format_reward_func()
        assert callable(reward_func)
        
        # Test the reward function
        completion = [{"role": "assistant", "content": "test"}]
        reward = reward_func(completion)
        assert reward == 1.0


class TestXMLParser:
    """Test the XMLParser class."""
    
    def test_init_simple_fields(self):
        parser = XMLParser(["reasoning", "answer"])
        assert len(parser._fields) == 2
        assert parser._fields[0] == ("reasoning", ["reasoning"])
        assert parser._fields[1] == ("answer", ["answer"])
        assert parser.answer_field == "answer"
    
    def test_init_tuple_fields(self):
        parser = XMLParser([("code", "answer"), "reasoning"])
        assert len(parser._fields) == 2
        assert parser._fields[0] == ("code", ["code", "answer"])
        assert parser._fields[1] == ("reasoning", ["reasoning"])
    
    def test_init_custom_answer_field(self):
        parser = XMLParser(["thinking", "result"], answer_field="result")
        assert parser.answer_field == "result"
    
    def test_init_empty_tuple_error(self):
        with pytest.raises(ValueError, match="Field tuple cannot be empty"):
            XMLParser([(), "reasoning"])
    
    def test_init_duplicate_field_error(self):
        with pytest.raises(ValueError, match="Duplicate field name"):
            XMLParser(["reasoning", "reasoning"])
    
    def test_init_invalid_field_type_error(self):
        with pytest.raises(TypeError, match="Each field must be a string or a tuple"):
            XMLParser([123, "reasoning"])
    
    def test_init_invalid_tuple_element_error(self):
        with pytest.raises(TypeError, match="All alternatives in a tuple must be strings"):
            XMLParser([("code", 123), "reasoning"])
    
    def test_parse_simple_xml(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning>Think step by step</reasoning>\n<answer>42</answer>"
        result = parser.parse(text)
        
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'answer')
        assert result.reasoning == "Think step by step"
        assert result.answer == "42"
    
    def test_parse_with_whitespace(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning>\n  Think step by step\n  </reasoning>\n<answer>\n  42\n  </answer>"
        result = parser.parse(text)
        
        assert result.reasoning == "Think step by step"
        assert result.answer == "42"
    
    def test_parse_without_strip(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning>\n  Think step by step\n  </reasoning>\n<answer>\n  42\n  </answer>"
        result = parser.parse(text, strip=False)
        
        # The regex pattern \s*(.*?)\s* captures content without leading/trailing whitespace
        # but strip=False prevents additional stripping of the captured content
        assert result.reasoning == "Think step by step"
        assert result.answer == "42"
    
    def test_parse_missing_fields(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning>Think step by step</reasoning>"
        result = parser.parse(text)
        
        assert result.reasoning == "Think step by step"
        assert result.answer is None
    
    def test_parse_alternative_tags(self):
        parser = XMLParser([("code", "answer"), "reasoning"])
        text = "<reasoning>Think</reasoning>\n<answer>42</answer>"
        result = parser.parse(text)
        
        assert result.reasoning == "Think"
        assert result.code is None
        assert result.answer == "42"
    
    def test_parse_multiline_content(self):
        parser = XMLParser(["code"])
        text = "<code>\ndef solve():\n    return 42\n</code>"
        result = parser.parse(text)
        
        assert "def solve():" in result.code
        assert "return 42" in result.code
    
    def test_parse_answer_from_completion(self):
        parser = XMLParser(["reasoning", "answer"])
        completion = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<reasoning>Simple addition</reasoning>\n<answer>4</answer>"}
        ]
        result = parser.parse_answer(completion)
        assert result == "4"
    
    def test_parse_answer_from_multiple_messages(self):
        parser = XMLParser(["reasoning", "answer"])
        completion = [
            {"role": "assistant", "content": "<reasoning>First thought</reasoning>"},
            {"role": "assistant", "content": "<reasoning>Better thought</reasoning>\n<answer>final</answer>"}
        ]
        result = parser.parse_answer(completion)
        assert result == "final"
    
    def test_parse_answer_no_answer_field(self):
        parser = XMLParser(["reasoning", "answer"])
        completion = [
            {"role": "assistant", "content": "<reasoning>Just thinking</reasoning>"}
        ]
        result = parser.parse_answer(completion)
        assert result is None
    
    def test_get_format_str_simple(self):
        parser = XMLParser(["reasoning", "answer"])
        format_str = parser.get_format_str()
        expected = "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
        assert format_str == expected
    
    def test_get_format_str_with_alternatives(self):
        parser = XMLParser([("code", "answer"), "reasoning"])
        format_str = parser.get_format_str()
        assert "<[ code | answer ]>" in format_str
        assert "</[ code | answer ]>" in format_str
        assert "<reasoning>" in format_str
    
    def test_get_fields(self):
        parser = XMLParser([("code", "answer"), "reasoning"])
        fields = parser.get_fields()
        assert fields == ["code", "reasoning"]
    
    def test_format_simple(self):
        parser = XMLParser(["reasoning", "answer"])
        formatted = parser.format(reasoning="Think step by step", answer="42")
        expected = "<reasoning>\nThink step by step\n</reasoning>\n<answer>\n42\n</answer>"
        assert formatted == expected
    
    def test_format_with_alternatives(self):
        parser = XMLParser([("code", "answer"), "reasoning"])
        formatted = parser.format(reasoning="Think", answer="42")
        expected = "<code>\n42\n</code>\n<reasoning>\nThink\n</reasoning>"
        assert formatted == expected
    
    def test_format_missing_field_error(self):
        parser = XMLParser(["reasoning", "answer"])
        with pytest.raises(ValueError, match="Missing value for field"):
            parser.format(reasoning="Think")
    
    def test_get_format_reward_func(self):
        parser = XMLParser(["reasoning", "answer"])
        reward_func = parser.get_format_reward_func()
        assert callable(reward_func)
        
        # Test well-formatted message
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>42</answer>"}]
        reward = reward_func(completion)
        assert reward > 0.0
        
        # Test poorly formatted message
        completion = [{"role": "assistant", "content": "Just plain text"}]
        reward = reward_func(completion)
        # Still gets 0.2 for correct spacing even without XML fields
        assert reward == 0.2
    
    def test_format_reward_func_partial_format(self):
        parser = XMLParser(["reasoning", "answer"])
        reward_func = parser.get_format_reward_func()
        
        # Test message with only one field
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>"}]
        reward = reward_func(completion)
        assert 0.0 < reward < 1.0
    
    def test_format_reward_func_multiple_messages(self):
        parser = XMLParser(["reasoning", "answer"])
        reward_func = parser.get_format_reward_func()
        
        completion = [
            {"role": "assistant", "content": "<reasoning>Think</reasoning>\n<answer>42</answer>"},
            {"role": "assistant", "content": "Plain text"}
        ]
        reward = reward_func(completion)
        assert 0.0 < reward < 1.0


class TestSmolaParser:
    """Test the SmolaParser class."""
    
    def test_init(self):
        parser = SmolaParser(["reasoning", "tool"])
        assert len(parser._fields) == 2
        assert parser._fields[0] == ("reasoning", ["reasoning"])
        assert parser._fields[1] == ("tool", ["tool"])
    
    def test_parse_basic_xml(self):
        parser = SmolaParser(["reasoning", "tool"])
        text = "<reasoning>I need to calculate</reasoning>\n<tool>{\"name\": \"calc\", \"args\": {}}</tool>"
        result = parser.parse(text)
        
        assert result.reasoning == "I need to calculate"
        assert result.tool == "{\"name\": \"calc\", \"args\": {}}"
    
    def test_parse_tool_with_json(self):
        parser = SmolaParser(["reasoning", "tool"])
        text = '<tool>{"name": "calculator", "args": {"expression": "2+2"}}</tool>'
        result = parser.parse(text)
        
        assert '"name": "calculator"' in result.tool
        assert '"expression": "2+2"' in result.tool
    
    def test_parse_invalid_json_in_tool(self):
        parser = SmolaParser(["reasoning", "tool"])
        text = '<tool>{"name": "calc", invalid json}</tool>'
        result = parser.parse(text)
        
        # Should still parse even with invalid JSON
        assert result.tool == '{"name": "calc", invalid json}'
    
    def test_get_format_reward_func(self):
        parser = SmolaParser(["reasoning", "tool"])
        reward_func = parser.get_format_reward_func()
        assert callable(reward_func)
        
        # Test well-formatted message
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<tool>{}</tool>"}]
        reward = reward_func(completion)
        assert reward > 0.0
    
    def test_get_fields(self):
        parser = SmolaParser([("code", "answer"), "reasoning"])
        fields = parser.get_fields()
        assert fields == ["code", "reasoning"]
    
    def test_format(self):
        parser = SmolaParser(["reasoning", "tool"])
        formatted = parser.format(reasoning="Think", tool='{"name": "calc"}')
        expected = '<reasoning>\nThink\n</reasoning>\n<tool>\n{"name": "calc"}\n</tool>'
        assert formatted == expected
    
    def test_format_reward_func_with_kwargs(self):
        parser = SmolaParser(["reasoning", "tool"])
        reward_func = parser.get_format_reward_func()
        
        completion = [{"role": "assistant", "content": "<reasoning>Think</reasoning>\n<tool>{}</tool>"}]
        reward = reward_func(completion, extra_param="test")
        assert reward > 0.0


class TestParserIntegration:
    """Integration tests across different parser types."""
    
    def test_parser_inheritance(self):
        """Test that XML and Smola parsers inherit from base Parser."""
        xml_parser = XMLParser(["reasoning", "answer"])
        smola_parser = SmolaParser(["reasoning", "tool"])
        
        assert isinstance(xml_parser, Parser)
        assert isinstance(smola_parser, Parser)
    
    def test_parser_polymorphism(self):
        """Test that parsers can be used polymorphically."""
        parsers = [
            Parser(),
            XMLParser(["reasoning", "answer"]),
            SmolaParser(["reasoning", "tool"])
        ]
        
        test_text = "This is test text"
        for parser in parsers:
            result = parser.parse(test_text)
            assert result is not None
    
    def test_all_parsers_have_format_reward_func(self):
        """Test that all parsers implement get_format_reward_func."""
        parsers = [
            Parser(),
            XMLParser(["reasoning", "answer"]),
            SmolaParser(["reasoning", "tool"])
        ]
        
        for parser in parsers:
            reward_func = parser.get_format_reward_func()
            assert callable(reward_func)
            
            # Test with basic completion
            completion = [{"role": "assistant", "content": "test"}]
            reward = reward_func(completion)
            assert isinstance(reward, float)
            assert 0.0 <= reward <= 1.0


class TestParserEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_xml_parser_malformed_xml(self):
        parser = XMLParser(["reasoning", "answer"])
        
        # Unclosed tag
        text = "<reasoning>Think step by step"
        result = parser.parse(text)
        assert result.reasoning is None
        
        # Mismatched tags
        text = "<reasoning>Think</answer>"
        result = parser.parse(text)
        assert result.reasoning is None
    
    def test_xml_parser_nested_tags(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning>Think <answer>42</answer> more</reasoning>"
        result = parser.parse(text)
        
        # Should match the first occurrence
        assert "Think" in result.reasoning
        assert "more" in result.reasoning
    
    def test_xml_parser_empty_tags(self):
        parser = XMLParser(["reasoning", "answer"])
        text = "<reasoning></reasoning>\n<answer></answer>"
        result = parser.parse(text)
        
        assert result.reasoning == ""
        assert result.answer == ""
    
    def test_parser_with_special_characters(self):
        parser = XMLParser(["code"])
        text = "<code>\nprint('Hello, World!')\nif x > 0:\n    print(\"Positive\")\n</code>"
        result = parser.parse(text)
        
        assert "print('Hello, World!')" in result.code
        assert "if x > 0:" in result.code
    
    def test_format_reward_func_no_assistant_messages(self):
        parser = XMLParser(["reasoning", "answer"])
        reward_func = parser.get_format_reward_func()
        
        completion = [{"role": "user", "content": "Question"}]
        reward = reward_func(completion)
        assert reward == 0.0
    
    def test_format_reward_func_empty_completion(self):
        parser = XMLParser(["reasoning", "answer"])
        reward_func = parser.get_format_reward_func()
        
        completion = []
        reward = reward_func(completion)
        assert reward == 0.0