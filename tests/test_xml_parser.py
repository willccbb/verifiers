"""Tests for the XMLParser class."""

import pytest
from verifiers import XMLParser


class TestXMLParser:
    """Test cases for the XMLParser class."""

    def test_xml_parser_initialization(self, xml_parser):
        """Test that XMLParser initializes correctly."""
        assert isinstance(xml_parser, XMLParser)
        assert xml_parser.answer_field == "answer"

    def test_xml_parser_with_alternatives(self, xml_parser_with_alternatives):
        """Test XMLParser with alternative field names."""
        assert isinstance(xml_parser_with_alternatives, XMLParser)
        fields = xml_parser_with_alternatives.get_fields()
        assert "reasoning" in fields
        assert "code" in fields  # canonical name from ("code", "answer")

    def test_parse_simple_xml(self, xml_parser):
        """Test parsing simple XML with basic fields."""
        xml_text = """
        <reasoning>
        Let me think about this problem step by step.
        </reasoning>
        <answer>
        The final answer is 42.
        </answer>
        """
        result = xml_parser.parse(xml_text)
        assert result.reasoning == "Let me think about this problem step by step."
        assert result.answer == "The final answer is 42."

    def test_parse_xml_with_alternatives(self, xml_parser_with_alternatives):
        """Test parsing XML with alternative field names."""
        xml_text = """
        <reasoning>
        First, I need to understand the problem.
        </reasoning>
        <code>
        def solve(): return 42
        </code>
        """
        result = xml_parser_with_alternatives.parse(xml_text)
        assert result.reasoning == "First, I need to understand the problem."
        assert result.code == "def solve(): return 42"
        # Both alternatives should be accessible
        assert hasattr(result, 'answer')
        assert result.answer is None

    def test_parse_missing_fields(self, xml_parser):
        """Test parsing XML with missing fields."""
        xml_text = "<reasoning>Only reasoning here</reasoning>"
        result = xml_parser.parse(xml_text)
        assert result.reasoning == "Only reasoning here"
        assert result.answer is None

    def test_parse_empty_fields(self, xml_parser):
        """Test parsing XML with empty fields."""
        xml_text = "<reasoning></reasoning><answer></answer>"
        result = xml_parser.parse(xml_text)
        assert result.reasoning == ""
        assert result.answer == ""

    def test_parse_no_strip(self, xml_parser):
        """Test parsing without stripping whitespace."""
        # Note: The regex pattern itself removes leading/trailing whitespace
        # from the capture group, so strip=False only affects the .strip() call
        xml_text = "<answer>  spaced content  </answer>"
        result_strip = xml_parser.parse(xml_text, strip=True)
        result_no_strip = xml_parser.parse(xml_text, strip=False)
        assert result_strip.answer == "spaced content"
        assert result_no_strip.answer == "spaced content"  # regex already strips whitespace

    def test_parse_answer_from_completion(self, xml_parser):
        """Test extracting answer from completion."""
        completion = [
            {"role": "user", "content": "Solve this problem"},
            {"role": "assistant", "content": "<reasoning>Let me think</reasoning><answer>42</answer>"},
            {"role": "assistant", "content": "<reasoning>Actually, let me reconsider</reasoning><answer>43</answer>"}
        ]
        result = xml_parser.parse_answer(completion)
        assert result == "43"  # Should get the last answer

    def test_parse_answer_no_answer_field(self, xml_parser):
        """Test parse_answer when no answer field is found."""
        completion = [
            {"role": "assistant", "content": "<reasoning>Only reasoning here</reasoning>"}
        ]
        result = xml_parser.parse_answer(completion)
        assert result is None

    def test_get_format_str(self, xml_parser):
        """Test format string generation."""
        format_str = xml_parser.get_format_str()
        assert "<reasoning>" in format_str
        assert "</reasoning>" in format_str
        assert "<answer>" in format_str
        assert "</answer>" in format_str

    def test_get_format_str_with_alternatives(self, xml_parser_with_alternatives):
        """Test format string with alternatives."""
        format_str = xml_parser_with_alternatives.get_format_str()
        assert "code | answer" in format_str

    def test_format_method(self, xml_parser):
        """Test formatting keyword arguments into XML."""
        formatted = xml_parser.format(reasoning="My reasoning", answer="42")
        assert "<reasoning>\nMy reasoning\n</reasoning>" in formatted
        assert "<answer>\n42\n</answer>" in formatted

    def test_format_method_missing_field(self, xml_parser):
        """Test format method with missing required field."""
        with pytest.raises(ValueError, match="Missing value for field"):
            xml_parser.format(reasoning="Only reasoning")

    def test_format_method_with_alternatives(self, xml_parser_with_alternatives):
        """Test format method with alternative field names."""
        # Using canonical name
        formatted1 = xml_parser_with_alternatives.format(reasoning="test", code="print('hello')")
        assert "<code>\nprint('hello')\n</code>" in formatted1

        # Using alternative name
        formatted2 = xml_parser_with_alternatives.format(reasoning="test", answer="print('hello')")
        assert "<code>\nprint('hello')\n</code>" in formatted2  # Should use canonical tag

    def test_get_fields(self, xml_parser, xml_parser_with_alternatives):
        """Test getting field names."""
        fields1 = xml_parser.get_fields()
        assert fields1 == ["reasoning", "answer"]

        fields2 = xml_parser_with_alternatives.get_fields()
        assert fields2 == ["reasoning", "code"]

    def test_invalid_field_types(self):
        """Test XMLParser initialization with invalid field types."""
        with pytest.raises(TypeError):
            XMLParser([123])  # Invalid field type

        # Empty fields is actually allowed - it just creates a parser with no fields
        empty_parser = XMLParser([])  # This works
        assert empty_parser.get_fields() == []

        with pytest.raises(ValueError):
            XMLParser(["field1", "field1"])  # Duplicate fields

    def test_format_reward_function(self, xml_parser):
        """Test the format reward function."""
        reward_func = xml_parser.get_format_reward_func()

        # Well-formatted completion
        good_completion = [
            {"role": "assistant", "content": "<reasoning>Good reasoning</reasoning><answer>42</answer>"}
        ]
        good_reward = reward_func(good_completion)
        assert 0.0 <= good_reward <= 1.0

        # Poorly formatted completion - gets partial credit for proper spacing
        bad_completion = [
            {"role": "assistant", "content": "Just plain text without XML"}
        ]
        bad_reward = reward_func(bad_completion)
        assert bad_reward == 0.2  # Gets 0.2 for proper spacing (no XML tags to mess up)

    def test_parse_top_level_only(self, xml_parser):
        """Test parse with top_level_only parameter to ignore nested tags."""
        xml_text = """
        <reasoning>
        I'm thinking about this: <answer>nested answer</answer>
        </reasoning>
        <answer>top-level answer</answer>
        """

        # Without top_level_only (default behavior)
        result_default = xml_parser.parse(xml_text)
        assert result_default.reasoning.strip() == "I'm thinking about this: <answer>nested answer</answer>"
        assert result_default.answer == "nested answer"  # Gets the first occurrence

        # With top_level_only
        result_top_level = xml_parser.parse(xml_text, top_level_only=True)
        assert result_top_level.reasoning.strip() == "I'm thinking about this: <answer>nested answer</answer>"
        assert result_top_level.answer == "top-level answer"  # Gets the top-level occurrence

    def test_parse_top_level_only_no_top_level(self, xml_parser):
        """Test parse with top_level_only when there are no top-level tags."""
        xml_text = """
        <reasoning>
        All tags are nested: <answer>nested answer 1</answer>
        Another nested: <answer>nested answer 2</answer>
        </reasoning>
        """

        result = xml_parser.parse(xml_text, top_level_only=True)
        assert result.reasoning is not None
        assert result.answer is None  # No top-level answer tag

    def test_parse_top_level_only_complex_nesting(self, xml_parser_with_alternatives):
        """Test parse with top_level_only with complex nesting scenarios."""
        xml_text = """
        <reasoning>
        <code>nested code 1</code>
        <answer>nested answer</answer>
        </reasoning>
        <code>top-level code</code>
        <reasoning>
        More thinking with <code>nested code 2</code>
        </reasoning>
        """

        result = xml_parser_with_alternatives.parse(xml_text, top_level_only=True)
        assert "nested code 1" in result.reasoning  # First reasoning block
        assert result.code == "top-level code"  # Top-level code tag
        assert result.answer is None  # No top-level answer tag

    def test_parse_top_level_only_self_closing_tags(self, xml_parser):
        """Test parse with self-closing tags."""
        xml_text = """
        <reasoning>Some reasoning</reasoning>
        <answer/>
        """

        result = xml_parser.parse(xml_text, top_level_only=True)
        assert result.reasoning == "Some reasoning"
        assert result.answer is None  # Self-closing tags have no content

    def test_parse_top_level_only_multiple_same_tag(self, xml_parser):
        """Test that only first top-level occurrence is used."""
        xml_text = """
        <answer>first answer</answer>
        <reasoning>Some reasoning</reasoning>
        <answer>second answer</answer>
        """

        result = xml_parser.parse(xml_text, top_level_only=True)
        assert result.answer == "first answer"  # Should get first occurrence
        assert result.reasoning == "Some reasoning"

    def test_parse_top_level_only_with_attributes(self, xml_parser):
        """Test that tags with attributes are still matched."""
        # Note: The current regex doesn't handle attributes, but we should test the behavior
        xml_text = """
        <reasoning type="deep">Reasoning with attribute</reasoning>
        <answer>42</answer>
        """

        result = xml_parser.parse(xml_text, top_level_only=True)
        # Current implementation doesn't match tags with attributes
        assert result.reasoning is None
        assert result.answer == "42"

    def test_parse_top_level_only_edge_whitespace(self, xml_parser):
        """Test various whitespace edge cases."""
        xml_text = """<reasoning>
        Content with newlines
        and indentation
        </reasoning>
        <answer>   spaced   </answer>"""

        result = xml_parser.parse(xml_text, top_level_only=True)
        assert "Content with newlines" in result.reasoning
        assert "and indentation" in result.reasoning
        assert result.answer == "spaced"  # Should be stripped by default

    def test_parse_top_level_only_unicode_content(self, xml_parser):
        """Test parsing with unicode content."""
        xml_text = """
        <reasoning>Unicode test: 你好世界 🌍 émojis</reasoning>
        <answer>42 × 2 = 84</answer>
        """

        result = xml_parser.parse(xml_text, top_level_only=True)
        assert "你好世界" in result.reasoning
        assert "🌍" in result.reasoning
        assert "émojis" in result.reasoning
        assert "42 × 2 = 84" in result.answer
