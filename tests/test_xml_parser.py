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

    def test_parse_all_single_occurrence(self, xml_parser):
        """Test parse_all with single occurrence of each field."""
        xml_text = """
        <reasoning>Single reasoning</reasoning>
        <answer>Single answer</answer>
        """
        result = xml_parser.parse_all(xml_text)
        assert result.reasoning == ["Single reasoning"]
        assert result.answer == ["Single answer"]

    def test_parse_all_multiple_occurrences(self, xml_parser):
        """Test parse_all with multiple occurrences of the same field."""
        xml_text = """
        <reasoning>First reasoning</reasoning>
        <answer>First answer</answer>
        <reasoning>Second reasoning</reasoning>
        <answer>Second answer</answer>
        """
        result = xml_parser.parse_all(xml_text)
        assert result.reasoning == ["First reasoning", "Second reasoning"]
        assert result.answer == ["First answer", "Second answer"]

    def test_parse_all_no_occurrences(self, xml_parser):
        """Test parse_all with no occurrences of fields."""
        xml_text = "Just plain text with no XML tags"
        result = xml_parser.parse_all(xml_text)
        assert result.reasoning == []
        assert result.answer == []

    def test_parse_all_mixed_occurrences(self, xml_parser):
        """Test parse_all with mixed occurrences (some fields present, others not)."""
        xml_text = """
        <reasoning>Only reasoning here</reasoning>
        <reasoning>More reasoning</reasoning>
        """
        result = xml_parser.parse_all(xml_text)
        assert result.reasoning == ["Only reasoning here", "More reasoning"]
        assert result.answer == []

    def test_parse_all_with_alternatives(self, xml_parser_with_alternatives):
        """Test parse_all with alternative field names."""
        xml_text = """
        <reasoning>First reasoning</reasoning>
        <code>First code</code>
        <reasoning>Second reasoning</reasoning>
        <answer>Alternative answer</answer>
        """
        result = xml_parser_with_alternatives.parse_all(xml_text)
        assert result.reasoning == ["First reasoning", "Second reasoning"]
        assert result.code == ["First code"]
        assert result.answer == ["Alternative answer"]

    def test_parse_all_no_strip(self, xml_parser):
        """Test parse_all without stripping whitespace."""
        xml_text = """
        <reasoning>  spaced reasoning  </reasoning>
        <answer>  spaced answer  </answer>
        """
        result_strip = xml_parser.parse_all(xml_text, strip=True)
        result_no_strip = xml_parser.parse_all(xml_text, strip=False)

        assert result_strip.reasoning == ["spaced reasoning"]
        assert result_strip.answer == ["spaced answer"]
        assert result_no_strip.reasoning == ["spaced reasoning"]  # regex pattern strips
        assert result_no_strip.answer == ["spaced answer"]

    def test_parse_all_malformed_xml(self, xml_parser):
        """Test parse_all with malformed XML tags."""
        xml_text = """
        <reasoning>Good reasoning</reasoning>
        <answer>Good answer</answer>
        <reasoning>Unclosed reasoning without proper closing
        <reasoning>Another good reasoning</reasoning>
        """
        result = xml_parser.parse_all(xml_text)
        # Regex will find content between properly matched opening/closing tags
        # The unclosed reasoning tag will match with the next closing tag
        assert "Good reasoning" in result.reasoning
        assert result.answer == ["Good answer"]
        # Check that we got some reasoning results (behavior depends on regex matching)
        assert len(result.reasoning) >= 1

    def test_parse_all_nested_tags(self, xml_parser):
        """Test parse_all with nested tags (should not match nested)."""
        xml_text = """
        <reasoning>
        Outer reasoning with <reasoning>nested reasoning</reasoning> inside
        </reasoning>
        <answer>Simple answer</answer>
        """
        result = xml_parser.parse_all(xml_text)
        # Due to non-greedy matching, this should work correctly
        assert len(result.reasoning) >= 1
        assert result.answer == ["Simple answer"]

    def test_parse_all_empty_xml_parser(self):
        """Test parse_all with XMLParser that has no fields."""
        empty_parser = XMLParser([])
        xml_text = "<reasoning>Should be ignored</reasoning><answer>Also ignored</answer>"
        result = empty_parser.parse_all(xml_text)
        # Should have no attributes since no fields defined
        assert not hasattr(result, 'reasoning')
        assert not hasattr(result, 'answer')
