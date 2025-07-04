"""Pytest configuration and fixtures for verifiers tests."""

import pytest
from verifiers.parsers import Parser, XMLParser, ThinkParser


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