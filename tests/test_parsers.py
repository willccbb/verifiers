import importlib.util
import pathlib
import types
import sys

parser_spec = importlib.util.spec_from_file_location(
    "parser",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/parsers/parser.py"),
)
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
Parser = parser_module.Parser

parsers_pkg = types.ModuleType('parsers')
parsers_pkg.parser = parser_module
parsers_pkg.Parser = parser_module.Parser
sys.modules['verifiers.parsers'] = parsers_pkg
sys.modules['verifiers.parsers.parser'] = parser_module

xml_spec = importlib.util.spec_from_file_location(
    "xml_parser",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/parsers/xml_parser.py"),
)
xml_module = importlib.util.module_from_spec(xml_spec)
xml_spec.loader.exec_module(xml_module)
XMLParser = xml_module.XMLParser


def test_parser_basic():
    parser = Parser()
    assert parser.parse('text') == 'text'
    messages = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'done'}]
    assert parser.parse_answer(messages) == 'done'


def test_xmlparser_parse_and_answer():
    parser = XMLParser(['reasoning', ('code', 'answer')])
    xml = '<reasoning>think</reasoning><code>print(1)</code>'
    parsed = parser.parse(xml)
    assert parsed.reasoning == 'think'
    assert parsed.code == 'print(1)'
    assert parsed.answer is None
    completion = [
        {'role': 'assistant', 'content': '<code>print(2)</code>'},
        {'role': 'assistant', 'content': '<answer>4</answer>'},
    ]
    assert parser.parse_answer(completion) == '4'


def test_xmlparser_format_and_format_str():
    parser = XMLParser(['reasoning', 'answer'])
    formatted = parser.format(reasoning='foo', answer='bar')
    assert '<reasoning>' in formatted and '</answer>' in formatted
    fmt = parser.get_format_str()
    assert '<reasoning>' in fmt and '<answer>' in fmt


def test_xmlparser_format_reward_func():
    parser = XMLParser(['reasoning', 'answer'])
    reward_func = parser.get_format_reward_func()
    good = [{'role': 'assistant', 'content': '<reasoning>hi</reasoning>'}]
    bad = [{'role': 'assistant', 'content': 'hi'}]
    assert reward_func(good) > reward_func(bad)
