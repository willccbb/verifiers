import importlib.util
import pathlib
import sys
import types

parser_spec = importlib.util.spec_from_file_location(
    "parser",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/parsers/parser.py"),
)
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
xml_spec = importlib.util.spec_from_file_location(
    "xml_parser",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/parsers/xml_parser.py"),
)
xml_module = importlib.util.module_from_spec(xml_spec)
xml_spec.loader.exec_module(xml_module)
parsers_pkg = types.ModuleType('parsers')
parsers_pkg.parser = parser_module
parsers_pkg.Parser = parser_module.Parser
parsers_pkg.xml_parser = xml_module
parsers_pkg.XMLParser = xml_module.XMLParser
sys.modules['verifiers.parsers'] = parsers_pkg
sys.modules['verifiers.parsers.parser'] = parser_module
sys.modules['verifiers.parsers.xml_parser'] = xml_module
fake_verifiers = types.ModuleType('verifiers')
fake_verifiers.RewardFunc = lambda *a, **k: 0.0
fake_verifiers.parsers = parsers_pkg
fake_verifiers.rubrics = types.ModuleType('rubrics')
fake_verifiers.rubrics.__path__ = []
sys.modules['verifiers.rubrics'] = fake_verifiers.rubrics
sys.modules['verifiers'] = fake_verifiers

rubric_spec = importlib.util.spec_from_file_location(
    "rubric",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/rubrics/rubric.py"),
)
rubric_module = importlib.util.module_from_spec(rubric_spec)
rubric_spec.loader.exec_module(rubric_module)
Rubric = rubric_module.Rubric
fake_verifiers.rubrics.rubric = rubric_module
sys.modules['verifiers.rubrics.rubric'] = rubric_module

rgroup_spec = importlib.util.spec_from_file_location(
    "rubric_group",
    str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/rubrics/rubric_group.py"),
)
rgroup_module = importlib.util.module_from_spec(rgroup_spec)
rgroup_spec.loader.exec_module(rgroup_module)
RubricGroup = rgroup_module.RubricGroup
fake_verifiers.rubrics.rubric_group = rgroup_module
sys.modules['verifiers.rubrics.rubric_group'] = rgroup_module

sys.modules['verifiers.rubrics.rubric'] = rubric_module
sys.modules['verifiers.rubrics.rubric_group'] = rgroup_module


def simple_func(prompt, completion, answer, **kw):
    return 1.0 if completion == answer else 0.0


def length_func(completion, **kw):
    return float(len(str(completion)))


def test_rubric_scoring():
    rubric = Rubric(funcs=[simple_func, length_func], weights=[1.0, 0.1])
    scores = rubric.score_rollouts(
        prompts=['p'], completions=['ans'], answers=['ans'],
        states=[{}], tasks=[None]
    )
    assert scores['simple_func'][0] == 1.0
    assert scores['length_func'][0] == len('ans')
    assert scores['reward'][0] == 1.0 + len('ans') * 0.1


def test_rubric_group():
    r1 = Rubric(funcs=[simple_func], weights=[1.0])
    r2 = Rubric(funcs=[length_func], weights=[0.2])
    group = RubricGroup([r1, r2])
    scores = group.score_rollouts(
        prompts=['p'], completions=['xy'], answers=['xy'],
        states=[{}], tasks=[None]
    )
    assert scores['simple_func'][0] == 1.0
    assert scores['length_func'][0] == len('xy')
    assert scores['reward'][0] == 1.0 + len('xy') * 0.2
