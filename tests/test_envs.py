import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

# Load parser module
parser_spec = importlib.util.spec_from_file_location(
    "parser", str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/parsers/parser.py")
)
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
Parser = parser_module.Parser

# Setup fake verifiers package before loading environment and rubric modules
parsers_pkg = types.ModuleType('parsers')
parsers_pkg.parser = parser_module
parsers_pkg.Parser = Parser
sys.modules['verifiers.parsers'] = parsers_pkg

fake_verifiers = types.ModuleType('verifiers')
fake_verifiers.RewardFunc = lambda *a, **k: 0.0
fake_verifiers.parsers = parsers_pkg
fake_verifiers.rubrics = types.ModuleType('rubrics')
fake_verifiers.rubrics.Rubric = None  # placeholder
sys.modules['verifiers.rubrics'] = fake_verifiers.rubrics
sys.modules['verifiers'] = fake_verifiers

# Load rubric module
rubric_spec = importlib.util.spec_from_file_location(
    "rubric", str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/rubrics/rubric.py")
)
rubric_module = importlib.util.module_from_spec(rubric_spec)
rubric_spec.loader.exec_module(rubric_module)
Rubric = rubric_module.Rubric
fake_verifiers.rubrics.rubric = rubric_module
fake_verifiers.rubrics.Rubric = Rubric
sys.modules['verifiers.rubrics.rubric'] = rubric_module

# Load environment modules
env_spec = importlib.util.spec_from_file_location(
    "environment", str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/envs/environment.py")
)
env_module = importlib.util.module_from_spec(env_spec)
env_spec.loader.exec_module(env_module)
envs_pkg = types.ModuleType('envs')
envs_pkg.environment = env_module
sys.modules['verifiers.envs'] = envs_pkg
sys.modules['verifiers.envs.environment'] = env_module
fake_verifiers.envs = envs_pkg

singleturn_spec = importlib.util.spec_from_file_location(
    "singleturn_env", str(pathlib.Path(__file__).resolve().parents[1] / "verifiers/envs/singleturn_env.py")
)
singleturn_module = importlib.util.module_from_spec(singleturn_spec)
singleturn_spec.loader.exec_module(singleturn_module)
SingleTurnEnv = singleturn_module.SingleTurnEnv


class DummyDataset:
    def __init__(self, data):
        self.data = list(data)
        self.column_names = list(self.data[0].keys()) if self.data else []

    def map(self, func, num_proc=None):
        return DummyDataset([func(x) for x in self.data])

    def select(self, indices):
        return DummyDataset([self.data[i] for i in indices])

    def shuffle(self, seed=0):
        return self

    def __getitem__(self, key):
        return [row[key] for row in self.data]


class DummyOpenAI:
    def __init__(self, text, base_url="http://localhost"):
        self.base_url = base_url
        chat_resp = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[chat_resp])))
        comp_resp = SimpleNamespace(text=text, finish_reason="stop")
        self.completions = SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[comp_resp]))


def test_format_prompt_and_dataset():
    ds = DummyDataset([{"question": "hi", "answer": "bye"}])
    env = SingleTurnEnv(dataset=ds, parser=Parser(), rubric=Rubric())
    item = env.dataset.data[0]
    prompt = item["prompt"]
    assert prompt[-1]["content"] == "hi"


def test_get_model_response_chat():
    ds = DummyDataset([{"question": "hi", "answer": "bye"}])
    env = SingleTurnEnv(dataset=ds)
    client = DummyOpenAI("hello")
    response = env.get_model_response(
        [{"role": "user", "content": "hi"}],
        client=client,
        model="gpt",
        message_type="chat",
    )
    assert response == "hello"


def test_singleturn_rollout_chat():
    ds = DummyDataset([{"question": "hi", "answer": "bye"}])
    env = SingleTurnEnv(dataset=ds)
    client = DummyOpenAI("completion")
    completion, state = env.rollout(
        client,
        "gpt",
        [{"role": "user", "content": "hi"}],
        "bye",
    )
    assert completion[0]["content"] == "completion" and state == {}
