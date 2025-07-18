from datasets import concatenate_datasets

from verifiers.envs.smola_tool_env import SmolaToolEnv
from verifiers.prompts.few_shots import CALCULATOR_SMOLA_FEW_SHOTS
from verifiers.prompts.system_prompts import MATH_SMOLA_PROMPT_TEMPLATE
from verifiers.utils.data_utils import load_example_dataset

try:
    from smolagents.default_tools import PythonInterpreterTool  # type: ignore

    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents to use SmolAgents tools.")


def load_environment(use_few_shot: bool = False, **kwargs):
    dataset = load_example_dataset("math", "train", n=6000)

    eval_aime24 = load_example_dataset("aime2024", n=30)
    eval_aime25 = load_example_dataset("aime2025", n=30)
    eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

    # Use SmolaAgents' PythonInterpreterTool as a replacement for the python tool
    python_tool = PythonInterpreterTool(authorized_imports=["math", "sympy", "numpy"])
    # Add our custom calculator tool
    calculator_tool = CalculatorTool()

    vf_env = SmolaToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
        few_shot=CALCULATOR_SMOLA_FEW_SHOTS if use_few_shot else [],
        tools=[python_tool, calculator_tool],
        max_steps=5,
    )

    return vf_env
