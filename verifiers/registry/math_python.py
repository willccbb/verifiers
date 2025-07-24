from math_verify import parse, verify

import verifiers as vf
from verifiers.tools import python
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    **kwargs,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    system_prompt = """\
In every turn, think step by step inside <think>...</think>.
Give your final answer inside \\boxed{{}}."""

    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(parser, completion, answer) -> float:
        completion_answer = parser.parse_answer(completion)
        if verify(parse(completion_answer), parse(answer)):
            return 1.0
        else:
            return 0.0

    def num_turns(parser, completion) -> float:
        num_assistant_messages = len(parser.get_assistant_messages(completion))
        return float(num_assistant_messages)

    def num_tool_calls(parser, completion) -> float:
        num_tool_calls = len(parser.get_tool_messages(completion))
        return float(num_tool_calls)

    def num_errors(parser, completion) -> float:
        num_errors = sum(
            [
                1.0
                for msg in parser.get_tool_messages(completion)
                if "error" in msg["content"].lower()
            ]
        )
        return num_errors

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, num_turns, num_tool_calls, num_errors],
        weights=[1.0, 0.0, 0.0, -0.1],
        parser=parser,
    )

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=3,
        **kwargs,
    )

    return vf_env
