from math_verify import parse, verify

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


def load_environment(system_prompt: str = BOXED_SYSTEM_PROMPT):
    eval_dataset = load_example_dataset("aime2024")
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(parser, completion, answer) -> float:
        completion_answer = parser.parse_answer(completion)
        parsed_completion_answer = parse(completion_answer, parsing_timeout=0)
        parsed_ground_truth_answer = parse(answer, parsing_timeout=0)
        if verify(
            parsed_completion_answer, parsed_ground_truth_answer, timeout_seconds=0
        ):
            return 1.0
        else:
            return 0.0

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
