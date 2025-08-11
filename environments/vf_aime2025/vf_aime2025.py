import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


def load_environment(system_prompt: str = BOXED_SYSTEM_PROMPT):
    eval_dataset = load_example_dataset("aime2025")
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

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
