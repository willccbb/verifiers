import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_environment(num_train_examples=-1, num_eval_examples=-1):
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    system_prompt = """
    Think step-by-step inside <think>...</think> tags.

    Then, give your final numerical answer inside \\boxed{{...}}.
    """
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
