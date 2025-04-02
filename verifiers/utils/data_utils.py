import random
import json
from typing import List, Dict, Callable, Any

from datasets import Dataset, load_dataset, concatenate_datasets # type: ignore

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def strip_non_numeric(text: str) -> str:
    return "".join(c for c in text if c.isdigit() or c == '.')

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_preprocess_fn(name: str) -> Callable[[Dict], Dict]: 
    if name == "aime2024":
        def preprocess_aime2024(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": str(int(x["answer"])),
                "task": "math"
            }
        return preprocess_aime2024
    elif name == "aime2025":
        def preprocess_aime2025(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": strip_non_numeric(x["answer"]),
                "task": "math"
            }
        return preprocess_aime2025
    elif name == "amc2023":
        def preprocess_amc2023(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
                "task": "math"
            }
        return preprocess_amc2023
    elif name in ["gpqa_diamond", "gpqa_main"]:
        def preprocess_gpqa(x: Dict[str, Any]) -> Dict[str, Any]:
            q = x["Question"]
            letters = ["A", "B", "C", "D"]
            random.shuffle(letters)
            itos = {k: v for k, v in enumerate(letters)} 
            ans = {
                itos[0]: x["Correct Answer"],
                itos[1]: x["Incorrect Answer 1"],
                itos[2]: x["Incorrect Answer 2"],
                itos[3]: x["Incorrect Answer 3"]
            }
            question = f"Question: {q}\n\n"
            question += f"A: {ans['A']}\n"
            question += f"B: {ans['B']}\n"
            question += f"C: {ans['C']}\n"
            question += f"D: {ans['D']}"

            return {
                "question": question, 
                "answer": itos[0],
                "task": "mc"
            }
        return preprocess_gpqa
    elif name == "gsm8k":
        def preprocess_gsm8k(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": extract_hash_answer(x["answer"]),
                "task": "math"
            }
        return preprocess_gsm8k
    elif name == "math":
        def preprocess_math(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": extract_boxed_answer(x["solution"]),
                "task": "math"
            }
        return preprocess_math
    elif name == "math500":
        def preprocess_math500(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
                "task": "math"
            }
        return preprocess_math500
    elif name == "mmlu":
        mmlu_map = ["A", "B", "C", "D"]
        def preprocess_mmlu(x: Dict[str, Any]) -> Dict[str, Any]:
            options = x["choices"]
            answer = x["answer"]
            question = f"Question: {x["question"]}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "temp_answer": mmlu_map[answer],
                "task": "mc"
            }
        return preprocess_mmlu
    elif name == "mmlu_pro":
        mmlu_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        def preprocess_mmlu(x: Dict[str, Any]) -> Dict[str, Any]:
            options = x["options"]
            answer = x["answer"]
            question = f"Question: {x["question"]}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "answer": answer,
                "task": "mc"
            }
        return preprocess_mmlu
    elif name == "openbookqa":
        def preprocess_openbookqa(x: Dict[str, Any]) -> Dict[str, Any]:
            choices_texts = x['choices']['text']
            choices_labels = x['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {x['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return {
                "question": question,
                "answer": x["answerKey"],
                "task": "mc"
            }
        return preprocess_openbookqa
    elif name in ["openrs", "openrs_easy", "openrs_hard"]:
        def preprocess_openrs(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
                "task": "math"
            }
        return preprocess_openrs
    elif name == "prime_code":
        def preprocess_prime_code(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["prompt"],
                "answer": x["verification_info"],
                "task": "code"
            }
        return preprocess_prime_code
    else:
        raise ValueError(f"Dataset {name} not supported for preprocess_dataset.")

def preprocess_dataset(name: str = "gsm8k",
                       split: str | None = None,
                       n: int | None = None,
                       seed: int = 0) -> Dataset:
    if name == "aime2024":
        if split is None:
            split = "train"
        dataset = load_dataset("HuggingFaceH4/aime_2024")[split] # type: ignore
    elif name == "aime2025":
        if split is None:
            split = "test"
        aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I")[split] # type: ignore
        aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II")[split] # type: ignore
        dataset = concatenate_datasets([aime_i, aime_ii]) # type: ignore
    elif name == "amc2023":
        if split is None:
            split = "train"
        dataset = load_dataset("knoveleng/AMC-23")[split] # type: ignore
    elif name == "gpqa_diamond":
        if split is None:
            split = "train"
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")[split] # type: ignore
    elif name == "gpqa_main":
        if split is None:
            split = "train"
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")[split] # type: ignore
    elif name == "gsm8k":
        if split is None:
            split = "test"
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
    elif name == "math":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
    elif name == "math500":
        if split is None:
            split = "test"
        dataset: Dataset = load_dataset("HuggingFaceH4/MATH-500")[split] # type: ignore
    elif name == "mmlu":
        if split is None:
            split = "dev"
        dataset = load_dataset("cais/mmlu", "all")[split] # type: ignore
    elif name == "mmlu_pro":
        if split is None:
            split = "validation"
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split] # type: ignore
    elif name == "openbookqa":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("allenai/openbookqa", "main")[split] # type: ignore
    elif name == "openrs":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("knoveleng/open-rs")[split] # type: ignore
    elif name == "openrs_easy":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("knoveleng/open-rs")[split] # type: ignore
        dataset = dataset.filter(lambda x: x["level"] == "Easy") # type: ignore
    elif name == "openrs_hard":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("knoveleng/open-rs")[split] # type: ignore
        dataset = dataset.filter(lambda x: x["level"] == "Hard") # type: ignore
    elif name == "prime_code":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("PrimeIntellect/verifiable-coding-problems")[split] # type: ignore
        dataset = dataset.filter(lambda x: x['prompt'].startswith("Solve the following coding problem using the programming language python:")) # type: ignore
    else:
        raise ValueError(f"Dataset {name} not supported for preprocess_dataset. \
Please ensure that the dataset is formatted with 'prompt' (str) and 'answer' (str) keys.")
    
    preprocess_fn = get_preprocess_fn(name)
    if n is not None and n > 0:
        dataset = dataset.shuffle(seed=seed).select(range(n)) # type: ignore
    dataset = dataset.map(preprocess_fn, num_proc=10, remove_columns=dataset.column_names) # type: ignore
    if "temp_answer" in dataset.column_names:
        dataset = dataset.rename_column("temp_answer", "answer")
    return dataset

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def format_dataset(dataset: Dataset,
                   system_prompt: str | None = None,
                   few_shot: List[Dict[str, str]] | None = None,
                   fewshot_prob: float = 1.0,
                   question_key: str = "question",
                   answer_key: str = "answer",
                   ) -> Dataset:
    return dataset.map(lambda x: {
        "prompt": format_prompt(x[question_key], system_prompt, few_shot, fewshot_prob),
        "answer": x[answer_key]
    }, num_proc=10)