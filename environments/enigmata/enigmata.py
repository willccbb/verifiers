import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset
from datasets import load_dataset
import json
import subprocess

import verifiers as vf


# Minimal defaults to satisfy environment construction. Adjust as needed by
# consumers of this module.
system_prompt = ""


def parser(x):
    return x


def _cwd(path: Path):
    """Context manager to temporarily change working directory."""

    class _Cwd:
        def __enter__(self):
            self._prev = os.getcwd()
            os.chdir(str(path))
            return self

        def __exit__(self, exc_type, exc, tb):
            os.chdir(self._prev)
            return False

    return _Cwd()


def normalize_problem(
    problem: Any,
    task_name: str,
    difficulty: str,
    split: str,
    language: str,
) -> Dict[str, Any]:
    """
    Normalize a single problem into the expected entity structure.

    - Returns a consistent schema to avoid Arrow type issues:
      {question: str, answer: str, task_name: str, difficulty: str,
       split: str, language: str, meta_json: Optional[str]}
    - `question` is mapped from `prompt` or `question`. If a non-string is
      provided, it is JSON-serialized.
    - `answer` is required; if a non-string is provided, it is JSON-serialized.
    - Original `meta` dict (if any) is serialized into `meta_json`.
    """

    if isinstance(problem, dict):
        obj: Dict[str, Any] = dict(problem)
    else:
        obj = {"data": problem}

    # Derive question
    question_value = obj.get("prompt")
    if question_value is None:
        question_value = obj.get("question")

    # Derive answer (required)
    answer_value = obj.get("answer")

    # Strict validation presence before coercion
    if question_value is None:
        raise ValueError("Normalized problem missing required field: question")
    if answer_value is None:
        raise ValueError("Normalized problem missing required field: answer")

    # Coerce to strings for Arrow consistency
    def to_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        # JSON-serialize complex types
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    question_text = to_text(question_value)
    answer_text = to_text(answer_value)

    # Prepare meta_json (original meta plus standard tags)
    meta_dict: Dict[str, Any] = {}
    if isinstance(obj.get("meta"), dict):
        try:
            meta_dict = dict(obj["meta"])  # copy
        except Exception:
            meta_dict = {}
    meta_dict.update(
        {
            "task_name": task_name,
            "difficulty": difficulty,
            "split": split,
            "language": language,
        }
    )
    try:
        meta_json: Optional[str] = json.dumps(meta_dict, ensure_ascii=False)
    except Exception:
        meta_json = None

    # Return consistent, flat record
    return {
        "question": question_text,
        "answer": answer_text,
        "task_name": task_name,
        "difficulty": difficulty,
        "split": split,
        "language": language,
        "meta_json": meta_json,
    }


def generate_dataset(
    difficulties: Optional[List[str]] = None,
    count: int = 100,
    split: str = "train",
    language: str = "en",
) -> Dataset:
    """
    Generate a HuggingFace Dataset by discovering task generators and
    producing examples across difficulties, mirroring `generate_all_tasks.py`.

    - Discovers tasks under `Enigmata/verifiable_tasks/tasks` that contain
      a `generator.py` with a `generate` callable.
    - Supports generator functions (with or without `count`) and regular
      functions that return one problem per invocation.
    - Attaches `task_name`, `difficulty`, `split`, `language` metadata.
    """

    selected_difficulties: List[str] = difficulties or ["easy", "medium", "hard"]
    problems_per_difficulty: int = count

    # Ensure `Enigmata` is on sys.path to import `verifiable_tasks.tasks.*`
    repo_root: Path = Path(__file__).parent
    enigmata_root: Path = repo_root / "Enigmata"
    if str(enigmata_root) not in sys.path:
        sys.path.append(str(enigmata_root))

    tasks_dir: Path = enigmata_root / "verifiable_tasks" / "tasks"

    # Reduce noise from progress bars used by some generators
    os.environ.setdefault("TQDM_DISABLE", "1")

    examples: List[Dict[str, Any]] = []
    if tasks_dir.exists() and tasks_dir.is_dir():
        for task_name in sorted(os.listdir(tasks_dir)):
            task_path = tasks_dir / task_name
            if not task_path.is_dir():
                continue
            if not (task_path / "generator.py").exists():
                continue

            module_path = f"verifiable_tasks.tasks.{task_name}.generator"
            try:
                # Some generators assume CWD is the repository root (`Enigmata`).
                with _cwd(enigmata_root):
                    generator_module = importlib.import_module(module_path)
            except Exception:
                # Skip tasks that fail to import
                continue

            if not hasattr(generator_module, "generate"):
                continue

            generate_func = getattr(generator_module, "generate")
            is_generator_func = inspect.isgeneratorfunction(generate_func)
            params = inspect.signature(generate_func).parameters

            for difficulty in selected_difficulties:
                collected: List[Any] = []
                try:
                    if is_generator_func:
                        if "count" in params:
                            with _cwd(enigmata_root):
                                gen = generate_func(
                                    problems_per_difficulty,
                                    difficulty=difficulty,
                                    language=language,
                                    split=split,
                                )
                            for i, problem in enumerate(gen):
                                collected.append(problem)
                                if i + 1 >= problems_per_difficulty:
                                    break
                        else:
                            for _ in range(problems_per_difficulty):
                                try:
                                    with _cwd(enigmata_root):
                                        gen = generate_func(
                                            difficulty=difficulty,
                                            language=language,
                                            split=split,
                                        )
                                        problem = next(gen)
                                    collected.append(problem)
                                except Exception:
                                    continue
                    else:
                        for _ in range(problems_per_difficulty):
                            try:
                                with _cwd(enigmata_root):
                                    problem = generate_func(
                                        difficulty=difficulty,
                                        language=language,
                                        split=split,
                                    )
                                collected.append(problem)
                            except Exception:
                                continue
                except Exception:
                    collected = []

                for problem in collected:
                    normalized = normalize_problem(
                        problem=problem,
                        task_name=task_name,
                        difficulty=difficulty,
                        split=split,
                        language=language,
                    )
                    examples.append(normalized)

    return Dataset.from_list(examples)


def _adapt_eval_dataset(external) -> Dataset:
    """
    Adapt a HF Dataset or DatasetDict (e.g., BytedTsinghua-SIA/Enigmata-Eval)
    to this environment's normalized schema by leveraging normalize_problem.

    - Merges all splits into a single Dataset
    - Uses fields: question, answer, task_name, difficulty, split, language, meta_json
    - Sets split="eval" for all adapted rows
    """
    normalized: List[Dict[str, Any]] = []

    def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        meta = row.get("meta") if isinstance(row, dict) else None
        difficulty = None
        language = None
        if isinstance(meta, dict):
            difficulty = meta.get("difficulty")
            language = meta.get("language")
        elif isinstance(meta, str):
            # Try to parse JSON-encoded meta to recover difficulty/language
            try:
                parsed_meta = json.loads(meta)
                if isinstance(parsed_meta, dict):
                    difficulty = parsed_meta.get("difficulty", difficulty)
                    language = parsed_meta.get("language", language)
                    # Use a shallow copy of row with dict meta so normalize_problem captures it
                    try:
                        row = dict(row)
                        row["meta"] = parsed_meta
                    except Exception:
                        pass
            except Exception:
                # Non-JSON string meta; ignore
                pass
        task_name = row.get("task_name") if isinstance(row, dict) else None
        # (Debug prints removed)
        return normalize_problem(
            problem=row,
            task_name=task_name or "unknown",
            difficulty=difficulty or "unknown",
            split="eval",
            language=language or "en",
        )

    # DatasetDict (mapping of splits) vs Dataset
    # Prefer explicit branch instead of catching exceptions that hide row-level errors
    if hasattr(external, "keys") and callable(getattr(external, "keys")):
        # Likely a DatasetDict
        for split_name in external.keys():
            split_ds = external[split_name]
            for row in split_ds:
                try:
                    normalized.append(normalize_row(row))
                except Exception:
                    # Skip rows that cannot be normalized
                    continue
    else:
        # Treat as a single Dataset
        for row in external:
            try:
                normalized.append(normalize_row(row))
            except Exception:
                continue

    return Dataset.from_list(normalized)


def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    use_predefined_eval_dataset: bool = False,
    system_prompt: str = "",
    tasks: Union[str, List[str]] = "all",
    **kwargs,
) -> vf.Environment:
    # The following block ensures that the 'Enigmata' repository, which contains
    # necessary components like task verifiers, is available locally. If it's not
    # found, it will be automatically cloned from its GitHub repository.

    # Define the expected local path for the Enigmata repository. It should be
    # a subdirectory named 'Enigmata' located in the same directory as this script.
    enigmata_root = Path(__file__).parent / "Enigmata"

    # Check if the Enigmata repository directory exists.
    if not enigmata_root.is_dir():
        print(
            "Local 'Enigmata' repository not found. Attempting to clone from GitHub..."
        )

        # The official URL for the Enigmata git repository.
        repo_url = "https://github.com/BytedTsinghua-SIA/Enigmata.git"

        try:
            # Execute the 'git clone' command using subprocess.
            # 'check=True' will raise a CalledProcessError if the command returns a non-zero exit code.
            # 'capture_output=True' and 'text=True' prevent git's output from printing to the console
            # unless an error occurs, in which case it's captured in stderr.
            subprocess.run(
                ["git", "clone", repo_url, str(enigmata_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Successfully cloned 'Enigmata' repository to: {enigmata_root}")
        except FileNotFoundError:
            # This exception is raised if the 'git' command is not found in the system's PATH.
            print("\nERROR: 'git' command not found.")
            print(
                "Please install Git and ensure it is accessible in your system's PATH to proceed."
            )
            # Re-raising the exception to halt execution as this is a critical dependency.
            raise
        except subprocess.CalledProcessError as e:
            # This exception is raised if 'git clone' fails for any other reason (e.g., network issues, permissions).
            print("\nERROR: Failed to clone the 'Enigmata' repository.")
            print(f"Git command failed with error:\n{e.stderr}")
            # Raise a new RuntimeError, chaining the original exception for better debugging.
            raise RuntimeError(
                "Could not clone the required 'Enigmata' repository."
            ) from e

    if use_predefined_eval_dataset:
        dataset = generate_dataset(
            count=num_train_examples,
        )
        raw_eval_dataset = load_dataset("BytedTsinghua-SIA/Enigmata-Eval")
        eval_dataset = _adapt_eval_dataset(raw_eval_dataset)
    else:
        dataset = generate_dataset(
            count=num_train_examples,
        )
        eval_dataset = generate_dataset(
            count=num_eval_examples,
        )

    # Optional task filtering: keep only specified tasks
    def _filter_by_tasks(ds: Dataset, selected_tasks: Union[str, List[str]]):
        if ds is None:
            return ds
        # If "all" or falsy, return as-is
        if not selected_tasks or selected_tasks == "all":
            return ds
        # Normalize to a set of names
        if isinstance(selected_tasks, str):
            names = {selected_tasks}
        else:
            names = set(selected_tasks)
        return ds.filter(lambda row: row.get("task_name") in names)

    dataset = _filter_by_tasks(dataset, tasks)
    eval_dataset = _filter_by_tasks(eval_dataset, tasks)

    # Ensure `Enigmata` is on sys.path so we can import task verifiers dynamically
    repo_root: Path = Path(__file__).parent
    enigmata_root: Path = repo_root / "Enigmata"
    if str(enigmata_root) not in sys.path:
        sys.path.append(str(enigmata_root))

    # Lightweight cache to avoid re-importing verifier modules repeatedly
    verify_fn_cache: Dict[str, Any] = {}

    def reward_func(prompt, parser, completion, answer, **kwargs):
        """
        Per-sample reward function.

        Mirrors the logic of `compute_score` in `Enigmata/test_eval.py` by:
        - Resolving the correct task verifier from `verifiable_tasks.tasks.<task>.verifier`
        - Passing through the model's raw completion, the ground-truth answer, and the task meta
        - Returning the verifier's numeric score (typically 0/1)

        Parameters
        - prompt: The current example's question as a string.
        - parser: A callable for post-processing the completion if needed (unused here).
        - completion: The model's output string for this example.
        - answer: The ground-truth answer string for this example.
        - **kwargs: Additional context keys such as `task_name`, `meta_json` (JSON
                    string), `meta` (dict/JSON string), or a nested `example`/`row`/`record`
                    dict carrying these fields.
        """

        # Defensive defaults
        task_name: str = "unknown"
        meta: Any = {}

        # Extract task_name/meta primarily from kwargs since prompt is a string
        try:
            # Direct keys
            task_name = kwargs.get("task_name", task_name)
            meta_json = kwargs.get("meta_json")
            meta_value = kwargs.get("meta")

            # Nested example/row/record containers if present
            container = (
                kwargs.get("example") or kwargs.get("row") or kwargs.get("record")
            )
            if isinstance(container, dict):
                task_name = container.get("task_name", task_name)
                if meta_json is None:
                    meta_json = container.get("meta_json")
                if meta_value is None:
                    meta_value = container.get("meta")

            # Decode meta, preferring meta_json if provided
            if isinstance(meta_json, str):
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    meta = {}
            elif isinstance(meta_value, dict):
                meta = dict(meta_value)
            elif isinstance(meta_value, str):
                try:
                    meta = json.loads(meta_value)
                except Exception:
                    meta = {}
        except Exception:
            task_name = task_name
            meta = {}

        # Resolve the verifier function for this task, with caching
        verify_fn = verify_fn_cache.get(task_name)
        if verify_fn is None:
            try:
                module = importlib.import_module(
                    f"verifiable_tasks.tasks.{task_name}.verifier"
                )
                candidate = getattr(module, "verify", None)
                if callable(candidate):
                    verify_fn = candidate
                    verify_fn_cache[task_name] = verify_fn
            except Exception:
                verify_fn = None

        # If we cannot resolve a verifier, return zero reward
        if verify_fn is None:
            return 0.0

        # Compute score using the task-specific verifier; mirror test_eval semantics
        try:
            # Keep completion as a raw string; verifiers expect string outputs
            solution_str = (
                completion if isinstance(completion, str) else str(completion)
            )
            score = verify_fn(solution_str, answer, meta)
            # Coerce to float for compatibility
            return float(score)
        except Exception:
            return 0.0

    # Single-function rubric with unit weight, using identity parser
    rubric = vf.Rubric(funcs=[reward_func], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
    return env
