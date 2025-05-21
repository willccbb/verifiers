import subprocess
from typing import List, Dict, Any, Tuple

from datasets import Dataset
from openai import OpenAI

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_PROMPT
from verifiers.rubrics import CodeRubric

class CodeEnv(MultiTurnEnv):
    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str = CODE_PROMPT,
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = 32,
                 max_steps: int = 5,
                 **kwargs):
        parser = XMLParser(fields=["think", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])
        rubric = CodeRubric(parser=parser, env_parser=self.env_parser)
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            parser=parser,
            rubric=rubric,
            max_steps=max_steps,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            **kwargs
        )
    
    def is_completed(self,
                    messages: List[Dict[str, str]],
                    state: Dict[str, Any],
                    **kwargs: Any) -> bool:
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def run_code(self, code: str, **kwargs: Any) -> str:
        try:
            # Run the code block in subprocess with 10-second timeout
            result = subprocess.run(
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 10 seconds"

    def env_response(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            if hasattr(parsed, 'code') and parsed.code is not None:
                output = self.run_code(parsed.code)
                if len(output.strip()) > 0:
                    env_response = {"role": "user", "content": self.env_parser.format(output=output)}
                    return env_response, state
                else:
                    env_response = {"role": "user", "content": "Error: Code execution returned empty output."}
                    return env_response, state
        except Exception:
            pass
        env_response = {"role": "user", "content": "Error: Code not found or invalid XML format. Please ensure correct formatting."}
        return env_response, state