import subprocess
from typing import List, Dict, Any

from datasets import Dataset

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_FEW_SHOT, CODE_PROMPT
from verifiers.rubrics import CodeRubric
from verifiers.utils import preprocess_dataset

class CodeEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str = CODE_PROMPT,
                 few_shot: List[Dict[str, str]] = CODE_FEW_SHOT[0],
                 sampling_args: Dict[str, Any] = {
                     "stop": ["</code>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True, 
                 max_steps: int = 5, **kwargs):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.llm_parser = XMLParser(fields=["reasoning", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])
        self.rubric = CodeRubric(parser=self.llm_parser, env_parser=self.env_parser)
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
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

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            if hasattr(parsed, 'code') and parsed.code is not None:
                output = self.run_code(parsed.code)
                if len(output.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(output=output)}
                else:
                    return {"role": "user", "content": "Error: Code execution returned empty output."}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Code not found or invalid XML format. Please ensure correct formatting."}