import subprocess
from typing import List, Dict, Any, Tuple

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.prompts import CODE_PROMPT
from verifiers.rubrics.codemath_rubric import CodeMathRubric
from verifiers.types import Message, Messages, State

class CodeMathEnv(MultiTurnEnv):
    def __init__(self,
                 system_prompt: str = CODE_PROMPT,
                 max_turns: int = 5,
                 **kwargs):
        parser = XMLParser(fields=["think", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])
        rubric = CodeMathRubric(parser=parser, env_parser=self.env_parser)
        super().__init__(
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs
        )

    def is_completed(self,
                    messages: Messages,
                    state: State,
                    **kwargs: Any) -> bool:
        try:
            assert isinstance(messages, list)
            parsed = self.parser.parse(messages[-1]['content'])
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def run_code(self, code: str, **kwargs: Any) -> str:
        try:
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
                     messages: Messages,
                     state: State,
                     **kwargs: Any) -> Tuple[Message, State]:
        try:
            assert isinstance(messages, list)
            parsed = self.parser.parse(messages[-1]['content'])
            if hasattr(parsed, 'code') and parsed.code is not None:
                output = self.run_code(parsed.code)
                if len(output.strip()) > 0:
                    env_response: Message = {"role": "user", "content": self.env_parser.format(output=output)} # type: ignore
                    return env_response, state 
                else:
                    env_response: Message = {"role": "user", "content": "Error: Code execution returned empty output."} # type: ignore
                    return env_response, state
        except Exception:
            pass
        env_response: Message = {"role": "user", "content": "Error: Code not found or invalid XML format. Please ensure correct formatting."} # type: ignore
        return env_response, state 