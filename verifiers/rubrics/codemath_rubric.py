from typing import List, Dict

from verifiers import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.rubrics import MathRubric


class CodeMathRubric(MathRubric):
    def __init__(self,
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],
                 parser: XMLParser = XMLParser(fields=["think", ("code", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["output"]),
                 **kwargs):
        super().__init__(funcs=funcs, weights=weights, parser=parser, **kwargs)
        self.env_parser = env_parser
        self.add_reward_func(self.code_execution_reward_func)

    def code_execution_reward_func(self,
                                   completion: List[Dict[str, str]],
                                   **kwargs) -> float:
        """Reward function that checks code execution success at each step."""
        def check_execution(completion: List[Dict[str, str]]) -> float:
            total_code_steps = 0
            successful_executions = 0
            
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'code') and parsed.code is not None:
                        total_code_steps += 1
                        # Look for the next user message (environment response)
                        if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                            env_response = completion[i + 1]['content']
                            parsed_response = self.env_parser.parse(env_response)
                            if hasattr(parsed_response, 'output') and parsed_response.output:
                                output = parsed_response.output
                                if len(output) > 0 and not output.startswith('Error:'):
                                    successful_executions += 1
            
            # Reward for proportion of successful executions if at least one attempt
            if total_code_steps == 0:
                return 0.0
            return float(successful_executions / total_code_steps)
        return check_execution(completion)
    

