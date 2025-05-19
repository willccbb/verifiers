from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class CodeRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("code", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["output"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func()
        ]

    def code_execution_reward_func(self,
                                   completions: List[List[Dict[str, str]]],
                                   **kwargs) -> List[float]:
        """Reward function that checks code execution success at each step."""
        def check_execution(trajectory: List[Dict[str, str]]) -> float:
            total_code_steps = 0
            successful_executions = 0
            
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'code') and parsed.code is not None:
                        total_code_steps += 1
                        # Look for the next user message (environment response)
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            env_response = trajectory[i + 1]['content']
                            parsed_response = self.env_parser.parse(env_response)
                            if hasattr(parsed_response, 'output') and parsed_response.output:
                                output = parsed_response.output
                                if len(output) > 0 and not output.startswith('Error:'):
                                    successful_executions += 1
            
            # Return proportional reward based on successful executions
            if total_code_steps == 0:
                return 0.0
            return 0.3 * (successful_executions / total_code_steps) + 0.05 * (successful_executions)
        
        return [check_execution(c) for c in completions]
    

