import logging
from typing import Any, List, Dict, Callable

class Parser:
    """
    Parser class for parsing LLM rollouts.

    Default behavior:
    - `parse` returns text as-is
    - `get_final_answer` returns the last message's content (or text if string)
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse(self, text: str) -> Any:
        return text
    
    def get_assistant_messages(self, completion: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'assistant']
    
    def parse_answer(self, completion: List[Dict[str, str]] | str) -> str | None:
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            return self.parse(completion[-1]["content"])
 
    def get_format_reward_func(self) -> Callable:
        """
        Reward function that checks if the final answer is formatted correctly.
        """
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0
        return format_reward_func