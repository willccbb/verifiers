import logging
from typing import Any, List, Dict

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
    
    def get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg['role'] == 'assistant']
    
    def get_final_answer(self, trajectory: List[Dict[str, str]] | str) -> str | None:
        if isinstance(trajectory, str):
            return self.parse(trajectory)
        else:
            return self.parse(trajectory[-1]["content"])