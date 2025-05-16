from abc import ABC, abstractmethod
from typing import Any, List, Dict


class Parser(ABC):
    @abstractmethod
    def parse(self, text: str) -> Any:
        pass
    
    @abstractmethod
    def parse_answer(self, trajectory: List[Dict[str, str]]) -> str | None:
        pass