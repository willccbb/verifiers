from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    description: str = ""

