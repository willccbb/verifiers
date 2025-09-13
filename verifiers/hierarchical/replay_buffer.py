from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


class UtteranceReplayBuffer:
    """
    Simple replay buffer for utterance-level transitions.

    Stores (messages, return) pairs where `messages` is a list of chat
    message dicts representing the conversation prefix (including the
    assistant utterance for the turn), and `return_` is the scalar target
    (e.g., Monte Carlo return or discounted return-to-go).
    """

    def __init__(self, capacity: int = 50000) -> None:
        self.capacity = int(capacity)
        self.storage: List[Tuple[List[Dict[str, Any]], float]] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, messages: List[Dict[str, Any]], return_: float) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append((messages, float(return_)))
        else:
            self.storage[self._next] = (messages, float(return_))
            self._next = (self._next + 1) % self.capacity

    def add_many(self, items: List[Tuple[List[Dict[str, Any]], float]]) -> None:
        for messages, ret in items:
            self.add(messages, ret)

    def sample(self, batch_size: int) -> List[Tuple[List[Dict[str, Any]], float]]:
        batch_size = min(batch_size, len(self.storage))
        return random.sample(self.storage, batch_size) if batch_size > 0 else []

