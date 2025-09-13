from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class UtteranceValueEstimator(nn.Module):
    """
    Utterance-level value network that predicts the expected final return
    given a conversation prefix up to and including the assistant's utterance
    for a turn.

    Features are computed as masked mean of the frozen LM input embeddings.
    Only the small MLP head is trained.
    """

    def __init__(self, lm: nn.Module, hidden_size: int = 1024) -> None:
        super().__init__()
        # Use the LM input embeddings as a frozen text encoder
        self.input_embeddings: nn.Embedding = lm.get_input_embeddings()  # type: ignore
        embed_dim = self.input_embeddings.embedding_dim
        for p in self.input_embeddings.parameters():
            p.requires_grad = False

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked-mean embedding from frozen LM input embeddings."""
        emb = self.input_embeddings(input_ids)  # (B, L, D)
        mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
        summed = (emb * mask).sum(dim=1)  # (B, D)
        denom = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.value_head(feats).squeeze(-1)

    @staticmethod
    def batch_tokenize_messages(
        tokenizer: Any,
        batch_messages: List[List[Dict[str, Any]]],
        max_length: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of chat-message sequences into tensors."""
        texts: List[str] = []
        for msgs in batch_messages:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            assert isinstance(text, str)
            texts.append(text)
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

