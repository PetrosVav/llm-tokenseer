from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelInput:
    q_ids: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_indices: torch.Tensor

    def to(self, device: str) -> "ModelInput":
        return ModelInput(
            q_ids=self.q_ids,
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_indices=self.token_indices.to(device),
        )


@dataclass
class InstructData:
    id: int
    question: str
    context: Optional[str] = None
    answer: Optional[str] = None
    num_tokens: Optional[int] = None
