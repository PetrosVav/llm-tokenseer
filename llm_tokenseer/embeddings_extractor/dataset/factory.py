from functools import partial
from typing import List

import torch
from torch.utils.data import DataLoader

from llm_tokenseer.embeddings_extractor.dataset.data_model import ModelInput
from llm_tokenseer.embeddings_extractor.dataset.dataset import TokenDataset
from llm_tokenseer.embeddings_extractor.dataset.prepare import ProcessedData
from llm_tokenseer.embeddings_extractor.dataset.prompt_template import PAD_TOKEN_ID


def pad_data(dataset: List[List[int]], pad_value: int) -> List[List[int]]:
    max_length = max(len(data) for data in dataset)
    padded_dataset = [list(data) + [pad_value] * (max_length - len(data)) for data in dataset]

    return padded_dataset


def process_batch(batch: List, pad_id: int) -> ModelInput:
    q_ids = [x["id"] for x in batch]
    pre_inputs_ids = [x["input_ids"] for x in batch]

    inputs_ids = torch.tensor(pad_data(pre_inputs_ids, pad_value=pad_id), dtype=torch.int64)

    attention_mask = ~(inputs_ids == pad_id)

    indices = attention_mask.count_nonzero(dim=-1) - 1  # .unsqueeze(0)

    return ModelInput(
        q_ids=q_ids,
        input_ids=inputs_ids,
        attention_mask=attention_mask,
        token_indices=indices,
    )


def create_dataset(processed_data: list[ProcessedData]) -> TokenDataset:
    return TokenDataset(processed_data)


def create_dataloader(
    dataset: TokenDataset, batch_size: int = 1, num_workers: int = 0, shuffle: bool = True
) -> DataLoader:
    process_batch_fn = partial(process_batch, pad_id=PAD_TOKEN_ID)
    return DataLoader(
        dataset,
        drop_last=False,
        shuffle=shuffle,
        collate_fn=process_batch_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
