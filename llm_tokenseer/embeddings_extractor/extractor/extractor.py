import json

import tensor_parallel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from llm_tokenseer.embeddings_extractor.config import ExtractorConfig
from llm_tokenseer.embeddings_extractor.dataset.data_model import ModelInput


class Extractor:
    def __init__(self, config: ExtractorConfig, loader: DataLoader):
        model = AutoModel.from_pretrained(config.embedding_mode, attn_implementation="eager")
        self.model = tensor_parallel.TensorParallelPreTrainedModel(model)
        self.loader = loader

    def extract(self) -> None:
        batch: ModelInput
        for _, batch in enumerate(tqdm(self.loader)):
            with torch.no_grad():
                batch = batch.to("cuda:0")
                inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

            hidden_states = outputs.hidden_states[-1].cpu()
            B, S, D = hidden_states.shape
            hidden_states = hidden_states[torch.arange(B), batch.token_indices.cpu()]

            for i in range(hidden_states.shape[0]):
                with open("embeddings.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": batch.q_ids[i],
                                "embedding": hidden_states[i].cpu().numpy().tolist(),
                            }
                        )
                    )
                    f.write("\n")
