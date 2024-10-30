import yaml
from pydantic import BaseModel, Field


class ExtractorConfig(BaseModel):
    batch_size: int = Field(default=1, description="The batch size for the dataloader")
    num_workers: int = Field(default=0, description="The number of workers for the dataloader")

    embedding_mode: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", description="Model name from huggingface")
    max_length: int = Field(default=1024 * 16, description="The maximum length of the input sequence")


def load_config(config_path: str) -> ExtractorConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return ExtractorConfig(**cfg)
