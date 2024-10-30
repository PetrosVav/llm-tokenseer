import click

from llm_tokenseer.embeddings_extractor import entrypoints


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--config_path", type=str, default="config/emb_config.yml")
@click.option("--dataset_path", type=str, default="data/alpaca_data_cleaned.jsonl")
def extract(config_path: str, dataset_path: str) -> None:
    entrypoints.extract(config_path, dataset_path)
