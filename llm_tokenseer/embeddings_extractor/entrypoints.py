from pathlib import Path

from llm_tokenseer.embeddings_extractor.config import load_config
from llm_tokenseer.embeddings_extractor.dataset.factory import create_dataloader, create_dataset
from llm_tokenseer.embeddings_extractor.dataset.prepare import load_raw_data, process_raw_data
from llm_tokenseer.embeddings_extractor.dataset.utils import load_processed_data, save_processed_data
from llm_tokenseer.embeddings_extractor.extractor.extractor import Extractor


def extract(config_path: str, dataset_path: str) -> None:
    """
    The entrypoint to the training components.

    :param config_path: the path to the configuration file
    :param dataset_path: the parent dataset path
    """
    config = load_config(config_path)
    dataset_bin = Path(dataset_path).with_suffix(".bin")
    if dataset_bin.exists():
        processed_data = load_processed_data(str(dataset_bin))
    else:
        raw_data = load_raw_data(dataset_path=dataset_path)
        processed_data = process_raw_data(raw_data, config.embedding_mode, config.max_length)

        save_processed_data(str(dataset_bin), processed_data)

    train_dataset = create_dataset(processed_data)
    train_dataloader = create_dataloader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False
    )

    extractor = Extractor(config, train_dataloader)

    extractor.extract()
