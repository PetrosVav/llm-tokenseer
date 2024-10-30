import os
import pickle

from llm_tokenseer.embeddings_extractor.dataset.prepare import ProcessedData


def save_processed_data(processed_data_path: str, processed_data: list[ProcessedData]) -> None:
    # make sure parent folder exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    with open(processed_data_path, "wb") as file:
        pickle.dump(processed_data, file)


def load_processed_data(processed_data_path: str) -> list[ProcessedData]:
    with open(processed_data_path, "rb") as file:
        return pickle.load(file)
