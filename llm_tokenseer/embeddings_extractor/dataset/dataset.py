from torch.utils.data import Dataset

from llm_tokenseer.embeddings_extractor.dataset.prepare import ProcessedData


class TokenDataset(Dataset):
    def __init__(self, processed_data: list[ProcessedData]):
        self.processed_data = processed_data

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, item: int) -> ProcessedData:
        return self.processed_data[item]
