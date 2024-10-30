import json
from typing import TypedDict

from tqdm import tqdm
from transformers import AutoTokenizer

from llm_tokenseer.embeddings_extractor.dataset.data_model import InstructData
from llm_tokenseer.embeddings_extractor.dataset.prompt_template import PAD_TOKEN, render_prompt


class ProcessedData(TypedDict):
    id: int
    input_ids: list[int]


def load_raw_data(dataset_path: str) -> list[InstructData]:
    """
    Loads the raw json data and creates objects of the InstructData
    to be used for training the model.
    """
    raw_data: list[InstructData] = []
    with open(dataset_path, "rb") as f:
        for line in f.readlines():
            raw_data_json: dict = json.loads(line)
            q_id = raw_data_json["id"]
            question = raw_data_json["instruction"]
            context = raw_data_json.get("input")
            answer = raw_data_json.get("output")
            num_tokens = raw_data_json.get("generation_length")

            raw_data.append(
                InstructData(
                    id=q_id,
                    context=context,
                    question=question,
                    answer=answer,
                    num_tokens=num_tokens,
                )
            )
    return raw_data


def process_raw_data(
    raw_data: list[InstructData], embedding_mode: str, max_length: int = 1024 * 16
) -> list[ProcessedData]:
    """
    Processes the raw loaded data. Encodes with the
    appropriate tokenizer and the processed objects are created
    """
    tokenizer = AutoTokenizer.from_pretrained(embedding_mode)
    tokenizer.pad_token = PAD_TOKEN
    processed_data: list[ProcessedData]
    processed_data = []
    for query in tqdm(raw_data, total=len(raw_data)):
        sample_encodings = tokenizer.encode_plus(
            text=render_prompt(query.question, query.context),
            truncation="only_first",
            max_length=max_length,
            add_special_tokens=False,
        )

        processed_data.append({"id": query.id, "input_ids": sample_encodings.data["input_ids"]})

    return processed_data


# print(
#     process_raw_data(
#         [InstructData(question="What is the capital of France?")], "meta-llama/Llama-3.1-8B-Instruct", max_length=2
#     )
# )
