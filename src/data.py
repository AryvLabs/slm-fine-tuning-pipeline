import json
from pathlib import Path

from datasets import Dataset


def load_raw_examples(path: str | Path, instruction_key: str, response_key: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Make sure the file exists and 'dataset.path' in config.yaml is correct."
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array at the top level of '{path}', got {type(raw).__name__}."
        )

    examples = []
    for i, item in enumerate(raw):
        if instruction_key not in item or response_key not in item:
            raise KeyError(
                f"Item {i} in '{path}' is missing key "
                f"'{instruction_key}' or '{response_key}'.\n"
                f"Check 'dataset.instruction_key' and 'dataset.response_key' in config.yaml."
            )
        examples.append(
            {"instruction": item[instruction_key], "response": item[response_key]}
        )

    print(f"Loaded {len(examples)} examples from '{path}'.")
    return examples


def to_chat_format(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "model", "content": example["response"]},
        ]
    }


def build_datasets(cfg: dict) -> tuple[Dataset, Dataset]:
    ds_cfg = cfg["dataset"]

    examples = load_raw_examples(
        path=ds_cfg["path"],
        instruction_key=ds_cfg["instruction_key"],
        response_key=ds_cfg["response_key"],
    )

    dataset = Dataset.from_list(examples).map(to_chat_format, remove_columns=["instruction", "response"])

    split = dataset.train_test_split(
        test_size=ds_cfg["test_size"],
        seed=ds_cfg["seed"],
    )
    train_ds, eval_ds = split["train"], split["test"]

    print(f"Train: {len(train_ds)} examples | Eval: {len(eval_ds)} examples")
    print("\nSample formatted example:")
    print(train_ds[0])

    return train_ds, eval_ds
