"""Read/write JSON files and build a vocabulary from the model's vocab.json."""

import json
import os
from typing import Any
from pydantic import BaseModel, ValidationError
from src.models import OutputResult


def load_json_array(path: str, model: type[BaseModel]) -> list[Any]:
    """Read a JSON array from a file and validate each entry as model."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    try:
        return [model.model_validate(entry) for entry in data]
    except ValidationError as exc:
        raise ValueError(f"invalid entry in {path}: {exc}") from exc


def write_results(path: str, results: list[OutputResult]) -> None:
    """Write the results as one JSON array to output."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = [result.model_dump() for result in results]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_vocabulary(model: Any) -> dict[int, str]:
    """Map every id in the model's vocab.json to the text it decodes to."""
    vocab_file_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_file_path, encoding="utf-8") as file:
            vocab = json.load(file)
    except Exception:
        raise ValueError(f"could not read {vocab_file_path}")

    # The file is a {text: id} dictionary, where ids == values. Its keys
    # are byte-substituted placeholders, not text a token really says,
    # which is why the text comes back from decode() instead.
    return {int(id): str(model.decode([int(id)])) for id in vocab.values()}
