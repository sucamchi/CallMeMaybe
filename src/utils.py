"""Read/write JSON files and build a vocabulary from the model's vocab.json."""

import json
import os
import sys
from typing import Any
from pydantic import ValidationError
from src.models import FunctionDef, Prompt, OutputResult


def read_json_array(path: str) -> list[object]:
    """Read a file and parse it as a JSON array."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_function_definitions(path: str) -> list[FunctionDef]:
    """Load the function catalog with error handling."""
    functions = []
    for entry in read_json_array(path):
        try:
            functions.append(FunctionDef.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(
                f"invalid function definition in {path}: {exc}") from exc

    if not functions:
        raise ValueError(f"{path} contains no functions")
    return functions


def load_prompts(path: str) -> list[Prompt]:
    """Load the prompts, skipping malformed entries with a warning."""
    prompts = []
    for index, entry in enumerate(read_json_array(path)):
        try:
            prompts.append(Prompt.model_validate(entry))
        except ValidationError as exc:
            print(
                f"Warning: skipping invalid prompt entry {index} in {path}: "
                f"{exc}", file=sys.stderr)
    return prompts


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

    # The file is a {text: id} map, so the ids are its values. Its keys
    # are byte-substituted placeholders, not text a token really says,
    # which is why the text comes back from decode() instead.
    return {int(id): str(model.decode([int(id)])) for id in vocab.values()}
