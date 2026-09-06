"""Load the two input JSON files and write the output JSON file."""

import json
import os
import sys

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
    """Load the prompts; malformed entries are skipped with a warning."""
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = [result.model_dump() for result in results]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
