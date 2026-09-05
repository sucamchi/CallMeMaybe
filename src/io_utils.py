"""Load the two input JSON files and write the output JSON file."""

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDef, Prompt, OutputResult


def read_json_array(path: Path) -> list[object]:
    """Read a file and parse it as a JSON array."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_function_definitions(path: Path) -> list[FunctionDef]:
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


def load_prompts(path: Path) -> list[Prompt]:
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


def write_results(path: Path, results: list[OutputResult]) -> None:
    """Write the results as one JSON array to output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
