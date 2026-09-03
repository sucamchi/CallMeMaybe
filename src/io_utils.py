"""Load the two input JSON files and write the output JSON file."""

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.schema import FunctionDef, Prompt, OutputResult


def _read_json_array(path: Path) -> list[object]:
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
    """Load the function catalog; any problem with it is fatal."""
    functions = []
    seen_names = set()
    for entry in _read_json_array(path):
        try:
            function = FunctionDef.model_validate(entry)
        except ValidationError as exc:
            raise ValueError(
                f"invalid function definition in {path}: {exc}") from exc
        # Two functions with one name would make the choice ambiguous.
        if function.name in seen_names:
            raise ValueError(
                f"duplicate function name in {path}: {function.name}")
        seen_names.add(function.name)
        functions.append(function)

    if not functions:
        raise ValueError(f"{path} contains no functions")
    return functions


def load_prompts(path: Path) -> list[Prompt]:
    """Load the prompts; malformed entries are skipped, not fatal."""
    prompts = []
    for index, entry in enumerate(_read_json_array(path)):
        try:
            prompts.append(Prompt.model_validate(entry))
        except ValidationError as exc:
            print(
                f"Warning: skipping malformed prompt at index {index} "
                f"in {path}: {exc}",
                file=sys.stderr)
    return prompts


def write_results(path: Path, results: list[OutputResult]) -> None:
    """Write the results as one JSON array, creating the output dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
