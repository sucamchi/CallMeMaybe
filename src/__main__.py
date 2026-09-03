"""Entry point: `python -m src [--functions_definition ...] [--input ...]
[--output ...]`."""

import argparse
import sys
from pathlib import Path
from typing import Any

from src import constraints, generator, io_utils

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the three file-path flags, all optional."""
    parser = argparse.ArgumentParser(
        prog="src", description="Turn prompts into structured function calls.")
    parser.add_argument(
        "--functions_definition", type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None, model: Any = None) -> None:
    """Run the whole pipeline; pass a fake model in to test without one."""
    args = parse_args(argv)
    functions = io_utils.load_function_definitions(args.functions_definition)
    prompts = io_utils.load_prompts(args.input)

    if model is None:
        # Imported here rather than at the top so that a bad input file
        # is reported instantly, not after torch has finished loading.
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
    context = constraints.build_generation_context(model)
    results = []
    for record in prompts:
        try:
            results.append(generator.generate_result_for_prompt(
                context, functions, record.prompt))
        except generator.GenerationError as exc:
            print(
                f"Warning: {exc}; using fallback for "
                f"prompt: {record.prompt!r}",
                file=sys.stderr)
            results.append(generator.fallback_result(functions, record.prompt))

    io_utils.write_results(args.output, results)


def main() -> None:
    """Handle exceptions and exits if failure."""
    try:
        run()
    except Exception as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
