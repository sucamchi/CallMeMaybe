"""Parse CLI, load input files, run the LLM, and write output file."""

import argparse
import sys
from src import constraints, generator, utils
from src.models import FunctionDef, Prompt

FUNCDEF = "data/input/functions_definition.json"
INPUT = "data/input/function_calling_tests.json"
OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Parse the three file-path flags, all optional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default=FUNCDEF)
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the whole pipeline, from input files to output file."""
    args = parse_args()

    print(f"Loading functions from {args.functions_definition}")
    functions = utils.load_json_array(args.functions_definition, FunctionDef)
    print(f"Loaded {len(functions)} functions")

    print(f"Loading prompts from {args.input}")
    prompts = utils.load_json_array(args.input, Prompt)
    print(f"Loaded {len(prompts)} prompts")

    print("Loading the model")
    from llm_sdk import Small_LLM_Model
    model = Small_LLM_Model()

    print("Building vocabulary and constraints")
    context = constraints.build_generation_context(model)

    print("Generating one function call per prompt")
    results = []
    for index, record in enumerate(prompts, start=1):
        print(f"({index}/{len(prompts)}) {record.prompt!r}")
        result = generator.generate_result_for_prompt(
            context, functions, record.prompt)
        print(f"{result.name}({result.parameters})")
        results.append(result)

    utils.write_results(args.output, results)
    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Error: {e}")
