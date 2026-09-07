"""Builds the prompt, drives constrained decoding per prompt, and
assembles the results."""

import json
from src.constraints import (
    GenerationContext, choose_from_candidates, generate_integer_value,
    generate_number_value, generate_string_value)
from src.models import FunctionDef, OutputResult


def build_prompt_text(functions: list[FunctionDef], prompt: str) -> str:
    """Build the text that will be fed to the model for one prompt."""
    lines = [
        "Translate the user request into exactly one function call.",
        "",
        "Example:",
        "- get_weather(city: string): Get the weather for a city.",
        'Request: "What is the weather like in Paris?"',
        '{"name": "get_weather", "parameters": {"city": "Paris"}}',
        "",
        "Functions:",
    ]
    for function in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in function.parameters.items()
        )
        lines.append(f"- {function.name}({params}): {function.description}")
    lines.append(f'Request: "{prompt}"')
    return "\n".join(lines) + "\n"


def choose_function(
        context: GenerationContext, prompt_ids: list[int],
        functions: list[FunctionDef]) -> FunctionDef:
    """Let the model pick one name out of the catalog, and nothing else."""
    by_name = {function.name: function for function in functions}
    candidates = {name: context.encode(name) for name in by_name}
    return by_name[choose_from_candidates(context, prompt_ids, candidates)]


def generate_value(
        context: GenerationContext, text: str,
        param_type: str) -> bool | int | float | str:
    """Fill one argument slot, decoded as its declared type.

    The running text is re-encoded here rather than spliced together
    from token ids: BPE token boundaries shift with what precedes them,
    so splicing can build a sequence the model has never seen.
    """
    if param_type == "number":
        return generate_number_value(context, context.encode(text))
    if param_type == "integer":
        return generate_integer_value(context, context.encode(text))
    if param_type == "boolean":
        candidates = {"true": context.encode("true"),
                      "false": context.encode("false")}
        choice = choose_from_candidates(
            context, context.encode(text), candidates)
        return choice == "true"
    # Any other type is written as a string. Its opening quote goes in
    # before generation so the model can see it is inside a string
    # before it writes the first character.
    return generate_string_value(context, context.encode(text + '"'))


def generate_result_for_prompt(
        context: GenerationContext, functions: list[FunctionDef],
        prompt: str) -> OutputResult:
    """Run the full skeleton-plus-slot generation for one prompt."""
    text = build_prompt_text(functions, prompt) + '{"name": "'
    function = choose_function(context, context.encode(text), functions)
    text += function.name + '", "parameters": {'

    parameters: dict[str, bool | int | float | str] = {}
    for index, (param_name, param) in enumerate(function.parameters.items()):
        if index > 0:
            text += ", "
        text += f'"{param_name}": '
        value = generate_value(context, text, param.type)
        parameters[param_name] = value
        # Writing the value back as JSON re-adds the quotes around a
        # string, escapes what is inside it, and turns a bool into
        # true/false, so the model always reads back valid JSON.
        text += json.dumps(value, ensure_ascii=False)

    return OutputResult(
        prompt=prompt, name=function.name, parameters=parameters)
