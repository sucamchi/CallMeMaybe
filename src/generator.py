"""Builds the prompt, drives constrained decoding per prompt, and
assembles the results.

Generation follows a "skeleton + slot" approach: the fixed JSON
punctuation (braces, keys, quotes, commas) is never a real decision,
so it's written directly as plain text. Only two kinds of things are
actual decisions handed to the model: which function to call, and
each argument's value -- both go through constrained decoding.
"""

import json

from src.constraints import (
    GenerationContext, choose_from_candidates,
    generate_number_value, generate_string_value)
from src.models import FunctionDef, OutputResult


def build_prompt_text(functions: list[FunctionDef], prompt: str) -> str:
    """Build the text that will be fed to the model for one prompt."""
    lines = [
        "You translate a user request into exactly one function call.",
        "Pick the single function whose description best matches what",
        "the user is asking for.",
        "",
        "Example:",
        "Available functions:",
        "- get_weather(city: string): Get the current weather for a city.",
        "- send_email(to: string, subject: string): Send an email.",
        'User request: "What is the weather like in Paris?"',
        "JSON:",
        '{"name": "get_weather", "parameters": {"city": "Paris"}}',
        "",
        "Now do the same for this request.",
        "Available functions:",
    ]
    for function in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in function.parameters.items()
        )
        lines.append(f"- {function.name}({params}): {function.description}")
    lines.append(f'User request: "{prompt}"')
    lines.append("JSON:")
    return "\n".join(lines)


def choose_function(
        context: GenerationContext, prompt_ids: list[int],
        functions: list[FunctionDef]) -> FunctionDef:
    """Let the model score every name in the catalog and take the best."""
    by_name = {function.name: function for function in functions}
    candidates = {name: context.encode(name) for name in by_name}
    return by_name[choose_from_candidates(context, prompt_ids, candidates)]


def generate_value(
        context: GenerationContext, prompt_ids: list[int],
        param_type: str) -> float | str | bool:
    """Fill one argument slot, decoded as its declared type."""
    if param_type == "number":
        return generate_number_value(context, prompt_ids)
    if param_type == "boolean":
        candidates = {"true": context.encode("true"),
                      "false": context.encode("false")}
        choice = choose_from_candidates(context, prompt_ids, candidates)
        return choice == "true"
    return generate_string_value(context, prompt_ids)


def generate_result_for_prompt(
        context: GenerationContext, functions: list[FunctionDef],
        prompt: str) -> OutputResult:
    """Run the full skeleton-plus-slot generation for one prompt."""
    text = build_prompt_text(functions, prompt) + '{"name": "'
    function = choose_function(context, context.encode(text), functions)
    text += function.name + '", "parameters": {'

    parameters: dict[str, float | str | bool] = {}
    for index, (param_name, param) in enumerate(function.parameters.items()):
        if index > 0:
            text += ", "
        text += f'"{param_name}": '
        # Any type that is not a number or a boolean is decoded as a
        # string, exactly as generate_value does it. A string is given
        # its opening quote up front so the model can see it is inside
        # one before it writes the first character.
        opening_quote = "" if param.type in ("number", "boolean") else '"'

        # The running text is re-encoded every time instead of splicing
        # id fragments together: BPE token boundaries shift with what
        # precedes them, so splicing can build a sequence the model has
        # never seen.
        prompt_ids = context.encode(text + opening_quote)
        value = generate_value(context, prompt_ids, param.type)
        parameters[param_name] = value
        # Writing the value back as JSON re-adds the quotes around a
        # string and turns a bool into true/false.
        text += json.dumps(value, ensure_ascii=False)

    return OutputResult(
        prompt=prompt, name=function.name, parameters=parameters)
