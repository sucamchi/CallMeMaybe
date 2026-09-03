"""Pydantic models for the functions, the prompts, and the results."""

from pydantic import BaseModel, Field


class FunctionParam(BaseModel):
    """One named argument of a function, with its expected JSON type."""
    type: str


class FunctionDef(BaseModel):
    """One function the model is allowed to call."""
    name: str
    description: str
    parameters: dict[str, FunctionParam] = Field(default_factory=dict)


class Prompt(BaseModel):
    """One natural-language request to turn into a function call."""
    prompt: str


class OutputResult(BaseModel):
    """One output entry: the chosen function and its typed arguments."""
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
