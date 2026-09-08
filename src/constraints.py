"""Constrained decoding: at each step, mask out any token that would break
JSON validity or the expected value type, and only pick from what remains."""

import json
from typing import Any, Callable
import numpy as np
from pydantic import BaseModel, ConfigDict
from src.utils import build_vocabulary

NUMBER_CHARS = set("0123456789.eE+-")
MAX_NUMBER_TOKENS = 20
MAX_STRING_TOKENS = 40


def is_number_prefix(text: str) -> bool:
    """True if text is a JSON number or could still grow into one.
    Example of a valid prefix: "-3.14e+0".
    Example of an invalid prefix: "1.2.3"."""
    if text in ("", "-"):
        return True

    # Part 1: the integer digits, after an optional minus sign.
    index = 1 if text[0] == "-" else 0
    integer_start = index
    while index < len(text) and text[index].isdigit():
        index += 1
    digit_count = index - integer_start
    if digit_count == 0:
        return False
    if digit_count > 1 and text[integer_start] == "0":
        return False  # JSON forbids leading zeros, e.g. "01"
    if index == len(text):
        return True

    # Part 2: the fraction, if a dot comes next.
    if text[index] == ".":
        index += 1
        digits_after_dot = 0
        while index < len(text) and text[index].isdigit():
            index += 1
            digits_after_dot += 1
        if index == len(text):
            return True  # a dot with no digits yet may still get some
        if digits_after_dot == 0:
            return False  # but not if something else already followed it

    # Part 3: the exponent, if an e comes next.
    if index < len(text) and text[index] in "eE":
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        while index < len(text) and text[index].isdigit():
            index += 1

    # Stopping early means a character no JSON number can hold.
    return index == len(text)


def is_integer_prefix(text: str) -> bool:
    """True if text is a whole number or could still grow into one.

    Same job as is_number_prefix, minus the fraction and the exponent:
    a parameter declared "integer" has to come out as 4, never as 4.0.
    """
    if text in ("", "-"):
        return True
    digits = text[1:] if text[0] == "-" else text
    if not digits.isdigit():
        return False
    return len(digits) == 1 or digits[0] != "0"  # JSON forbids "01"


def is_number_token(text: str) -> bool:
    """True if text uses only characters a JSON number can contain."""
    return all(char in NUMBER_CHARS for char in text)


def is_string_token(text: str) -> bool:
    """True if a token may be picked while a JSON string value is open.

    A token carrying the closing quote, like ')",', counts as allowed:
    that is how the model says the value is finished, and only the part
    before the quote is kept. Forbidding them instead would throw away
    the characters sitting in front of the quote.

    Characters below 0x20 are illegal raw inside a JSON string, so a
    token holding one before its quote is never allowed.
    """
    body = text.partition('"')[0]
    return all(ord(char) >= 0x20 for char in body)


def build_char_class_mask(
        vocabulary: dict[int, str], vocab_size: int,
        is_allowed: Callable[[str], bool]) -> np.ndarray:
    """Precompute which token ids decode to a given char class."""
    mask = np.zeros(vocab_size, dtype=bool)
    for token_id, text in vocabulary.items():
        if 0 <= token_id < vocab_size and text and is_allowed(text):
            mask[token_id] = True
    return mask


class GenerationContext(BaseModel):
    """The model plus everything precomputed from it, built once per run."""

    # Pydantic builds a validator per field when the class is defined,
    # and it has no rules for a numpy array, so without this setting the
    # class statement itself raises. It falls back to a plain isinstance
    # check for the two masks. The other fields validate as usual.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    vocabulary: dict[int, str]
    number_mask: np.ndarray
    string_mask: np.ndarray

    def encode(self, text: str) -> list[int]:
        """Encode text to a plain list of token ids."""
        return [int(token_id) for token_id in self.model.encode(text)[0]]

    def logits(self, token_ids: list[int]) -> np.ndarray:
        """One score per possible next token, given the ids so far."""
        return np.array(self.model.get_logits_from_input_ids(token_ids))

    def decode(self, token_id: int) -> str:
        """What a token id says, or "" for an id we could not decode."""
        return self.vocabulary.get(token_id, "")


def build_generation_context(model: Any) -> GenerationContext:
    """Load the vocabulary and precompute the masks, once per run."""
    vocabulary = build_vocabulary(model)
    # The mask has to line up with a logits row, and the model's output
    # layer can be wider than the vocab file, so ask the model itself.
    probe_ids = [int(token_id) for token_id in model.encode(" ")[0]]
    vocab_size = len(model.get_logits_from_input_ids(probe_ids))
    return GenerationContext(
        model=model,
        vocabulary=vocabulary,
        number_mask=build_char_class_mask(
            vocabulary, vocab_size, is_number_token),
        string_mask=build_char_class_mask(
            vocabulary, vocab_size, is_string_token))


def generate_number_token(
        context: GenerationContext, prompt_ids: list[int],
        stays_valid: Callable[[str], bool]) -> str:
    """Generate a numeric value one token at a time, as raw text.

    Stopping rule: every allowed token is part of the number, so nothing
    in the allowed set ever means "done". At each step we also look at
    what the model would have picked with no mask at all. As soon as the
    two disagree, the model would rather write something outside the
    number (a comma, a closing brace), which is how it tells us it has
    finished.
    """
    token_ids = list(prompt_ids)
    text = ""

    for _ in range(MAX_NUMBER_TOKENS):
        logits = context.logits(token_ids)

        # Character class alone is not enough: "1", "." and "2" are all
        # number characters, but "1.2." is not a number, so every
        # candidate is re-checked against the text generated so far.
        allowed = context.number_mask.copy()
        for token_id in np.nonzero(allowed)[0]:
            if not stays_valid(text + context.decode(int(token_id))):
                allowed[token_id] = False
        if not allowed.any():
            break

        # Forbidden tokens are dropped to -inf so that argmax can never
        # land on one, however well the model scored them.
        choice = int(np.argmax(np.where(allowed, logits, -np.inf)))
        if int(np.argmax(logits)) != choice:
            break

        token_ids.append(choice)
        text += context.decode(choice)

    return text


def generate_float(
        context: GenerationContext, prompt_ids: list[int]) -> float:
    """Generate a JSON number, falling back to 0.0 if none came out."""
    text = generate_number_token(context, prompt_ids, is_number_prefix)
    try:
        return float(text)
    except ValueError:
        return 0.0


def generate_int(
        context: GenerationContext, prompt_ids: list[int]) -> int:
    """Generate a whole JSON number, falling back to 0 if none came out."""
    text = generate_number_token(context, prompt_ids, is_integer_prefix)
    try:
        return int(text)
    except ValueError:
        return 0


def unescape(body: str) -> str:
    """Turn a raw string body into the text it stands for.

    The model writes JSON-escapes itself, so a Windows path arrives with
    every backslash doubled the way JSON asks for. Reading the body back
    with json is what collapses each pair into the one character the
    value really holds. A half-written escape is not valid JSON, and
    there the body is kept exactly as it came.
    """
    try:
        return str(json.loads('"' + body + '"'))
    except json.JSONDecodeError:
        return body


def generate_string(
        context: GenerationContext, prompt_ids: list[int]) -> str:
    """Generate the body of a JSON string, quotes excluded.

    Generation ends when the model picks a token holding the closing
    quote, so the model decides the length of the value itself.
    """
    token_ids = list(prompt_ids)
    body = ""

    for _ in range(MAX_STRING_TOKENS):
        logits = context.logits(token_ids)
        choice = int(np.argmax(
            np.where(context.string_mask, logits, -np.inf)))
        text, closing_quote, _ = context.decode(choice).partition('"')
        body += text
        if closing_quote:
            break
        token_ids.append(choice)

    return unescape(body)


def choose_from_candidates(
        context: GenerationContext, prompt_ids: list[int],
        candidates: dict[str, list[int]]) -> str:
    """Walk the candidates' token sequences together and return the best.

    Only a candidate's own tokens are ever offered to the model, so a
    name that was not in the catalog can never come out: the same effect
    as masking every other token to -inf, without ever building a mask.

    Every step drops the candidates that disagree with the token just
    picked. A step where the survivors all want the same token needs no
    model call at all, which is most of what makes this cheap: every
    function name in the catalog starts with the same "fn" token.
    """
    names = [name for name, ids in candidates.items() if ids]
    chosen: list[int] = []

    while len(names) > 1:
        step = len(chosen)
        # A survivor whose tokens are all matched is the answer already:
        # the others only carry on past it, the way fn_read_file carries
        # on past fn_read, and nothing is left to tell them apart on.
        for name in names:
            if len(candidates[name]) == step:
                return name

        options = {candidates[name][step] for name in names}
        if len(options) == 1:
            chosen.append(options.pop())
        else:
            logits = context.logits(prompt_ids + chosen)
            chosen.append(max(options, key=lambda token_id: logits[token_id]))
        names = [name for name in names
                 if candidates[name][:len(chosen)] == chosen]

    return names[0] if names else ""
