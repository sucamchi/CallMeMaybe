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


def is_number_prefix_valid(text: str) -> bool:
    """True if text is a JSON number or could still grow into one.

    This runs on half-finished values, so "-" and "1." have to pass
    even though neither is a number yet. Only text that can never
    become one, like "01" or "1.2.", is rejected.
    """
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


def is_json_number(text: str) -> bool:
    """True if text uses only characters a JSON number can contain."""
    return all(char in NUMBER_CHARS for char in text)


def is_string_safe_text(text: str) -> bool:
    """True if text can sit inside a JSON string body.

    Leaving the quote out of this class is what ends a string value:
    generation stops as soon as the model wants a token the mask
    forbids, and what it wants after a finished value is the closing
    quote. Characters below 0x20 are illegal raw inside a JSON string,
    and a backslash would open an escape the model may never close.
    """
    return all(char not in '"\\' and ord(char) >= 0x20 for char in text)


def build_char_class_mask(
        vocabulary: dict[int, str], vocab_size: int,
        is_allowed: Callable[[str], bool]) -> np.ndarray:
    """Precompute which token ids decode to a given char class."""
    mask = np.zeros(vocab_size, dtype=bool)
    for token_id, text in vocabulary.items():
        # The vocab file can name ids past the end of a logits row (see
        # build_generation_context), and those have no slot to mask in.
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
            vocabulary, vocab_size, is_json_number),
        string_mask=build_char_class_mask(
            vocabulary, vocab_size, is_string_safe_text))


def generate_masked_text(
        context: GenerationContext, prompt_ids: list[int],
        mask: np.ndarray, max_tokens: int,
        stays_valid: Callable[[str], bool] | None = None) -> str:
    """Generate text one token at a time, never leaving the mask.

    Stopping rule: since only allowed tokens can be picked, nothing in
    the allowed set ever means "done". So at every step we also look at
    what the model would have picked with no mask at all. As soon as the
    two disagree, the model would rather say something outside the value
    (a closing quote, a comma), which is how it tells us it has
    finished.
    """
    generated_ids: list[int] = []
    text = ""

    for _ in range(max_tokens):
        logits = context.logits(prompt_ids + generated_ids)

        allowed = mask
        if stays_valid is not None:
            # Character class alone is not enough for numbers: "1", "."
            # and "2" are all number characters, but "1.2." is not a
            # number, so every candidate is re-checked against the text
            # generated so far.
            allowed = mask.copy()
            for token_id in np.nonzero(allowed)[0]:
                if not stays_valid(text + context.decode(int(token_id))):
                    allowed[token_id] = False
        if not allowed.any():
            break

        # Forbidden tokens are dropped to -inf so that argmax can never
        # land on one, however well the model scored them.
        constrained_choice = int(np.argmax(np.where(allowed, logits, -np.inf)))
        if int(np.argmax(logits)) != constrained_choice:
            break

        generated_ids.append(constrained_choice)
        text += context.decode(constrained_choice)

    return text


def generate_number_value(
        context: GenerationContext, prompt_ids: list[int]) -> float:
    """Generate a JSON number, falling back to 0.0 if none came out."""
    text = generate_masked_text(
        context, prompt_ids, context.number_mask, MAX_NUMBER_TOKENS,
        is_number_prefix_valid)
    try:
        return float(json.loads(text))
    except ValueError:
        return 0.0


def generate_string_value(
        context: GenerationContext, prompt_ids: list[int]) -> str:
    """Generate the body of a JSON string, quotes excluded."""
    return generate_masked_text(
        context, prompt_ids, context.string_mask, MAX_STRING_TOKENS)


def _log_softmax_at(logits: np.ndarray, token_id: int) -> float:
    """Log-probability of one token id under the model's full distribution.

    Softmax is exp(logit) / sum(exp(logits)). Subtracting the largest
    logit before any exp() is the usual guard against exp() overflowing
    on a big one: it cancels out of the division, so the answer is the
    same, but nothing is ever raised above exp(0).
    """
    peak = np.max(logits)
    log_sum_exp = peak + np.log(np.sum(np.exp(logits - peak)))
    return float(logits[token_id] - log_sum_exp)


def choose_from_candidates(
        context: GenerationContext, prompt_ids: list[int],
        candidates: dict[str, list[int]]) -> str:
    """Score each candidate's token sequence and return the best name.

    Only a candidate's own tokens are ever read, so a name that was not
    offered can never come out, the same effect as masking every other
    token to -inf without ever building the mask.

    Scores are log-probabilities rather than raw logits, summed and then
    divided by the token count: raw logits are not comparable across
    candidates of different lengths, and dividing by the length keeps a
    long name from losing purely for being long.
    """
    best_name = ""
    best_score = float("-inf")
    # Candidates share prefixes: every function name starts with the
    # same "fn_" token, and every candidate's first step sees the same
    # context. So each logits row is computed once and reused, keyed by
    # the tokens consumed so far (the prompt is fixed for the call).
    logits_by_prefix: dict[tuple[int, ...], np.ndarray] = {}

    for name, token_ids in candidates.items():
        if not token_ids:
            continue  # a name that encodes to nothing has no score
        total_log_prob = 0.0
        running_ids = list(prompt_ids)
        for position, token_id in enumerate(token_ids):
            prefix = tuple(token_ids[:position])
            if prefix not in logits_by_prefix:
                logits_by_prefix[prefix] = context.logits(running_ids)
            total_log_prob += _log_softmax_at(
                logits_by_prefix[prefix], token_id)
            running_ids.append(token_id)
        average_log_prob = total_log_prob / len(token_ids)
        if average_log_prob > best_score:
            best_score = average_log_prob
            best_name = name

    return best_name
