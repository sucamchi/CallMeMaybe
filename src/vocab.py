"""Turns a GPT2-style BPE vocab.json into an id -> text map.

The tokenizer's vocab.json does not map ids to plain text.
It maps ids to a string where every raw byte has been substituted
for a printable unicode character. To know what a token id actually decodes
to, we must reverse that byte-to-unicode substitution ourselves.
"""

import json
from pathlib import Path


def byte_to_unicode() -> dict[int, str]:
    """Build the standard GPT2 byte -> printable character mapping.

    Bytes that are already printable ASCII/Latin-1 characters
    map to themselves. Every other byte (control characters, the
    space byte, etc.) gets assigned an unused printable character
    further up the unicode range.
    """
    bad_bytes = (
        set(range(0, ord("!")))
        | set(range(ord("~") + 1, ord("¡")))
        | set(range(ord("¬") + 1, ord("®")))
    )
    mapping = {}
    next_free_char = 256
    for byte in range(256):
        if byte in bad_bytes:
            mapping[byte] = chr(next_free_char)
            next_free_char += 1
        else:
            mapping[byte] = chr(byte)
    return mapping


def build_vocabulary(vocab_file_path: str) -> dict[int, str]:
    """Load a vocab.json file and decode every token id to its text."""
    try:
        raw_vocab = json.loads(
            Path(vocab_file_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"could not read vocab file {vocab_file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid vocab file {vocab_file_path}: {exc}") from exc

    char_to_byte = {char: byte
                    for byte, char in byte_to_unicode().items()}
    id_to_text: dict[int, str] = {}
    for token_string, token_id in raw_vocab.items():
        try:
            raw_bytes = bytes(char_to_byte[char] for char in token_string)
        except KeyError:
            continue  # not a byte-substituted token: nothing to decode
        id_to_text[token_id] = raw_bytes.decode("utf-8", errors="replace")
    return id_to_text
