"""Turns a GPT2-style byte-level BPE vocab.json into an id -> text map.

This is the one piece of genuinely fiddly logic in the project, so it
stays dense and commented rather than "simplified": the tokenizer's
vocab.json does not map ids to plain text. It maps ids to a
printable-but-fake string where every raw byte has been substituted
for a printable unicode character (this is how GPT2/Qwen style
tokenizers keep every byte value, including unprintable ones,
representable as JSON text). To know what a token id actually decodes
to, we must reverse that byte-to-unicode substitution ourselves.
"""

import json
from pathlib import Path


def byte_to_unicode() -> dict[int, str]:
    """Build the standard GPT2 byte -> printable character mapping.

    Bytes that are already "nice" printable ASCII/Latin-1 characters
    map to themselves. Every other byte (control characters, the
    space byte, etc.) gets assigned an unused printable character
    further up the unicode range, so every byte has a printable
    stand-in.
    """
    nice_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(nice_bytes)
    next_free_char = 256
    for byte in range(256):
        if byte not in nice_bytes:
            nice_bytes.append(byte)
            chars.append(next_free_char)
            next_free_char += 1
    return {byte: chr(char) for byte, char in zip(nice_bytes, chars)}


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
