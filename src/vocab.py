"""Turn a GPT2-style BPE vocab.json into an id -> text map.

A vocab.json file cannot hold raw bytes, so GPT2 rewrites every byte as
a printable character before saving. That means the file maps ids to
substituted text, not to the text a token really says. To find out what
a token id decodes to, we have to undo the substitution ourselves.
"""
import json


def byte_to_unicode() -> dict[int, str]:
    """Build the standard GPT2 byte -> printable character mapping.

    Bytes that are already printable stand for themselves. Every other
    byte (control codes, the space byte, and a few gaps in latin-1)
    borrows an unused character further up in unicode.
    """
    printable = (
        set(range(ord("!"), ord("~") + 1))    # printable ASCII
        | set(range(ord("¡"), ord("¬") + 1))  # printable latin-1
        | set(range(ord("®"), 256))           # printable latin-1
    )
    mapping = {}
    next_free_char = 256
    for byte in range(256):
        if byte in printable:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(next_free_char)
            next_free_char += 1
    return mapping


def build_vocabulary(vocab_file_path: str) -> dict[int, str]:
    """Load a vocab.json file and decode every token id to its text."""
    try:
        with open(vocab_file_path, encoding="utf-8") as file:
            vocab = json.load(file)
    except OSError as exc:
        raise ValueError(
            f"could not read vocab file {vocab_file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid vocab file {vocab_file_path}: {exc}") from exc

    char_to_byte = {char: byte
                    for byte, char in byte_to_unicode().items()}
    id_to_text: dict[int, str] = {}
    for token_string, token_id in vocab.items():
        try:
            raw_bytes = bytes(char_to_byte[char] for char in token_string)
        except KeyError:
            continue  # not a byte-substituted token: nothing to decode
        # A single accented letter or emoji can be split across two
        # tokens, so one token's bytes are not always valid UTF-8 on
        # their own. Those broken halves become U+FFFD instead of
        # raising, and the masks simply never allow them.
        id_to_text[token_id] = raw_bytes.decode("utf-8", errors="replace")
    return id_to_text
