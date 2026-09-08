*This project has been created as part of the 42 curriculum by scamlett.*
# CallMeMaybe

## Description
CallMeMaybe is a program that turns a natural-language request
into a structured **function call**. For example, given the prompt:

```
"What is the sum of 2 and 3?"
```

it does not answer "5". Instead, it answers:

```json
{"prompt": "What is the sum of 2 and 3?",
 "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

It works out *which* tool would answer the question and *what arguments* to hand it. 

It runs a small local model (Qwen3-0.6B) through `llm_sdk`. Asked
politely for JSON, the model returns something parseable maybe
**33%** of the time. This tool returns valid, schema-correct JSON
**100% of the time**, because validity never depends on the model
behaving. It is enforced while the text is being generated. That
technique is called **constrained decoding**.


## How it works

### 1. Models read tokens, not letters

Text is chopped into **tokens**, usually word fragments, and each one
is looked up as an integer id. Here is the real tokenization of the
sample prompt, straight out of `model.encode()`:

```
"What is the sum of 2 and 3?"
 -> ['What', ' is', ' the', ' sum', ' of', ' ', '2', ' and', ' ', '3', '?']
 -> [ 3838,   374,   279,    2629,   315,  220,  17,   323,  220,  18,  30 ]
```

### 2. The model outputs logits, one per possible token

Hand the model a list of ids and it returns one **logit** per token in
its vocabulary: 151,936 numbers for Qwen3-0.6B. A logit is an
unnormalised score: higher means "more likely to come next". So the
next token is chosen by taking the `argmax` of that vector.

### 3. Generation is a loop

```python
ids = model.encode(prompt)
while not finished:
    logits = model.get_logits_from_input_ids(ids)
    ids.append(argmax(logits))
```

### 4. Constrained decoding

Between "get the logits" and "pick the best one", insert one step: set the
logit of every token you do not want to negative infinity.

```python
logits = model.get_logits_from_input_ids(ids)
logits = numpy.where(mask, logits, -numpy.inf)  # mask = tokens we allow
next_id = int(numpy.argmax(logits))             # can only be an allowed one
```

A `-inf` token can never win an `argmax`. The model still decides, but
only among the options that were permitted. If a number is being
generated and only number-shaped tokens are allowed, the next character
*cannot* be anything else.


### 5. What the SDK is asked for

Only four public methods are used:

| SDK method | Used by | For |
| --- | --- | --- |
| `encode(text)` | `GenerationContext.encode` | text to token ids |
| `get_logits_from_input_ids(ids)` | `GenerationContext.logits` | one score per next token |
| `get_path_to_vocab_file()` | `build_vocabulary` | locating `vocab.json` |
| `decode([id])` | `build_vocabulary` | what one token id says |

`GenerationContext` in [`src/constraints.py`](src/constraints.py) wraps
the first two, so every SDK call and every tensor-to-list conversion
lives in one place, and the rest of the code deals only in plain `int`
lists and numpy arrays.

### 6. `vocab.json` does not contain plain text

To mask tokens you must know what each id *says*, so you need an
`{id: text}` map from `vocab.json` and a way to decode each id.

Tokenizers are byte-level: a token is a sequence of raw **bytes**, and
bytes like `0x00` or `0x0A` cannot sit inside a JSON string. So
GPT2-style tokenizers substitute every byte with a printable Unicode
stand-in before writing the file. A space becomes `Ġ`, a newline
becomes `Ċ`:

```
vocab.json:  "Ġsum" -> 2629      after decoding:  2629 -> " sum"
vocab.json:  "Ċ"    -> 198       after decoding:  198  -> "\n"
```

`build_vocabulary` in [`src/utils.py`](src/utils.py) reads the ids
(the *values*, since the file is a `{text: id}` map) out of it and asks
the SDK's `decode()` what each one says, turning 151,643 vocabulary
entries into a real `dict[int, str]`. The substitution table is not
rebuilt here: the tokenizer behind `decode()` already reverses it, and
reimplementing it would only be a second copy to keep correct.

## Algorithm explanation

Generation follows a **skeleton + slot** approach.

In the following example:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The braces, the `"name"` key, the colons, the commas, the argument
names, the closing braces: once the shape is known, every one of them
is forced and written as plain text (the *skeleton*). Only two things are real decisions (the *slots*):

### Slot 1: which function to call

`choose_from_candidates()` in
[`src/constraints.py`](src/constraints.py).

Every name in the catalog is encoded to its token ids, and the walk
goes through the candidates side by side, one position at a time. At
each position the surviving candidates offer their next token, the
model picks the best of exactly those, and every candidate that wanted
a different one is dropped. Because only a candidate's own tokens are
ever offered, a name that was not in the catalog can never come out.
That is the same effect as masking every other token to `-inf`,
without ever building the mask.

A position where the survivors all want the same token is skipped
without asking the model: there is nothing to decide. On a real
catalog that is nearly all of the work, because every name starts with
the same `fn` token and they diverge on the second one.

### Slot 2: each argument's value

`generate_number_text()` and `generate_string_value()` in
[`src/constraints.py`](src/constraints.py), one token at a time, with
real logit masking.

- **The masks are precomputed once** at startup from the decoded
  vocabulary (`build_char_class_mask`); rebuilding a 151,936-entry
  array per token would dominate the runtime. For Qwen3-0.6B:
  - `number_mask`: the **84** tokens made only of `0-9 . e E + -`
  - `string_mask`: the **148,552** tokens that may appear while a
    string value is open
- **Character class alone is not enough for numbers.** `1`, `.` and `2`
  are all number characters, but `1.2.` is not a number, and neither
  are `007`, `12.` or `1e`. So each candidate token is also tested with
  `is_number_prefix(text_so_far + token_text)`, a small state machine
  over the JSON number grammar. Only 84 tokens are in the mask, so this
  recheck is cheap.
- **`integer` is its own type, not a flavour of `number`.** A parameter
  declared `integer` has to arrive as `4` and never as `4.0`, because
  the function on the other side type-checks its arguments. Same mask
  and same loop, with a stricter prefix test (`is_integer_prefix`): no
  dot, no exponent.
- **The two value kinds stop differently**, because they signal
  completion differently:
  - A **number** has no token that means "finished": every allowed
    token continues it. So the *unmasked* argmax is computed as well,
    and while the two agree generation continues. The moment the
    model's free choice would rather write something outside the mask
    (a comma, a closing brace), the slot ends.
  - A **string** does have one: the closing `"`. Tokens carrying a
    quote, like `)",`, are in `string_mask` deliberately (1,344 of
    them). When the model picks one, the part before the quote is kept
    and the value ends there.
  - Hard caps (20 tokens for a number, 40 for a string) mean neither
    can hang.
- **Booleans** reuse slot 1's machinery: `"true"` and `"false"` are
  the two candidates.
- **Any other type** is generated as a string rather than rejected.

### Putting one prompt together

`generate_result_for_prompt()` in
[`src/generator.py`](src/generator.py) grows one string, asking the
model only at the slots. Each finished value is written back into that
string with `json.dumps()`, which re-adds the quotes around a string,
escapes what is inside it, and turns a bool into `true`/`false`, so the
running text stays valid JSON for the next decision to be conditioned
on.

The prompt itself (one instruction line plus one worked example using
invented function names) only influences *which* valid answer comes
out. It could never produce invalid output, because structure comes
from the mask, not from asking nicely.

## Design decisions
- **One context object instead of four arguments.**
  `GenerationContext` (a pydantic model) holds the model, the decoded
  vocabulary and both masks. So the decoding functions take
  `(context, prompt_ids)` and nothing else, and its three small methods
  (`encode`, `logits`, `decode`) keep the SDK's tensor handling in
  exactly one place.
- **Every bad input file is fatal.** A broken catalog, a broken prompt
  list, a file that is not an array, an array that is empty: each one
  stops the run with a single readable message on stderr and a
  non-zero exit.
- **Lazy `llm_sdk` import.** `main()` imports `Small_LLM_Model` after
  the input files have been read, so a bad path or malformed JSON is
  reported instantly rather than after torch has finished loading.


## Performance analysis

6 functions, 11 prompts, 1-3 parameters each, on an RTX 4060 laptop
GPU:

| Stage | Cost |
| --- | --- |
| Model load | 2.1 s |
| Startup: decode vocab + build both masks | 1.1 s |
| Generation, 11 prompts | 2.9 s |
| Forward passes | 109 total |
| Time spent inside those passes | 2.8 s, 26 ms each, **97%** of the run |


**JSON reliability is 100%.** A value
physically cannot contain a character its mask forbids, so the output
is always parseable and always matches the declared type. A worse
model would give worse *answers*, never invalid *output*.

**Accuracy** Which function and which argument
values come out depends on how well a 0.6B model scores the right
tokens. It picks the correct function on every prompt tested, and gets
aproximately 80%-90% of the argument values right.


## Challenges faced

- **Understanding tokenization, logits and constrained decoding.** 
  The model is a black box, and the
  only way to know what it is doing is to look at its inputs and outputs
  and reading articles and documentation. The SDK's `get_logits_from_input_ids()`
  returns a tensor of shape `(1, vocab_size)`, which is not documented
  anywhere, so the first few attempts were full of shape errors.

- **Decoding `vocab.json` correctly.** The file maps ids to
  byte-substituted placeholder strings, not to text, and every mask is
  built from that map, so a mistake here surfaces as "the model is
  bad" rather than as an obvious error. Solved by asking the SDK's
  `decode()`.


## Testing strategy

**Edge-case inputs.** `data/input/tests/` holds
alternative catalogs and prompt lists that go well past the bundled
samples, for example;
invalid JSON, an empty array, missing fields, and missing files.

What this checks is the program's real behaviour end to end: that a
catalog of unfamiliar function definitions still produces one valid entry per prompt,
and that a malformed file produces a readable error and a non-zero exit
instead of a traceback.

## Example usage

Given a prompt file containing:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
]
```
and a function catalog containing:
```json
[
  {
    "name": "fn_add_numbers",
    "parameters": [
      {"name": "a", "type": "number"},
      {"name": "b", "type": "number"}
    ]
  }
]
```

`data/output/function_calling_results.json` will contain:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```


## Instructions

Requires Python 3.10+ and uv.

```bash
make install # installs everything, including llm_sdk and its dependencies
make run     # runs the CLI against the default `data/input/` files
make test    # one full run plus the malformed-input cases
make lint    # flake8 and mypy
make debug   # runs it under pdb
make clean   # removes caches
```


## Resources
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [OpenAI GPT-2 source code](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Hugging Face tokenizers: byte-level BPE](https://huggingface.co/docs/tokenizers/en/index)
- [uv documentation](https://docs.astral.sh/uv/)
- [Function calling internals: Grammar and Constrained Sampling](https://www.salmanq.com/blog/llm-constrained-sampling/)
- [Controlling your LLM: Deep dive into Constrained Generation](https://medium.com/@docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)
- [Logits and next-token prediction](https://mikexcohen.substack.com/p/llm-breakdown-26-logits-and-next)
- [Constrained Decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)

## AI usage

AI was used as a tutor to understand new concepts (tokenization, logits, constrained decoding...) and to help in error handling and edge-case testing. All code is reviewed, adapted and understood by the author.