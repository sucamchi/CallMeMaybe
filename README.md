*This project has been created as part of the 42 curriculum by scamlett*
# CallMeMaybe

## Description
CallMeMaybe is a command-line tool that turns a natural-language request
into a structured **function call**. Given:

```
"What is the sum of 2 and 3?"
```

it does not answer `5`. It answers:

```json
{"prompt": "What is the sum of 2 and 3?",
 "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

That is all function calling is: a translation layer between how humans
ask and how programs are called. The program works out *which* tool
would answer the question and *what arguments* to hand it. It never
adds anything.

It runs a small local model (Qwen3-0.6B) through `llm_sdk`. Asked
politely for JSON, a model that size returns something parseable maybe
a third of the time. This tool returns valid, schema-correct JSON
**100% of the time**, because validity never depends on the model
behaving — it is enforced while the text is being generated. That
technique is called **constrained decoding**, and the rest of this
README explains it from the ground up.

---

## How it works, from scratch

Five ideas, each small on its own. If you already know them, skip to
[Algorithm explanation](#algorithm-explanation).

### 1. Models read tokens, not letters

Text is chopped into **tokens** — usually word fragments — and each one
is looked up as an integer id. Here is the real tokenization of the
sample prompt, straight out of `model.encode()`:

```
"What is the sum of 2 and 3?"
 -> ['What', ' is', ' the', ' sum', ' of', ' ', '2', ' and', ' ', '3', '?']
 -> [ 3838,   374,   279,    2629,   315,  220,  17,   323,  220,  18,  30 ]
```

Note the leading spaces: `' sum'` (id 2629) and `'sum'` (id 1242) are
**different tokens**. This trips people up constantly, and it is why
this project always re-encodes the whole running text instead of gluing
pre-encoded pieces together.

### 2. The model outputs logits, one per possible token

Hand the model a list of ids and it returns one **logit** per token in
its vocabulary — 151,936 numbers for Qwen3-0.6B. A logit is an
unnormalised score: higher means "more likely to come next". The
familiar "the model wrote a sentence" is just:

```python
next_id = argmax(logits)      # the highest-scoring token wins
```

`llm_sdk.get_logits_from_input_ids(ids)` gives exactly one step of
this, deliberately. Writing the loop around it is the project.

### 3. Generation is a loop you write yourself

```python
ids = model.encode(prompt)
while not done:
    logits = model.get_logits_from_input_ids(ids)
    ids.append(argmax(logits))
```

Run that a few hundred times and you have a paragraph. Run it with a
censored list of candidates and you have this project.

### 4. Constrained decoding: censor the list before you pick

Here is the whole trick, and it is smaller than it sounds. Between
"get the logits" and "pick the best one", insert one step: set the
logit of every token you do not want to negative infinity.

```python
logits = model.get_logits_from_input_ids(ids)
logits = numpy.where(mask, logits, -numpy.inf)  # mask = tokens we allow
next_id = int(numpy.argmax(logits))             # can only be an allowed one
```

A `-inf` token can never win an `argmax`. The model still decides — but
only among the options that were permitted. If a number is being
generated and only number-shaped tokens are allowed, the next character
*cannot* be anything else. Not "usually". Cannot.

That is why reliability jumps to 100%: validity stops being something
you hope for and becomes something the data structure makes impossible
to violate.

### 5. `vocab.json` does not contain plain text

To mask tokens you must know what each id *says* — you need an
`{id: text}` map. `get_path_to_vocab_file()` hands you a `vocab.json`
that looks like a `{text: id}` map. It isn't quite.

Tokenizers are byte-level: a token is a sequence of raw **bytes**, and
bytes like `0x00` or `0x0A` cannot sit inside a JSON string. So
GPT2-style tokenizers substitute every byte with a printable Unicode
stand-in before writing the file. A space becomes `Ġ`, a newline
becomes `Ċ`:

```
vocab.json:  "Ġsum" -> 2629      after decoding:  2629 -> " sum"
vocab.json:  "Ċ"    -> 198       after decoding:  198  -> "\n"
```

[`src/vocab.py`](src/vocab.py) rebuilds that byte-to-unicode table (the
standard one from OpenAI's GPT-2 `encoder.py`) and reverses it, turning
151,643 vocabulary entries into a real `dict[int, str]`. Get this wrong
and every mask built from it is subtly wrong, in a way that looks like
"the model is just bad" rather than like a bug.

---

## Algorithm explanation

Generation follows a **skeleton + slot** approach.

Look hard at the output and ask which characters were ever in doubt:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The braces, the `"name"` key, the colons, the commas, the argument
names, the closing braces — once the shape is known, every one of them
is forced. There is no decision to make, so there is no reason to ask
the model and nothing to mask. They are written directly as plain text
(the *skeleton*). Only two things are real decisions (the *slots*):

### Slot 1: which function to call

`choose_from_candidates()` in
[`src/constraints.py`](src/constraints.py).

Every name in the catalog is encoded to its token ids. For each
candidate, its own tokens are scored one at a time under the shared
prompt context, and the best-scoring name wins. Because only a
candidate's own tokens are ever read, a name that was not offered can
never come out — the same effect as masking every other token to
`-inf`, without ever building the mask.

Two details matter:

- **Log-probabilities, not raw logits.** A logit only means something
  relative to the other logits at that same step, so raw logits are not
  comparable across candidates. Each token's score is converted with a
  numerically stable log-softmax (`_log_softmax_at`).
- **Divided by the token count.** Log-probabilities are negative, so
  summing them punishes long names: a nine-token name would lose to a
  four-token one every time, whatever the prompt asked for. Dividing by
  the length puts every candidate on equal footing.

### Slot 2: each argument's value

`_generate_masked_text()` in [`src/constraints.py`](src/constraints.py),
one token at a time, with real logit masking.

- **The masks are precomputed once** at startup from the decoded
  vocabulary (`build_char_class_mask`); rebuilding a 151,936-entry
  array per token would dominate the runtime. For Qwen3-0.6B:
  - `number_mask` — the **84** tokens made only of `0-9 . e E + -`
  - `string_mask` — the **146,899** tokens with no `"`, no `\` and no
    control character, so nothing generated ever needs escaping
- **Character class alone is not enough for numbers.** `1`, `.` and `2`
  are all number characters, but `1.2.` is not a number — and neither
  are `007`, `12.` or `1e`. So each candidate token is also tested with
  `is_number_prefix_valid(text_so_far + token_text)`, a small state
  machine over the JSON number grammar. Only 84 tokens are in the mask,
  so this recheck is cheap.
- **Stopping rule.** Nothing inside the allowed set ever means "done" —
  every digit token continues the number. So at each step the
  *unmasked* argmax is computed as well. While the two agree,
  generation continues. The moment the model's free choice would rather
  say something outside the mask (a closing quote, a comma), that is
  read as "the model thinks this value is finished" and the slot ends.
  Hard caps (20 tokens for a number, 40 for a string) mean it can never
  hang.
- **Booleans** reuse slot 1's machinery: `"true"` and `"false"` are
  scored as two candidates.
- **Any other type** (absent from the sample data, but possible) is
  generated as a string, with a warning on stderr instead of a crash.

### Putting one prompt together

`generate_result_for_prompt()` in
[`src/generator.py`](src/generator.py) grows one string, asking the
model only at the `?` marks:

```
<instructions, worked example, function catalog, user request>
JSON:
{"name": "?                                        <- slot 1
{"name": "fn_add_numbers", "parameters": {"a": ?    <- slot 2 (number)
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": ?
```

Each finished value is written back into the text with `json.dumps()`,
which re-adds the quotes around a string and turns a bool into
`true`/`false`, so the running text stays valid JSON for the next
decision to be conditioned on.

The prompt itself (instructions plus one worked example using invented
function names) only influences *which* valid answer comes out.
Deleting it would hurt accuracy and could never produce invalid output
— structure comes from the mask, not from asking nicely.

---

## Code map

| File | What lives there |
| --- | --- |
| [`src/__main__.py`](src/__main__.py) | CLI flags, the pipeline, and the single place errors become a message instead of a traceback |
| [`src/io_utils.py`](src/io_utils.py) | Reading the two input JSON files, writing the output one |
| [`src/schema.py`](src/schema.py) | The pydantic models: a function, a parameter, a prompt, a result |
| [`src/vocab.py`](src/vocab.py) | `vocab.json` -> `{id: text}` (idea 5 above) |
| [`src/constraints.py`](src/constraints.py) | `GenerationContext`, the masks, and all constrained decoding |
| [`src/generator.py`](src/generator.py) | Prompt text, function choice, per-prompt assembly, fallback |

The flow is one straight line — no callbacks, no inheritance:

```
parse args -> load catalog -> load prompts -> load model
           -> build context (decode vocab, precompute masks)
           -> per prompt: choose function, then fill each slot
           -> write the JSON array
```

---

## Design decisions
- **One context object instead of four arguments.**
  `GenerationContext` (a pydantic model) holds the model, the decoded
  vocabulary and both masks — everything every decoding step needs. So
  the decoding functions take `(context, prompt_ids)` and nothing else,
  and its three small methods (`encode`, `logits`, `text_of`) keep the
  SDK's tensor handling in exactly one place.
- **Functions over classes everywhere else.** Only things holding real
  data are pydantic models, as required: `FunctionDefinition`,
  `FunctionParameter`, `PromptRecord` and `ResultRecord` at the I/O
  boundary, plus the context above. The decoding logic is plain
  functions with local variables — there is no other state worth
  grouping into an object.
- **Fatal vs. survivable failures, decided deliberately.** A broken
  catalog is fatal: nothing can be called, so the run stops with a
  clear message. A broken *prompt* is not — it is skipped with a
  warning and the rest still run. And if generation for one prompt
  fails or runs away, `fallback_result()` still emits a valid,
  schema-correct entry, so the output array always holds exactly one
  entry per accepted prompt.
- **Re-encoding the running text** after every skeleton insertion,
  instead of splicing pre-encoded id fragments together. BPE token
  boundaries shift with what precedes them, so splicing can build a
  sequence the model never saw in training. Re-encoding a short string
  each time removes a whole class of boundary bugs at negligible cost.
- **Forbid, rather than escape.** A `"` or `\` inside a generated
  string would break the JSON around it. Both were handled by leaving
  those tokens out of `string_mask` entirely, so nothing that needs
  escaping can ever be produced, and no escaping pass is needed
  afterwards. The cost is real and worth naming at defence: a value
  that genuinely wants a backslash — a regex like `\d+`, a Windows
  path — is unreachable. Escaping on the way out would allow those,
  but then a generated `"` could no longer be read as "the value is
  finished", which is exactly what the stopping rule relies on.
- **Lazy `llm_sdk` import.** `run()` imports `Small_LLM_Model` only
  when no model was passed in, so a bad input file is reported
  instantly rather than after torch has loaded — and the whole pipeline
  can be driven by a fake model with no download and no GPU.

---

## Performance analysis
Measured on the reference machine (NVIDIA RTX 4060 Laptop GPU, which
`llm_sdk` auto-selects, `float16`) against the bundled sample files:
5 functions, 11 prompts, 1-3 parameters each.

| Stage | Cost |
| --- | --- |
| Model load | 2.6 s |
| Startup: decode vocab + build both masks | 1.6 s (0.4 s of it decoding 151,643 entries) |
| Whole pipeline, 11 prompts | 14.2 s |
| Forward passes | 291 total, ~26 per prompt |
| Time spent inside those passes | 11.4 s — 39 ms each, **80%** of the run |

The dominant cost is the number of `get_logits_from_input_ids` calls,
one full forward pass each. The masking itself does not show up: it is
a single vectorised `numpy.where` + `argmax` per step, and the
number-prefix recheck loops over only the 84 tokens in the number mask.
The remaining ~20% is startup, tokenizer calls and JSON I/O.

Two consequences worth stating plainly:

- **Reliability is 100% by construction**, not by measurement. A value
  physically cannot contain a character its mask forbids, so the output
  is always parseable and always matches the declared type. A worse
  model would give worse *answers*, never invalid *output*.
- **Accuracy is the model's job.** Which function and which argument
  values come out depends on how well a 0.6B model scores the right
  tokens. On the 11 sample prompts it picks the correct function every
  time, and the correct argument values everywhere except one regex
  case: for *"replace all numbers ... with NUMBERS"* it fills `regex`
  with the literal `34` from the sentence instead of a general
  pattern. A larger model would choose better; nothing about the
  output's validity changes either way.

That is comfortably inside the subject's "under five minutes for all
prompts" — about 20x of headroom on GPU, and still well inside it on
CPU.

---

## Challenges faced
- **Decoding `vocab.json` correctly.** The file maps ids to
  byte-substituted placeholder strings, not to text. Getting a usable
  `id -> text` map meant reimplementing the standard GPT2
  byte-to-unicode table and reversing it. This is the one genuinely
  fiddly piece in the project and the one everything depends on: every
  mask is built from that map, so a mistake here surfaces as "the model
  is bad", not as an obvious error.
- **Deciding when a value is "done".** JSON has no "end of number"
  token, and by construction every allowed token continues the value.
  Comparing the model's free (unmasked) choice against the masked one
  at every step turned out to be a simple, faithful way to let the
  model signal completion itself, with no dedicated stop token.
- **A raw-logit scoring bug.** The first version of function selection
  summed each candidate's raw logits, which silently favours whichever
  name has more tokens; every prompt picked the longest name. The fix
  was length-normalised log-probabilities — the standard way to compare
  sequences of different lengths.
- **Validating without the real model.** Downloading and running a 0.6B
  model for every check is slow and non-deterministic. A stand-in
  implementing the same four public methods and returning scripted
  logits makes the decoding logic testable in milliseconds.

---

## Testing strategy
Testing happens at two levels.

**Committed edge-case inputs.** `data/input/extra_tests/` holds
alternative catalogs and prompt lists that go well past the bundled
samples, because the subject warns that the input files are swapped
during peer review. They cover `boolean` arguments and a zero-argument
function (neither appears in the samples), a catalog of near-identical
descriptions where only the wording distinguishes the right function,
argument types outside `number`/`string`/`boolean`, number shapes JSON
is picky about (huge, negative, decimal, and `007`, which must not
survive as a leading zero), string values containing quotes,
backslashes, newlines and non-ASCII, and prompts that are unrelated,
empty, whitespace-only or injection-flavoured. A second group of
deliberately malformed files checks the error paths the subject names:
invalid JSON, a JSON object where an array is required, an empty
catalog, missing fields, duplicate function names, and missing files.
`data/input/extra_tests/README.md` lists every case with the outcome it
should produce. They are run by pointing the normal CLI at them:

```
uv run python -m src \
  --functions_definition data/input/extra_tests/functions_mixed_types.json \
  --input data/input/extra_tests/prompts_mixed_types.json \
  --output data/output/mixed_types.json
```

**A fake model during development.** Because `run()` accepts a model
object, every part of the program (vocab decoding, number/string/
boolean generation, the full pipeline, the CLI error paths) can be
driven by a stand-in implementing the same public interface as
`llm_sdk.Small_LLM_Model` and returning scripted logits — no download,
no GPU, fully deterministic. That is what caught the raw-logit scoring
bug above.

Final verification is end to end against the real model: `make run` on
the bundled files, checking the output is valid JSON, has one entry per
prompt with exactly the three required keys, and that the names and
argument values are actually right. Every change also has to leave
`make lint` and `make lint-strict` clean.

---

## Example usage
```
uv sync
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Or simply `make run` for the default paths. Given a prompt file
containing `"What is the sum of 2 and 3?"`,
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

Every entry has exactly three keys — `prompt`, `name`, `parameters` —
and nothing else. Numbers come back as JSON numbers, booleans as
`true`/`false`, strings as strings.

Failures are reported, never crashed:

```
$ uv run python -m src --functions_definition missing.json
Error: could not read missing.json: [Errno 2] No such file or directory: 'missing.json'
```

# Instructions
- Requires Python 3.10+ and uv.
- `make install` (or `uv sync`) installs everything, including
  `llm_sdk` (a local path dependency) and its own dependencies (torch,
  transformers, huggingface-hub) needed to actually run the model.
- `make run` runs the CLI against the default `data/input/` files.
- `make debug` runs it under `pdb`.
- `make lint` / `make lint-strict` run `flake8` and `mypy`.
- `make clean` removes caches.

## Resources
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
  — the number and string grammars that `is_number_prefix_valid` and
  `is_string_safe_text` implement.
- [OpenAI GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
  — source of the standard byte-to-unicode table reimplemented in
  [`src/vocab.py`](src/vocab.py).
- [Hugging Face tokenizers: byte-level BPE](https://huggingface.co/docs/tokenizers/en/index)
- [uv documentation](https://docs.astral.sh/uv/)

## AI usage

