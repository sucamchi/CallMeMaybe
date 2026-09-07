*This project has been created as part of the 42 curriculum by scamlett.*
# CallMeMaybe

## Description
CallMeMaybe is a command-line tool that turns a natural-language request
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

The program works out *which* tool would answer the question and *what arguments* to hand it. 
It never adds anything.

It runs a small local model (Qwen3-0.6B) through `llm_sdk`. Asked
politely for JSON, the model returns something parseable maybe
33% of the time. This tool returns valid, schema-correct JSON
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
unnormalised score: higher means "more likely to come next".
So the next token is chosen by taking the `argmax` of that vector:

```python
next_id = argmax(logits)      # the highest-scoring token wins
```

### 3. Generation is written as a loop: ask the model, pick the best token, append it, repeat.

```python
ids = model.encode(prompt)
while not finished:
    logits = model.get_logits_from_input_ids(ids)
    ids.append(argmax(logits))
```

### 4. Constrained decoding: censor the list before you pick

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

That is how reliability jumps to 100%.

### 5. `vocab.json` does not contain plain text

To mask tokens you must know what each id *says*, so you need an
`{id: text}` map. `get_path_to_vocab_file()` hands you a `vocab.json`
that looks like a `{text: id}` map.

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

Only the ids the file names are decoded. The special tokens appended
after it (`<|im_start|>`, `<think>`, `<tool_call>`) are deliberately
left out, so no mask can ever allow one.


## Algorithm explanation

Generation follows a **skeleton + slot** approach.

In the following example:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The braces, the `"name"` key, the colons, the commas, the argument
names, the closing braces: once the shape is known, every one of them
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
never come out. That is the same effect as masking every other token
to `-inf`, without ever building the mask.

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
  - `number_mask`: the **84** tokens made only of `0-9 . e E + -`
  - `string_mask`: the **146,899** tokens with no `"`, no `\` and no
    control character, so nothing generated ever needs escaping
- **Character class alone is not enough for numbers.** `1`, `.` and `2`
  are all number characters, but `1.2.` is not a number, and neither
  are `007`, `12.` or `1e`. So each candidate token is also tested with
  `is_number_prefix_valid(text_so_far + token_text)`, a small state
  machine over the JSON number grammar. Only 84 tokens are in the mask,
  so this recheck is cheap.
- **Stopping rule.** Nothing inside the allowed set ever means "done",
  because every digit token continues the number. So at each step the
  *unmasked* argmax is computed as well. While the two agree,
  generation continues. The moment the model's free choice would rather
  say something outside the mask (a closing quote, a comma), that is
  read as "the model thinks this value is finished" and the slot ends.
  Hard caps (20 tokens for a number, 40 for a string) mean it can never
  hang.
- **Booleans** reuse slot 1's machinery: `"true"` and `"false"` are
  scored as two candidates.
- **Any other type** (absent from the sample data, but possible) is
  generated as a string rather than rejected.

### Putting one prompt together

`generate_result_for_prompt()` in
[`src/generator.py`](src/generator.py) grows one string, asking the
model only at the `?` marks:

Each finished value is written back into the text with `json.dumps()`,
which re-adds the quotes around a string and turns a bool into
`true`/`false`, so the running text stays valid JSON for the next
decision to be conditioned on.

The prompt itself (instructions plus one worked example using invented
function names) only influences *which* valid answer comes out. It
could never produce invalid output, because structure comes from the
mask, not from asking nicely. It is not decoration, though: removing
the worked example drops function selection on the sample set from
11/11 to 7/11, because the model starts picking the function with the
most arguments over the one whose description matches.


## Execution flow

The flow is one straight line:


parse args -> load catalog -> load prompts -> load model -> build context (decode vocab, precompute masks) -> per prompt: choose function, then fill each slot -> write the JSON array



## Design decisions
- **One context object instead of four arguments.**
  `GenerationContext` (a pydantic model) holds the model, the decoded
  vocabulary and both masks. So
  the decoding functions take `(context, prompt_ids)` and nothing else,
  and its three small methods (`encode`, `logits`, `decode`) keep the
  SDK's tensor handling in exactly one place.
- **Functions over classes everywhere else.** Only things holding real
  data are pydantic models, as required: `FunctionDef`, `FunctionParam`,
  `Prompt` and `OutputResult` at the I/O boundary, plus the context
  above. The decoding logic is plain
  functions with local variables.
- **Fatal vs. survivable failures, decided deliberately.** A broken
  catalog is fatal: nothing can be called, so the run stops with a
  clear message. A broken *prompt* is not: it is skipped with a
  warning and the rest still run, so the output array holds exactly one
  entry per accepted prompt. There is no per-prompt recovery path
  beyond that: generation cannot fail once a catalog has loaded, and a
  handler for a case that cannot arise would be dead code.
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
  that genuinely wants a backslash (a regex like `\d+`, a Windows
  path) is unreachable. Escaping on the way out would allow those,
  but then a generated `"` could no longer be read as "the value is
  finished", which is exactly what the stopping rule relies on.
- **Lazy `llm_sdk` import.** `run()` imports `Small_LLM_Model` after
  the input files have been read, so a bad path or malformed JSON is
  reported instantly rather than after torch has finished loading.


## Performance analysis

5 functions, 11 prompts, 1-3 parameters each.

| Stage | Cost |
| --- | --- |
| Model load | 2.6 s |
| Startup: decode vocab + build both masks | 1.6 s (0.4 s of it decoding 151,643 entries) |
| Whole pipeline, 11 prompts | 14.2 s |
| Forward passes | 291 total|
| Time spent inside those passes | 11.4 s, 39 ms each, **80%** of the run |

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

## Challenges faced
- **Decoding `vocab.json` correctly.** The file maps ids to
  byte-substituted placeholder strings, not to text, and every mask is
  built from that map, so a mistake here surfaces as "the model is
  bad" rather than as an obvious error. The first version rebuilt the
  standard GPT2 byte-to-unicode table by hand and reversed it. It is
  now the SDK's `decode()` that does this, which is the same table
  reached through the tokenizer that wrote the file instead of a second
  copy of it maintained here. Checked against the hand-rolled version:
  identical text for all 151,643 entries.
- **Knowing where the vocabulary stops.** The ids in `vocab.json` run
  to 151,642, the logits row is 151,936 wide, and the 26 ids in between
  are special tokens. 14 of those 26 are not marked "special" in the
  tokenizer, so `decode()` returns their literal text (`<tool_call>`,
  `<think>`) rather than `""`, and any of those would pass the string
  mask's character test. Driving the map from the file's ids, not from
  the width of a logits row, is what keeps them out.
- **Deciding when a value is "done".** JSON has no "end of number"
  token, and by construction every allowed token continues the value.
  Comparing the model's free (unmasked) choice against the masked one
  at every step turned out to be a simple, faithful way to let the
  model signal completion itself, with no dedicated stop token.
- **A raw-logit scoring bug.** The first version of function selection
  summed each candidate's raw logits, which silently favours whichever
  name has more tokens; every prompt picked the longest name. The fix
  was length-normalised log-probabilities, the standard way to compare
  sequences of different lengths.
- **Validating without the real model.** Downloading and running a 0.6B
  model for every check is slow and non-deterministic. A stand-in
  implementing the same four public methods and returning scripted
  logits makes the decoding logic testable in milliseconds.


## Testing strategy
Testing happens at two levels.

**Committed edge-case inputs.** `data/input/tests/` holds
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
catalog, missing fields, and missing files.

**`make test` runs them.** There is no test framework and no test
code: the target invokes the same CLI a reviewer would. It is one
generation run, `functions_all.json` (17 functions) against
`prompts_all.json` (40 prompts), so the model is loaded exactly once,
followed by eight malformed-input cases inverted with a shell `!`.
Those pass by failing, and they are nearly free: a bad input file is
reported before `llm_sdk` is ever imported. Make stops at the first
case that misbehaves. The whole target takes about three minutes and
writes to the same `data/output/function_calling_results.json` as
`make run`, with no separate output files.

What this checks is the program's real behaviour end to end: that a
catalog of unfamiliar shapes still produces one valid entry per prompt,
and that a malformed file produces a readable error and a non-zero exit
instead of a traceback. Both merged files are themselves entirely
valid and load with no warning, so anything printed during that first
run is a real problem, not expected noise. What it deliberately
does not do is assert which function the model picks: that is the
model's judgement, it is reviewed by reading the output, and pinning it
down in an assertion would only encode today's answers.

Final verification is end to end against the real model: `make run` on
the bundled files, checking the output is valid JSON, has one entry per
prompt with exactly the three required keys, and that the names and
argument values are actually right. Every change also has to leave
`make lint` and `make lint-strict` clean.

## Example usage
```
uv sync
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

## Instructions

Requires Python 3.10+ and uv.

```bash
make install # installs everything, including llm_sdk and its dependencies
make run     # runs the CLI against the default `data/input/` files
make debug   # runs it under pdb
make clean   # removes caches
```

## Bonus


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