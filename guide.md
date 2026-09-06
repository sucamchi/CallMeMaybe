# Call Me Maybe — a guide to the project

You start with three things: `en.subject.pdf`, the `llm_sdk/` package,
and a `data/` folder holding two small JSON files. Everything else is
yours to write.

This guide explains what the project actually asks for, the handful of
ideas you need before you can write a line of it, and an order to build
it in. It does not hand you the solution — the evaluation is a defence,
and code you cannot explain is worse than no code.

---

## 1. What you are building, in one paragraph

A program that reads a natural-language request and answers with a
**function call**, not with the answer.

Given `"What is the sum of 40 and 2?"`, a normal chatbot says *"42"*.
Your program instead says:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

It never adds anything. It works out *which* tool would answer the
question and *what arguments* to hand it. That is all function calling
is: a translation layer between how humans ask and how programs are
called.

The catch is the reliability bar. A 0.6-billion-parameter model asked
politely for JSON produces valid JSON maybe a third of the time. The
subject demands **100%**. Getting from 30% to 100% is the entire
project.

---

## 2. What you are given, and what you must write

**Given:**

- `llm_sdk/` — a thin wrapper around a local model (Qwen3-0.6B). Four
  methods matter:
  - `get_logits_from_input_ids(ids)` — the important one. Hand it a list
    of token ids, get back a score for every possible next token.
  - `get_path_to_vocab_file()` — where the tokenizer's vocabulary lives.
  - `encode(text)` — text to token ids.
  - `decode(ids)` — token ids back to text (optional).
- `data/input/functions_definition.json` — the catalog of functions you
  are allowed to call: name, description, argument names and types,
  return type.
- `data/input/function_calling_tests.json` — a list of prompts to
  process.

**You write:** a `src/` package, a `pyproject.toml`, a `Makefile`, a
`README.md`, and a `.gitignore`. Run as:

```
uv run python -m src [--functions_definition FILE] [--input FILE] [--output FILE]
```

with sensible defaults pointing into `data/`. Output goes to
`data/output/function_calling_results.json` as a JSON array where every
entry has exactly three keys — `prompt`, `name`, `parameters` — and
nothing else.

**Hard limits, from the subject:**

- Python 3.10+, clean `flake8`, clean `mypy` with the flags the subject
  lists, type hints everywhere, docstrings.
- All classes must be **pydantic** models.
- You may use `numpy` and `json`. You may **not** use `torch`,
  `transformers`, `outlines`, `dspy`, or anything similar. Your only
  route to the model is `llm_sdk`.
- No private methods or attributes from `llm_sdk` (nothing starting with
  `_`).
- **The function must be chosen by the LLM.** No keyword matching, no
  "if 'sum' in prompt". This is checked.
- The program must never crash. Every error path gets a clear message.

---

## 3. The five ideas you need first

Do not start coding until these feel solid. Most of the difficulty in
this project is conceptual, not technical.

### 3.1 Models see token ids, not text

Text is chopped into **tokens** — usually word fragments — and each is
looked up as an integer id. `"What is the sum"` might become
`["What", " is", " the", " sum"]` and then `[3838, 374, 279, 2694]`.
Note the leading spaces: they belong to the token. This trips people up
constantly.

### 3.2 The model outputs logits, one per possible token

Give the model a sequence of ids and it returns a **logit** for every
token in its vocabulary — roughly 150,000 numbers for Qwen. A logit is
an unnormalised score: higher means "more likely to come next". Softmax
turns the whole list into probabilities, but you rarely need to.

### 3.3 Generation is a loop, one token at a time

```
ids = encode(prompt)
repeat:
    logits = get_logits_from_input_ids(ids)
    next_id = argmax(logits)          # pick the best-scoring token
    ids.append(next_id)
```

That is genuinely all "the model wrote a paragraph" means: this loop,
run a few hundred times. **You have to write this loop yourself.** The
SDK gives you one step of it, deliberately.

### 3.4 Constrained decoding: censor the list before you pick

Here is the whole trick, and it is smaller than it sounds.

Between "get the logits" and "pick the best one", you insert a step:
set the logit of every token you don't want to negative infinity.

```
logits = get_logits_from_input_ids(ids)
logits[every token that would break my format] = -inf
next_id = argmax(logits)              # can only be a token you allowed
```

An `-inf` token can never win an `argmax`. So the model still decides —
but only from options you permitted. If you are in the middle of a
number and only digit tokens are allowed, the output *cannot* be
anything but a digit. Not "usually". Cannot.

This is why the reliability jumps to 100%: validity stops being
something you hope for and becomes something the data structure makes
impossible to violate.

The subject is explicit that this must hold for the **schema**, not just
for JSON syntax. If the catalog says an argument is a `number`, only
tokens that keep it a valid number may be allowed — not merely tokens
that keep the JSON parseable.

### 3.5 The vocabulary file does not contain plain text

This is the part nobody warns you about, and it will eat a day if you
let it.

To mask tokens you must know what each token id *says* — you need a
`{id: text}` map. `get_path_to_vocab_file()` gives you a `vocab.json`
that looks like a `{text: id}` map. It isn't quite.

Tokenizers are byte-level: a token is a sequence of raw **bytes**, and
bytes include things like `0x00` and `0x0A` that cannot sit inside a
JSON string. So GPT-2-style tokenizers substitute every byte for some
printable Unicode character before writing the file. A space becomes
`Ġ`. A newline becomes `Ċ`. Byte `0x00` becomes something else again.

So `vocab.json` maps *byte-substituted placeholder strings* to ids. To
recover real text you must rebuild that byte-to-character table and
reverse it. The table is a fixed, known thing — the standard reference
is the `bytes_to_unicode` function in OpenAI's GPT-2 `encoder.py`.
Reimplement it; it is about ten lines.

Get this wrong and every mask you build is subtly wrong, in a way that
looks like "the model is just bad". Test it on its own, early.

---

## 4. Why you are forbidden from just asking nicely

The subject bans the obvious approach in a red box:

> Your solution must NOT rely on the model spontaneously producing
> correct JSON from a prompt.

Two reasons. The practical one: a 0.6B model will hand you
` ```json`, or trailing prose, or a missing brace, often enough to
blow the 100% requirement. The real one: prompting is not a skill this
project is teaching. Sampling under constraints is.

A useful test of your own design — if deleting your prompt text would
break JSON validity rather than just hurt accuracy, you have built the
wrong thing. Structure should come from the mask. The prompt should
only influence *which* valid answer you get.

---

## 5. The realisation that makes this tractable

Look hard at what you are producing:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

Now ask which characters were ever in doubt.

The `{`, the `"name"`, the `:`, the `"parameters"`, the argument names,
the commas, the closing braces — none of them. Once you know the shape
of the output and which function was picked, every one of those is
forced. There is no decision to make, so there is no reason to ask the
model and no reason to mask anything.

Only two things are genuine decisions:

1. **which function**, and
2. **each argument's value**.

So you don't constrain a free-running generator into producing JSON.
You write the JSON yourself, leaving blanks, and use the model only to
fill the blanks. The fixed text is just string concatenation. Far
simpler, far faster, and impossible to get structurally wrong.

Two design questions follow, and they are the interesting part of the
project:

- **Choosing the function.** You have a fixed list of candidate names.
  How do you make the model rank them, using only logits, without
  letting it invent a sixth name? (Think about scoring each candidate's
  token sequence rather than generating freely. Then think about what
  goes wrong when one name is four tokens and another is nine — raw
  scores are not comparable across different lengths. Working that out
  is worth the time; it is a classic bug and a good thing to be able to
  explain at defence.)
- **Knowing when a value ends.** You are generating the digits of a
  number with only digit tokens allowed. Nothing in that allowed set
  means "done". How does the model tell you it has finished, if every
  token it can pick continues the number? (Hint: you are allowed to
  look at what the model *would* have chosen without the mask.)

---

## 6. A build order

Each milestone is independently checkable. Do not skip ahead — a bug in
step 2 is invisible until step 6 and looks like a model problem.

**Step 0 — Skeleton.** `pyproject.toml` with `numpy` and `pydantic` and
`llm_sdk` as a local path dependency; the `Makefile` rules the subject
lists (`install`, `run`, `debug`, `clean`, `lint`, `lint-strict`); a
`.gitignore`; an empty `src/` that runs and exits. Confirm `make lint`
passes on nothing before it has to pass on something.

**Step 1 — Data in and out.** Pydantic models for a function
definition, a prompt, a result. Load both input files, validate them,
write a hardcoded result array out. Handle every failure the subject
names: missing file, invalid JSON, not an array, wrong fields. Decide
deliberately which failures are fatal (a broken catalog — you cannot
call anything) and which are survivable (one broken prompt among
twenty — skip it with a warning and keep going).

At this point you have a working program that is simply wrong. That is
a good place to be.

**Step 2 — The vocabulary.** Load `vocab.json`, reverse the
byte-substitution, produce `{id: text}`. Verify by hand: look up the
ids for a few tokens you can predict, and check that the token for a
space really comes back as `" "` and not `"Ġ"`. Getting this right
before anything depends on it will save you hours.

**Step 3 — Talk to the model at all.** Encode a short prompt, call
`get_logits_from_input_ids`, take the argmax, map it through your
vocabulary, print it. Loop twenty times. You should see plausible
English. If you see garbage, the fault is in step 2, not the model.

**Step 4 — Masking.** Build a boolean array over the vocabulary marking
which tokens are permissible right now. Start with the easy one: tokens
made only of digits and the characters a JSON number allows. Apply it
with `numpy.where(mask, logits, -inf)` and confirm the argmax is now
always a digit. Precompute these masks once at startup — rebuilding a
150,000-entry array per token is the difference between seconds and
minutes.

**Step 5 — One value.** Generate a single number into a slot, then a
single string. This is where the "when does it end" question has to be
answered. Note that character classes alone are not enough for numbers:
`12.3.4`, `007` and `1e` are all made entirely of allowed characters
and all invalid JSON. You need a little state, not just a set.

**Step 6 — Choose the function.** Score the candidate names, pick one.
Watch for the length bug mentioned above; it is easy to introduce and
easy to miss, because the output still looks valid.

**Step 7 — Assemble.** Walk the chosen function's parameters, emit the
fixed JSON text around each slot, fill each slot by type. Feed the
growing text back as context so each decision sees what came before.

**Step 8 — Harden.** Caps on every loop so a stuck generation cannot
hang. A fallback result so one bad prompt still yields a valid entry
rather than a short output array. Then `make lint`, `make lint-strict`,
and the README.

---

## 7. Testing

The subject asks for edge cases explicitly, and warns that **your input
files will be swapped during peer review**. If anything in your code
assumes five functions, or these argument names, or that types are only
`number` and `string`, it will break in front of your evaluator.

There is a ready-made set of awkward inputs in
[data/input/extra_tests/](data/input/extra_tests/) — booleans, a
zero-argument function, huge and negative and leading-zero numbers,
embedded quotes and backslashes and newlines, non-ASCII, unrelated and
empty and injection-flavoured prompts, near-identical function
descriptions, unknown argument types, plus a batch of deliberately
malformed files. Its `README.md` says what each one should do. Point
the CLI at them:

```
uv run python -m src \
  --functions_definition data/input/extra_tests/functions_mixed_types.json \
  --input data/input/extra_tests/prompts_mixed_types.json \
  --output data/output/mixed_types.json
```

One more technique worth the hour it costs: your program should accept
a model object rather than constructing one deep inside itself. Then
you can pass in a **fake model** with the same four public methods,
returning scripted logits. Every piece of logic becomes testable in
milliseconds with no download and no GPU, and it is deterministic, so a
failure is reproducible. Real model runs are for accuracy; the fake is
for correctness.

---

## 8. The README is graded

Do not leave it to the last hour. Beyond the standard 42 sections, the
subject requires:

- **Algorithm explanation** — your constrained decoding, in detail.
- **Design decisions** — and *why*, not just what.
- **Performance analysis** — accuracy, speed, reliability, with actual
  numbers.
- **Challenges faced** — real ones you hit and how you fixed them. The
  vocabulary decoding and the value-termination problem are honest
  answers here.
- **Testing strategy**.
- **Example usage**.

First line, italicised, exactly:
`*This project has been created as part of the 42 curriculum by <login>*`

Written in English. And the `Resources` section must state where you
used AI and for what.

---

## 9. Traps worth knowing about in advance

- **Leading spaces in tokens.** `"hello"` and `" hello"` are different
  tokens with different ids. Mixing them up produces output that is
  nearly right and confusing to debug.
- **Re-encoding vs. splicing.** Token boundaries shift depending on
  neighbouring characters, so gluing pre-encoded fragments together can
  produce a sequence the model never sees during training. Re-encoding
  the whole running string is slightly slower and removes a whole class
  of bug.
- **Leading zeros.** `007` is not valid JSON. Neither is `12.`, `1e`,
  or `+5`. A digit-only mask permits all four.
- **Unbounded loops.** Every generation loop needs a hard cap. The
  subject asks for all prompts in under five minutes.
- **Escaping.** A `"` or `\` inside a generated string value breaks the
  JSON around it. Either forbid those tokens in your mask or escape
  what comes out — but decide which, deliberately.
- **`data/output/` must not be committed.** The subject says so
  explicitly; put it in `.gitignore`.
- **Do not commit a virtualenv.** The reviewer runs `uv sync`.
