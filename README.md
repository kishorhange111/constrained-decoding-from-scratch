# constrained-decoding-from-scratch

[![CI](https://github.com/kishorhange111/constrained-decoding-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/constrained-decoding-from-scratch/actions/workflows/ci.yml)

Make an LLM emit JSON that **always** parses and matches a schema by masking, at every
decoding step, the tokens that would leave the grammar. Libraries such as Outlines,
XGrammar and llguidance do this. Here every layer is written by hand:

```
JSON Schema ──► regex ──► Thompson NFA ──► DFA (subset construction) ──► pruned, minimal DFA
                                                                              │
                         vocabulary (151k byte strings) ──► token index: table[state, token]
                                                                              │
                                    decoding: logits[~allowed(state)] = −∞, state ← table[state, tok]
```

## Pieces

**[`regex.py`](src/cdec/regex.py): a regex engine over bytes.** It works on bytes, not
characters, because tokens are byte strings and a token can end mid-way through a UTF-8
character. It has a recursive-descent parser (classes, escapes, groups, `| * + ?`,
`{m,n}`), Thompson's NFA construction and subset construction to a DFA. Two further passes
follow:
- **pruning** removes every state from which no accepting state is reachable. The decoder
  depends on this: any token that leads to a live state can always be completed.
- **Moore minimisation** merges equivalent states.

**[`schema.py`](src/cdec/schema.py): JSON Schema to regex.** It covers objects with fixed
key order and optional keys, strings (with length bounds and `pattern`), integers,
numbers, booleans, null, enum and const, and arrays with item counts. A string's contents
are matched as **valid UTF-8 byte sequences** (RFC 3629, with no overlongs or surrogates),
so a decoded output never fails `bytes.decode`. By default the
regex describes compact JSON. With `to_regex(schema, ws=WS)`, whitespace (`[ \n]*`) is also
allowed between structural characters, so the model can write the indented layout it was
trained on rather than being forced off it.

**[`index.py`](src/cdec/index.py): the token index.** For every DFA state and every
vocabulary token, it records where the token's bytes lead, or −1 if the DFA dies. There are
two builders, which a test checks give identical tables:
- a **byte trie** of the vocabulary, walked depth-first per state, which prunes a subtree as
  soon as the DFA dies;
- a **vectorized** builder that advances all (state, token) pairs one byte position at a
  time with NumPy.

The table is dense (int16 when it fits). On the GPU, a step's mask is then one row of a
cached boolean matrix, with no host transfer.

**[`decode.py`](src/cdec/decode.py): masked greedy or sampled decoding.** EOS is allowed only
in accepting states. A state can be live in the DFA yet have no token that continues it: a
test hit this with a toy vocabulary that couldn't spell a key. Real byte-level
vocabularies contain all 256 single bytes, so it cannot happen there.

## Tests

There are 29, on CPU:
- **The regex engine agrees with Python's `re.fullmatch`** on 16 patterns. Each is checked
  on random strings and on random walks of the DFA, so positive cases are covered too.
- The textbook `(a|b)*abb` minimises to 4 states.
- UTF-8 literals and the empty language work, and bad patterns raise.
- The schema regex accepts valid instances and rejects 8 kinds of invalid ones (missing,
  reordered or extra keys, wrong type, bounds, enum).
- The whitespace grammar accepts both compact and `indent=2` JSON, but never whitespace
  inside a key, number or literal.
- **Every random string in a schema's language parses with `json.loads` and validates.**
- The vectorized and trie indexes are identical and match a brute-force walk over every
  (state, token) pair.
- Masks are padded to the model's vocabulary, and EOS is allowed exactly in accepting
  states.
- **A random-weight GPT-2, sampling at temperature 1 under the constraint, always produces
  valid JSON**, while unconstrained it doesn't.

```
pytest -q                            # 29 tests, CPU
python -m cdec.bench --out results   # Qwen2.5-0.5B-Instruct: free vs constrained, index stats
```

## Results (Tesla T4, Colab)

Qwen2.5-0.5B-Instruct, greedy, 24 prompts per schema, ≤ 256 new tokens. The output is in
[`results/bench.md`](results/bench.md), and [`results/bench.json`](results/bench.json)
includes an example output per schema and mode.

| schema | mode | parses | satisfies schema | degenerate strings | right tool | mean tokens | ms/token | DFA states |
|---|---|---|---|---|---|---|---|---|
| person | free | 100% | 100% | 3/142 | – | 56 | 29.1 | – |
| person | compact | 100% | 100% | **76/157** | – | 32 | 29.0 | 161 |
| person | ws | 100% | 100% | 3/136 | – | 51 | 29.4 | 161 |
| order | free | 100% | 100% | 0/114 | – | 151 | 28.0 | – |
| order | compact | 100% | 100% | 0/120 | – | 92 | 27.5 | 259 |
| order | ws | 100% | 100% | 0/115 | – | 150 | 27.8 | 263 |
| tool_call | free | 100% | 100% | 0/72 | 100% | 49 | 27.8 | – |
| tool_call | compact | 100% | 100% | 6/72 | **75%** | 33 | 27.5 | 125 |
| tool_call | ws | 100% | 100% | 0/72 | 100% | 49 | 27.9 | 126 |

"Degenerate" means a string value that is empty or a single character. The date in every
tool call was right in all three modes.

**A grammar that guarantees validity can still damage the content.** On these schemas and
prompts, the 0.5B model already produced valid JSON 100% of the time without any
constraint, so the constraint had nothing to fix. Restricting it to *compact* JSON then made
things worse:
- On `person`, **48% of string values became degenerate** (76 of 157, against 3 of 142
  free). A typical output was `{"name":".","age":0,"email":"",…}`.
- On `tool_call`, it picked the right tool only **75%** of the time, against 100% free.
  For example, it called `convert_currency` for a Paris weather request.

The model writes JSON in the indented form it was trained on. The compact grammar masks
every newline and space, which pushes it onto token sequences it rarely saw, and the
content drifts.

**Allowing whitespace between tokens (`ws`) removes the damage.** Degenerate strings drop
to 3 of 136 and tool choice returns to 100%. For each schema, the saved example (the first prompt)
is character-for-character the free model's own answer, minus the ```` ```json ```` fence. The
fix costs at most 4 DFA states (259 → 263) and no measurable time per token.

Other measurements:
- **Masking is free at this scale.** Time per token is 27.5–29.4 ms in every mode. The
  mask is one row of a cached boolean matrix on the GPU.
- **Index build** takes 1.5–3.8 s per schema over 151,665 tokens. The pure-Python trie walk
  took 2.0 s against the vectorized builder's 1.5 s for `tool_call`, so the two are
  comparable on these schemas.
- **`maxLength` multiplies states.** One string with `maxLength = k` goes from 15 states
  (unbounded) to 835 at k = 64, and the index table from 4 MiB to 242 MiB. Bounded
  repetition is the expensive part of a JSON-schema grammar.
