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
so a decoded output never fails `bytes.decode`.

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

There are 28, on CPU:
- **The regex engine agrees with Python's `re.fullmatch`** on 16 patterns. Each is checked
  on random strings and on random walks of the DFA, so positive cases are covered too.
- The textbook `(a|b)*abb` minimises to 4 states.
- UTF-8 literals and the empty language work, and bad patterns raise.
- The schema regex accepts valid instances and rejects 8 kinds of invalid ones (missing,
  reordered or extra keys, wrong type, bounds, enum).
- **Every random string in a schema's language parses with `json.loads` and validates.**
- The vectorized and trie indexes are identical and match a brute-force walk over every
  (state, token) pair.
- Masks are padded to the model's vocabulary, and EOS is allowed exactly in accepting
  states.
- **A random-weight GPT-2, sampling at temperature 1 under the constraint, always produces
  valid JSON**, while unconstrained it doesn't.

```
pytest -q                            # 28 tests, CPU
python -m cdec.bench --out results   # Qwen2.5-0.5B-Instruct: free vs constrained, index stats
```

**Results are pending** from the Colab run.
