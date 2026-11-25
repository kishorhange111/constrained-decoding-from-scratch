Qwen/Qwen2.5-0.5B-Instruct on **Tesla T4**, greedy, ≤ 256 new tokens, 24 prompts per schema. Vocabulary: 151,665 tokens (bytes extracted in 0.3 s).

| schema | mode | parses | satisfies schema | degenerate strings | right tool | right date | hit token limit | mean tokens | ms/token | DFA states | index build (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| person | free | 100% | 100% | 3/142 | – | – | 0% | 56 | 29.1 | – | – |
| person | compact | 100% | 100% | 76/157 | – | – | 0% | 32 | 29.0 | 161 | 2.0 |
| person | ws | 100% | 100% | 3/136 | – | – | 0% | 51 | 29.4 | 161 | 1.9 |
| order | free | 100% | 100% | 0/114 | – | – | 0% | 151 | 28.0 | – | – |
| order | compact | 100% | 100% | 0/120 | – | – | 0% | 92 | 27.5 | 259 | 3.3 |
| order | ws | 100% | 100% | 0/115 | – | – | 0% | 150 | 27.8 | 263 | 3.8 |
| tool_call | free | 100% | 100% | 0/72 | 100% | 100% | 0% | 49 | 27.8 | – | – |
| tool_call | compact | 100% | 100% | 6/72 | 75% | 100% | 0% | 33 | 27.5 | 125 | 1.5 |
| tool_call | ws | 100% | 100% | 0/72 | 100% | 100% | 0% | 49 | 27.9 | 126 | 1.5 |

Index build for `tool_call` (125 states × 151,665 tokens): vectorized 1.5 s, pure-Python trie walk 2.0 s.

### State blowup from bounded repetition: one string with `maxLength = k`

| k | DFA states | index build (s) | index table (MiB) |
|---|---|---|---|
| ∞ | 15 | 0.4 | 4 |
| 8 | 107 | 1.4 | 31 |
| 16 | 211 | 2.3 | 61 |
| 32 | 419 | 5.2 | 121 |
| 64 | 835 | 9.3 | 242 |
