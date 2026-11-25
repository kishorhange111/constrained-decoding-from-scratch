"""Unconstrained vs grammar-constrained JSON generation with a real tokenizer and model.

    python -m cdec.bench --out results [--model Qwen/Qwen2.5-0.5B-Instruct]

There are three schemas (a flat record, nested arrays of objects, a tool call) and 24 chat
prompts each, with greedy decoding of up to 256 tokens. For the unconstrained output, a
fenced ```json block is extracted if there is one, then the run counts how often the text
parses and how often it satisfies the schema (key order ignored, to be fair). For the
constrained output, the run reports the same, plus the index statistics (DFA states, index
entries, build time) and the cost per token of masking.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch

from . import schema as S
from .decode import generate
from .index import TokenIndex, token_bytes
from .regex import compile_regex

SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "employed": {"type": "boolean"},
        },
        "required": ["name", "age", "email", "skills", "employed"],
    },
    "order": {
        "type": "object",
        "properties": {
            "order_id": {"type": "integer", "minimum": 0},
            "status": {"enum": ["pending", "shipped", "delivered", "cancelled"]},
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "qty": {"type": "integer", "minimum": 0},
                        "price": {"type": "number"},
                    },
                    "required": ["sku", "qty", "price"],
                },
            },
        },
        "required": ["order_id", "status", "items"],
    },
    "tool_call": {
        "type": "object",
        "properties": {
            "tool": {"enum": ["get_weather", "search_flights", "convert_currency"]},
            "arguments": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string", "pattern": r"[0-9]{4}-[0-9]{2}-[0-9]{2}"},
                    "amount": {"type": "number"},
                },
                "required": [],
            },
        },
        "required": ["tool", "arguments"],
    },
}

SUBJECTS = {
    "person": [
        "a software engineer",
        "a retired teacher",
        "a nurse",
        "a chef",
        "a student",
        "a pilot",
        "a musician",
        "a data scientist",
    ],
    "order": [
        "three books",
        "a laptop and a mouse",
        "garden tools",
        "two phone cases",
        "coffee beans",
        "a bicycle",
        "kitchen knives",
        "office chairs",
    ],
    "tool_call": [
        "What's the weather in Paris tomorrow (2025-03-14)?",
        "Find flights to Tokyo on 2025-07-01.",
        "Convert 250 dollars to euros.",
        "Is it raining in Mumbai today, 2025-06-20?",
        "Flights to Berlin on 2025-12-24",
        "How much is 99.5 USD in yen?",
        "Weather for Sydney on 2025-01-05",
        "Book a flight to Lima, 2025-09-09",
    ],
}


def prompts(name: str, n: int) -> list[str]:
    sch = json.dumps(SCHEMAS[name])
    subj = SUBJECTS[name]
    out = []
    for i in range(n):
        s = subj[i % len(subj)]
        if name == "tool_call":
            task = f"User request: {s}\nChoose a tool and its arguments."
        else:
            task = f"Create a realistic JSON record for {s} (example #{i + 1})."
        out.append(f"{task}\nReply with a single JSON object matching this JSON Schema:\n{sch}")
    return out


def extract(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def check(text: str, schema: dict) -> tuple[bool, bool]:
    try:
        v = json.loads(extract(text))
    except (json.JSONDecodeError, ValueError):
        return False, False
    return True, S.validate(v, schema, ordered=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=256)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
    eos = tok.eos_token_id
    t0 = time.perf_counter()
    vocab = token_bytes(tok)
    vocab_s = time.perf_counter() - t0
    rows, examples = [], {}
    for name, sch in SCHEMAS.items():
        t0 = time.perf_counter()
        dfa = compile_regex(S.to_regex(sch))
        compile_s = time.perf_counter() - t0
        index = TokenIndex.build(dfa, vocab, eos_id=eos)
        trie_s = None
        if name == "tool_call":  # the pure-Python trie builder, timed once for comparison
            trie = TokenIndex.build(dfa, vocab, eos_id=eos, method="trie")
            assert (trie.table == index.table).all()
            trie_s = trie.build_seconds
        for mode in ("free", "constrained"):
            ok_parse = ok_schema = n_tok = 0
            gen_s = 0.0
            for i, p in enumerate(prompts(name, a.n)):
                msgs = [{"role": "user", "content": p}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(dev)
                torch.cuda.synchronize() if dev == "cuda" else None
                t1 = time.perf_counter()
                out = generate(model, ids, index if mode == "constrained" else None, a.max_new, eos)
                torch.cuda.synchronize() if dev == "cuda" else None
                gen_s += time.perf_counter() - t1
                text = tok.decode(out)
                pp, ss = check(text, sch)
                ok_parse += pp
                ok_schema += ss
                n_tok += len(out)
                if i == 0:
                    examples[f"{name}/{mode}"] = text
            row = {
                "schema": name,
                "mode": mode,
                "parse_rate": ok_parse / a.n,
                "schema_valid_rate": ok_schema / a.n,
                "mean_tokens": n_tok / a.n,
                "ms_per_token": gen_s / max(n_tok, 1) * 1e3,
                "dfa_states": dfa.n_states,
                "compile_s": compile_s,
                "index_build_s": index.build_seconds,
                "index_entries": index.n_entries,
                "trie_build_s": trie_s,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    dev_name = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
    lines = [
        f"{a.model} on **{dev_name}**, greedy, ≤ {a.max_new} new tokens, {a.n} prompts per schema."
        f" Vocabulary: {len(vocab):,} tokens (bytes extracted in {vocab_s:.1f} s).",
        "",
        "| schema | mode | parses | satisfies schema | mean tokens | ms/token |"
        " DFA states | index build (s) | index entries |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['schema']} | {r['mode']} | {r['parse_rate']:.0%} |"
            f" {r['schema_valid_rate']:.0%} |"
            f" {r['mean_tokens']:.0f} | {r['ms_per_token']:.1f} | {r['dfa_states']} |"
            f" {r['index_build_s']:.1f} | {r['index_entries']:,} |"
        )
    tr = next(r for r in rows if r["trie_build_s"] is not None)
    lines += [
        "",
        f"Index build for `tool_call` ({tr['dfa_states']} states × {len(vocab):,} tokens):"
        f" vectorized {tr['index_build_s']:.1f} s,"
        f" pure-Python trie walk {tr['trie_build_s']:.1f} s.",
        "",
        "### State blowup from bounded repetition: one string with `maxLength = k`",
        "",
        "| k | DFA states | index build (s) | index table (MiB) |",
        "|---|---|---|---|",
    ]
    blowup = []
    for k in (None, 8, 16, 32, 64):
        sch = {"type": "string"} if k is None else {"type": "string", "maxLength": k}
        d = compile_regex(S.to_regex(sch))
        ix = TokenIndex.build(d, vocab, eos_id=eos)
        b = {"maxLength": k, "states": d.n_states, "build_s": ix.build_seconds}
        b["table_MiB"] = ix.table.nbytes / 2**20
        blowup.append(b)
        kk = "∞" if k is None else k
        lines.append(f"| {kk} | {b['states']} | {b['build_s']:.1f} | {b['table_MiB']:.0f} |")
        del ix
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"rows": rows, "blowup": blowup, "examples": examples}, f, indent=1, ensure_ascii=False
        )
    print(text)


if __name__ == "__main__":
    main()
