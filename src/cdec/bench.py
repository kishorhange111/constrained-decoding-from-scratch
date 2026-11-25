"""Unconstrained vs grammar-constrained JSON generation with a real tokenizer and model.

    python -m cdec.bench --out results [--model Qwen/Qwen2.5-0.5B-Instruct]

There are three schemas (a flat record, nested arrays of objects, a tool call) and 24 chat
prompts each, with greedy decoding of up to 256 tokens, in three modes:
- ``free``: unconstrained. A fenced ```json block is extracted if there is one.
- ``compact``: constrained to compact JSON, with no whitespace anywhere.
- ``ws``: constrained, with JSON whitespace (``[ \\n]*``) allowed between tokens, so the model
  can keep the indented layout it was trained on.

Validity alone is not the whole story: a grammar can force an output that parses but says
nothing. So the run also measures content. For ``tool_call`` it checks that the tool is the
right one for the request and that the date matches the one in the prompt. For every schema
it counts degenerate string values (empty, or a single character) and runs cut off at the
token limit. It also reports index statistics (DFA states, index entries, build time) and
the cost per token of masking.
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


# The right tool for each tool_call subject above.
EXPECTED_TOOL = [
    "get_weather",
    "search_flights",
    "convert_currency",
    "get_weather",
    "search_flights",
    "convert_currency",
    "get_weather",
    "search_flights",
]
MODES = {"free": None, "compact": "", "ws": S.WS}


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


def parse(text: str):
    try:
        return json.loads(extract(text))
    except (json.JSONDecodeError, ValueError):
        return None


def strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)


def content(name: str, i: int, v) -> dict:
    """Content checks on a parsed output ``v`` for prompt ``i``."""
    ss = list(strings(v)) if v is not None else []
    out = {"degenerate": sum(len(x.strip()) <= 1 for x in ss), "n_strings": len(ss)}
    if name == "tool_call":
        subj = SUBJECTS[name][i % len(SUBJECTS[name])]
        out["tool_ok"] = isinstance(v, dict) and v.get("tool") == EXPECTED_TOOL[i % 8]
        m = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", subj)
        if m:
            args = v.get("arguments") if isinstance(v, dict) else None
            out["date_ok"] = isinstance(args, dict) and args.get("date") == m.group(0)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def _fmt(x, spec=""):
    return "–" if x is None else format(x, spec)


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
        for mode, ws in MODES.items():
            dfa = index = None
            compile_s = trie_s = None
            if ws is not None:
                t0 = time.perf_counter()
                dfa = compile_regex(S.to_regex(sch, ws=ws))
                compile_s = time.perf_counter() - t0
                index = TokenIndex.build(dfa, vocab, eos_id=eos)
                if name == "tool_call" and mode == "compact":  # pure-Python trie, timed once
                    trie = TokenIndex.build(dfa, vocab, eos_id=eos, method="trie")
                    assert (trie.table == index.table).all()
                    trie_s = trie.build_seconds
                    del trie
            ok_parse = ok_schema = n_tok = hit_max = 0
            agg = {"degenerate": 0, "n_strings": 0, "tool_ok": [], "date_ok": []}
            gen_s = 0.0
            for i, p in enumerate(prompts(name, a.n)):
                msgs = [{"role": "user", "content": p}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(dev)
                torch.cuda.synchronize() if dev == "cuda" else None
                t1 = time.perf_counter()
                out = generate(model, ids, index, a.max_new, eos)
                torch.cuda.synchronize() if dev == "cuda" else None
                gen_s += time.perf_counter() - t1
                text = tok.decode(out)
                v = parse(text)
                ok_parse += v is not None
                ok_schema += v is not None and S.validate(v, sch, ordered=False)
                n_tok += len(out)
                hit_max += len(out) >= a.max_new
                c = content(name, i, v)
                agg["degenerate"] += c["degenerate"]
                agg["n_strings"] += c["n_strings"]
                for k in ("tool_ok", "date_ok"):
                    if k in c:
                        agg[k].append(c[k])
                if i == 0:
                    examples[f"{name}/{mode}"] = text
            row = {
                "schema": name,
                "mode": mode,
                "parse_rate": ok_parse / a.n,
                "schema_valid_rate": ok_schema / a.n,
                "mean_tokens": n_tok / a.n,
                "ms_per_token": gen_s / max(n_tok, 1) * 1e3,
                "hit_max_new": hit_max / a.n,
                "degenerate_strings": agg["degenerate"],
                "n_strings": agg["n_strings"],
                "tool_acc": _mean(agg["tool_ok"]),
                "date_acc": _mean(agg["date_ok"]),
                "dfa_states": dfa.n_states if dfa else None,
                "compile_s": compile_s,
                "index_build_s": index.build_seconds if index else None,
                "index_entries": index.n_entries if index else None,
                "trie_build_s": trie_s,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del index
    dev_name = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
    lines = [
        f"{a.model} on **{dev_name}**, greedy, ≤ {a.max_new} new tokens, {a.n} prompts per schema."
        f" Vocabulary: {len(vocab):,} tokens (bytes extracted in {vocab_s:.1f} s).",
        "",
        "| schema | mode | parses | satisfies schema | degenerate strings | right tool |"
        " right date | hit token limit | mean tokens | ms/token | DFA states | index build (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['schema']} | {r['mode']} | {r['parse_rate']:.0%} |"
            f" {r['schema_valid_rate']:.0%} | {r['degenerate_strings']}/{r['n_strings']} |"
            f" {_pct(r['tool_acc'])} | {_pct(r['date_acc'])} | {r['hit_max_new']:.0%} |"
            f" {r['mean_tokens']:.0f} | {r['ms_per_token']:.1f} | {_fmt(r['dfa_states'])} |"
            f" {_fmt(r['index_build_s'], '.1f')} |"
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
