import json
import random
import re

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cdec import schema as S
from cdec.decode import generate
from cdec.index import TokenIndex, bytes_to_unicode
from cdec.regex import RegexError, compile_regex, determinize, prune, to_nfa

PATTERNS = [
    r"abc",
    r"a|b|cd",
    r"(ab)*c",
    r"a+b?c*",
    r"[a-c]{2,3}x",
    r"x{3}",
    r"(?:ab|a)(?:bc|c)",
    r"[^ab]+",
    r"\d+(\.\d+)?",
    r"(a|b)*abb",
    r"\w{1,4}\s?\w*",
    r"[-a]z",
    r"a.c",
    r"(a{0,2}b){2,}",
    r"",
    r"(|a)b",
]
ALPHABET = "abcxz0123 .-_\n"


def random_strings(rng, n=400, max_len=7):
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_len))) for _ in range(n)]


def sample_accepted(dfa, rng, n=50, max_len=40):
    """Random walks through the DFA that stop at accepting states."""
    out = []
    for _ in range(n * 20):
        s, bs = dfa.start, []
        for _ in range(max_len):
            if dfa.accept[s] and rng.random() < 0.3:
                out.append(bytes(bs))
                break
            choices = [b for b in range(256) if dfa.trans[s][b] >= 0]
            if not choices:
                if dfa.accept[s]:
                    out.append(bytes(bs))
                break
            b = rng.choice(choices)
            bs.append(b)
            s = dfa.trans[s][b]
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- regex engine


@pytest.mark.parametrize("pat", PATTERNS)
def test_regex_agrees_with_python_re(pat):
    rng = random.Random(hash(pat) % 1000)
    dfa = compile_regex(pat)
    rx = re.compile(pat, re.DOTALL if False else 0)
    for s in random_strings(rng):
        assert dfa.fullmatch(s.encode()) == (rx.fullmatch(s) is not None), (pat, s)
    for bs in sample_accepted(dfa, rng):  # positives, which random strings rarely hit
        assert rx.fullmatch(bs.decode("latin-1")) is not None, (pat, bs)


def test_minimal_dfa_for_textbook_example():
    # (a|b)*abb: the minimal DFA has 4 states (dragon book), plus nothing dead after pruning
    raw = prune(determinize(to_nfa("(a|b)*abb")))
    assert compile_regex("(a|b)*abb").n_states == 4 <= raw.n_states


def test_utf8_literals_and_empty_language():
    dfa = compile_regex("é+")
    assert dfa.fullmatch("éé".encode()) and not dfa.fullmatch("é".encode()[:1])
    assert compile_regex("[^\\x00-\\xff]").n_states == 1  # nothing matches: dead start


def test_bad_patterns_raise():
    for bad in ["(ab", "*a", "a{3,1}", "[z-a]"]:
        with pytest.raises(RegexError):
            compile_regex(bad)


# ---------------------------------------------------------------- JSON schema

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"},
        "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3},
        "score": {"type": "number"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
}


def test_schema_accepts_valid_and_rejects_invalid():
    dfa = compile_regex(S.to_regex(PERSON))
    good = [
        {"name": "Ann", "age": 31, "active": True},
        {"name": 'Bo "x"', "age": 0, "email": "b@x.io", "active": False},
        {"name": "", "age": 7, "tags": ["a", "c"], "score": -1.5e3, "active": True},
        {"name": "Zoë ✓", "age": 12, "tags": [], "active": False},
    ]
    bad = [
        {"name": "Ann", "active": True},  # missing required
        {"age": 3, "name": "Ann", "active": True},  # wrong order
        {"name": "Ann", "age": -1, "active": True},  # minimum
        {"name": "Ann", "age": 1.5, "active": True},  # not an integer
        {"name": "x" * 13, "age": 1, "active": True},  # maxLength
        {"name": "A", "age": 1, "tags": ["d"], "active": True},  # enum
        {"name": "A", "age": 1, "tags": ["a"] * 4, "active": True},  # maxItems
        {"name": "A", "age": 1, "active": True, "extra": 1},  # closed object
    ]
    for v in good:
        assert S.validate(v, PERSON) and dfa.fullmatch(S.dumps(v).encode()), v
    for v in bad:
        assert not S.validate(v, PERSON) and not dfa.fullmatch(S.dumps(v).encode()), v


def test_every_string_in_the_schema_language_parses_and_validates():
    dfa = compile_regex(S.to_regex(PERSON))
    rng = random.Random(0)
    samples = sample_accepted(dfa, rng, n=60, max_len=400)
    assert len(samples) >= 20
    parsed = 0
    for bs in samples:
        try:
            v = json.loads(bs.decode("utf-8"))
        except UnicodeDecodeError:
            continue  # the byte-level grammar allows any byte ≥ 0x80 inside strings
        assert S.validate(v, PERSON), bs
        parsed += 1
    assert parsed >= 10


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4},
        {
            "type": "object",
            "properties": {"a": {"type": "null"}, "b": {"const": 3}},
            "required": [],
        },
        {"enum": [1, "x", None, False]},
    ],
)
def test_small_schemas_roundtrip(schema):
    dfa = compile_regex(S.to_regex(schema))
    for bs in sample_accepted(dfa, random.Random(1), n=30, max_len=60):
        assert S.validate(json.loads(bs), schema), bs


# ---------------------------------------------------------------- token index + decoding


def toy_vocab(rng, n=300):
    alphabet = b'{}[]":,0123456789abcdefghijklmnopqrstuvwxyz-. '
    vocab = [bytes([c]) for c in alphabet]
    while len(vocab) < n - 1:
        vocab.append(bytes(rng.choice(alphabet) for _ in range(rng.randint(2, 6))))
    return vocab + [b""]  # last id: EOS (no bytes)


def test_token_index_matches_brute_force():
    rng = random.Random(2)
    vocab = toy_vocab(rng)
    dfa = compile_regex(S.to_regex(PERSON))
    vec = TokenIndex.build(dfa, vocab, eos_id=len(vocab) - 1)
    trie = TokenIndex.build(dfa, vocab, eos_id=len(vocab) - 1, method="trie")
    assert (vec.table == trie.table).all()
    for s in range(dfa.n_states):
        for t, bs in enumerate(vocab):
            want = dfa.walk(s, bs) if bs else -1
            assert vec.table[s, t] == want


def test_mask_pads_model_vocab_and_allows_eos_only_when_accepting():
    vocab = [b"a", b"b", b""]
    dfa = compile_regex("ab?")
    idx = TokenIndex.build(dfa, vocab, eos_id=2)
    m0 = idx.mask(dfa.start, 5)
    assert m0.tolist() == [True, False, False, False, False]
    s1 = idx.advance(dfa.start, 0)
    assert idx.mask(s1, 5).tolist() == [False, True, True, False, False]


def test_constrained_decoding_always_yields_valid_json():
    rng = random.Random(3)
    vocab = toy_vocab(rng)
    eos = len(vocab) - 1
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=512, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()
    schema = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["id", "ok"],
    }
    idx = TokenIndex.build(compile_regex(S.to_regex(schema)), vocab, eos_id=eos)
    g = torch.Generator().manual_seed(0)
    for _ in range(8):
        toks = generate(model, torch.tensor([[0]]), idx, 200, eos, temperature=1.0, generator=g)
        text = b"".join(vocab[t] for t in toks)
        assert S.validate(json.loads(text), schema), text
        # without the index, the random model essentially never produces valid JSON
    free = generate(model, torch.tensor([[0]]), None, 40, eos)
    assert not compile_regex(S.to_regex(schema)).fullmatch(b"".join(vocab[t] for t in free))


def test_bytes_to_unicode_is_a_bijection():
    m = bytes_to_unicode()
    assert sorted(m) == list(range(256)) and len(set(m.values())) == 256
