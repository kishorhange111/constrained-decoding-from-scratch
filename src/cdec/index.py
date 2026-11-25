"""The token index: for every DFA state, which vocabulary tokens are allowed and where they lead.

A token is allowed in state s if walking its bytes through the DFA from s never hits the
dead state. Because the DFA is pruned, every surviving state can still reach an accepting
one, so any allowed token keeps the output completable. This is the "index" of Outlines
(Willard & Louf 2023). It is stored as a dense table ``table[s, t]`` holding the next state,
or −1. On the GPU the allowed-token mask for a step is then one row of ``table >= 0``,
with no host–device traffic during decoding.

There are two builders, which give identical tables:
- **trie**: the vocabulary is arranged into a byte trie. Each state walks it depth-first,
  abandoning a subtree the moment the DFA dies, and tokens sharing a prefix share work. The
  cost is pure Python per trie node visited, which is slow when a state allows almost every
  token (inside a JSON string).
- **vectorized**: all (state, token) pairs advance together, one byte position at a time,
  with NumPy fancy indexing into the transition table (plus an absorbing dead row). The
  cost is ``states × vocab × max token length`` simple integer ops, with no Python per token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .regex import DFA


class _Node:
    __slots__ = ("children", "tokens")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.tokens: list[int] = []


def build_trie(vocab: list[bytes]) -> _Node:
    root = _Node()
    for tid, bs in enumerate(vocab):
        if not bs:  # special / empty tokens never match grammar bytes
            continue
        node = root
        for b in bs:
            node = node.children.setdefault(b, _Node())
        node.tokens.append(tid)
    return root


def _table_trie(dfa: DFA, vocab: list[bytes]) -> np.ndarray:
    table = np.full((dfa.n_states, len(vocab)), -1, dtype=np.int32)
    root = build_trie(vocab)
    for s in range(dfa.n_states):
        stack = [(root, s)]
        row = table[s]
        while stack:
            node, q = stack.pop()
            tr = dfa.trans[q]
            for b, child in node.children.items():
                q2 = tr[b]
                if q2 < 0:
                    continue
                for tid in child.tokens:
                    row[tid] = q2
                if child.children:
                    stack.append((child, q2))
    return table


def _table_vectorized(dfa: DFA, vocab: list[bytes]) -> np.ndarray:
    S, V = dfa.n_states, len(vocab)
    dead = S
    T = np.array(dfa.trans, dtype=np.int32)
    T[T < 0] = dead
    T = np.vstack([T, np.full((1, 256), dead, dtype=np.int32)])  # absorbing dead row
    lengths = np.array([len(b) for b in vocab])
    L = int(lengths.max()) if V else 0
    byte_mat = np.zeros((V, max(L, 1)), dtype=np.int32)
    for t, bs in enumerate(vocab):
        byte_mat[t, : len(bs)] = np.frombuffer(bs, dtype=np.uint8)
    cur = np.repeat(np.arange(S, dtype=np.int32)[:, None], V, axis=1)
    order = np.argsort(-lengths, kind="stable")  # longest first: active tokens are a prefix
    cur = cur[:, order]
    byte_mat, lengths_sorted = byte_mat[order], lengths[order]
    for j in range(L):
        n_active = int((lengths_sorted > j).sum())
        sub = cur[:, :n_active]
        cur[:, :n_active] = T[sub, byte_mat[:n_active, j][None, :]]
    cur[cur == dead] = -1
    cur[:, lengths_sorted == 0] = -1
    table = np.empty_like(cur)
    table[:, order] = cur
    return table


@dataclass
class TokenIndex:
    dfa: DFA
    table: np.ndarray  # (states, vocab) int16/int32: next state, or −1 if forbidden
    eos_id: int | None
    build_seconds: float
    _masks: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, dfa: DFA, vocab: list[bytes], eos_id=None, method="vectorized") -> TokenIndex:
        t0 = time.perf_counter()
        fn = {"vectorized": _table_vectorized, "trie": _table_trie}[method]
        table = fn(dfa, vocab)
        if dfa.n_states < np.iinfo(np.int16).max:
            table = table.astype(np.int16)  # halves memory; −1 still marks forbidden
        return cls(dfa, table, eos_id, time.perf_counter() - t0)

    def allowed(self, state: int) -> np.ndarray:
        return np.nonzero(self.table[state] >= 0)[0]

    def has_continuation(self, state: int) -> bool:
        return bool((self.table[state] >= 0).any())

    def mask(self, state: int, vocab_size: int, device=None) -> torch.Tensor:
        """Boolean mask over the model's vocabulary (which may be padded beyond the
        tokenizer's). EOS is allowed exactly in accepting states."""
        key = (str(device), vocab_size)
        if key not in self._masks:
            m = torch.zeros(self.table.shape[0], vocab_size, dtype=torch.bool)
            m[:, : self.table.shape[1]] = torch.from_numpy(self.table >= 0)
            if self.eos_id is not None:
                m[:, self.eos_id] = torch.tensor(self.dfa.accept)
            self._masks[key] = m.to(device)
        return self._masks[key][state]

    def advance(self, state: int, token: int) -> int:
        return int(self.table[state, token])

    @property
    def n_entries(self) -> int:
        return int((self.table >= 0).sum())


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte → printable-character map (used by byte-level BPE vocabularies)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


def token_bytes(tokenizer) -> list[bytes]:
    """Raw bytes of every token of a byte-level BPE tokenizer (GPT-2, Qwen, Llama-3 style).

    Special/added tokens map to ``b""`` so they are never allowed by the grammar.
    """
    dec = {c: b for b, c in bytes_to_unicode().items()}
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}))
    out = []
    for tid in range(len(tokenizer)):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tid in special or tok is None:
            out.append(b"")
            continue
        try:
            out.append(bytes(dec[c] for c in tok))
        except KeyError:  # not a byte-level token
            out.append(b"")
    return out
