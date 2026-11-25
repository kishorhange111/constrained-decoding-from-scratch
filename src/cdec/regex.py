"""A regex engine over **bytes**: parser → Thompson NFA → DFA (subset construction) →
minimal DFA (Moore partition refinement), with dead-end states pruned.

Bytes rather than characters, because LLM tokens are byte strings. A GPT-2 token can end
in the middle of a UTF-8 character, so the automaton has to be able to stop there too.
Non-ASCII literals in a pattern are expanded to their UTF-8 byte sequence.

Supported syntax (always a *full* match):
- literals and escapes: ``\\.``, ``\\\\``, ``\\n``, ``\\t``, ``\\xHH``;
- classes ``\\d \\w \\s`` (and ``\\D \\W \\S``), ``[a-z_]`` and ``[^"\\\\]``;
- ``.`` (any byte except newline);
- groups ``( )`` and ``(?: )``, alternation ``|``;
- quantifiers ``* + ?``, ``{m}``, ``{m,}`` and ``{m,n}``.

Byte sets are Python ints used as 256-bit masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALL = (1 << 256) - 1


def _mask(*bytes_: int) -> int:
    m = 0
    for b in bytes_:
        m |= 1 << b
    return m


def _range(lo: int, hi: int) -> int:
    return ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)


DIGIT = _range(48, 57)
WORD = DIGIT | _range(65, 90) | _range(97, 122) | _mask(95)
SPACE = _mask(32, 9, 10, 13, 12, 11)
CLASS_ESC = {"d": DIGIT, "w": WORD, "s": SPACE}
CHAR_ESC = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "0": 0}


# ---------------------------------------------------------------- parser -> AST
# Nodes: ("set", mask) | ("cat", [n]) | ("alt", [n]) | ("rep", n, lo, hi_or_None) | ("eps",)


class RegexError(ValueError):
    pass


class _Parser:
    def __init__(self, pattern: str):
        self.p, self.i = pattern, 0

    def peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def take(self):
        c = self.peek()
        if c is None:
            raise RegexError(f"unexpected end of pattern {self.p!r}")
        self.i += 1
        return c

    def parse(self):
        node = self.alt()
        if self.peek() is not None:
            raise RegexError(f"unexpected {self.peek()!r} at {self.i} in {self.p!r}")
        return node

    def alt(self):
        opts = [self.cat()]
        while self.peek() == "|":
            self.i += 1
            opts.append(self.cat())
        return opts[0] if len(opts) == 1 else ("alt", opts)

    def cat(self):
        items = []
        while self.peek() not in (None, "|", ")"):
            items.append(self.repeat())
        if not items:
            return ("eps",)
        return items[0] if len(items) == 1 else ("cat", items)

    def repeat(self):
        node = self.atom()
        while True:
            c = self.peek()
            if c == "*":
                self.i += 1
                node = ("rep", node, 0, None)
            elif c == "+":
                self.i += 1
                node = ("rep", node, 1, None)
            elif c == "?":
                self.i += 1
                node = ("rep", node, 0, 1)
            elif c == "{" and self._is_brace():
                lo, hi = self._brace()
                node = ("rep", node, lo, hi)
            else:
                return node

    def _is_brace(self):
        j = self.p.find("}", self.i)
        body = self.p[self.i + 1 : j] if j > 0 else ""
        parts = body.split(",")
        return (
            j > 0
            and 1 <= len(parts) <= 2
            and parts[0].isdigit()
            and (len(parts) == 1 or parts[1] == "" or parts[1].isdigit())
        )

    def _brace(self):
        j = self.p.index("}", self.i)
        parts = self.p[self.i + 1 : j].split(",")
        self.i = j + 1
        lo = int(parts[0])
        hi = lo if len(parts) == 1 else (None if parts[1] == "" else int(parts[1]))
        if hi is not None and hi < lo:
            raise RegexError(f"bad repeat {{{lo},{hi}}}")
        return lo, hi

    def atom(self):
        c = self.take()
        if c == "(":
            if self.p.startswith("?:", self.i):
                self.i += 2
            node = self.alt()
            if self.take() != ")":
                raise RegexError("missing )")
            return node
        if c == "[":
            return ("set", self._class())
        if c == ".":
            return ("set", ALL & ~_mask(10))
        if c == "\\":
            return self._escape(in_class=False)
        if c in "*+?)":
            raise RegexError(f"nothing to repeat / unbalanced at {self.i - 1} in {self.p!r}")
        return _literal(c)

    def _escape(self, in_class: bool):
        c = self.take()
        if c.lower() in CLASS_ESC:
            m = CLASS_ESC[c.lower()]
            return ("set", m if c.islower() else ALL & ~m)
        if c == "x":
            h = self.take() + self.take()
            return ("set", _mask(int(h, 16)))
        if c in CHAR_ESC:
            return ("set", _mask(CHAR_ESC[c]))
        return _literal(c)

    def _class(self) -> int:
        neg = self.peek() == "^"
        if neg:
            self.i += 1
        m, first = 0, True
        while True:
            c = self.take()
            if c == "]" and not first:
                break
            first = False
            if c == "\\":
                node = self._escape(in_class=True)
                if node[0] != "set":
                    raise RegexError("multi-byte escape in class")
                lo_mask = node[1]
                if self.peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("]", ""):
                    lo = lo_mask.bit_length() - 1
                    if lo_mask != 1 << lo:
                        raise RegexError("range from a class escape")
                    m |= self._range_to(lo)
                else:
                    m |= lo_mask
                continue
            if ord(c) > 127:
                raise RegexError("only ASCII is supported inside [...]")
            if self.peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("]", ""):
                m |= self._range_to(ord(c))
            else:
                m |= _mask(ord(c))
        return (ALL & ~m) if neg else m

    def _range_to(self, lo: int) -> int:
        self.i += 1  # '-'
        c = self.take()
        if c == "\\":
            node = self._escape(in_class=True)
            hi = node[1].bit_length() - 1
        else:
            hi = ord(c)
        if hi < lo or hi > 255:
            raise RegexError(f"bad range {lo}-{hi}")
        return _range(lo, hi)


def _literal(ch: str):
    bs = ch.encode("utf-8")
    nodes = [("set", _mask(b)) for b in bs]
    return nodes[0] if len(nodes) == 1 else ("cat", nodes)


def parse(pattern: str):
    return _Parser(pattern).parse()


# ---------------------------------------------------------------- Thompson NFA


@dataclass
class NFA:
    eps: list[list[int]] = field(default_factory=list)
    trans: list[list[tuple[int, int]]] = field(default_factory=list)  # (mask, target)
    start: int = 0
    accept: int = 0

    def new(self) -> int:
        self.eps.append([])
        self.trans.append([])
        return len(self.eps) - 1


def _build(nfa: NFA, node) -> tuple[int, int]:
    kind = node[0]
    if kind == "eps":
        s = nfa.new()
        return s, s
    if kind == "set":
        s, t = nfa.new(), nfa.new()
        nfa.trans[s].append((node[1], t))
        return s, t
    if kind == "cat":
        s, t = _build(nfa, node[1][0])
        for child in node[1][1:]:
            s2, t2 = _build(nfa, child)
            nfa.eps[t].append(s2)
            t = t2
        return s, t
    if kind == "alt":
        s, t = nfa.new(), nfa.new()
        for child in node[1]:
            s2, t2 = _build(nfa, child)
            nfa.eps[s].append(s2)
            nfa.eps[t2].append(t)
        return s, t
    if kind == "rep":
        _, child, lo, hi = node
        s = t = nfa.new()
        for _ in range(lo):  # mandatory copies
            s2, t2 = _build(nfa, child)
            nfa.eps[t].append(s2)
            t = t2
        if hi is None:  # Kleene star on one more copy
            s2, t2 = _build(nfa, child)
            loop_end = nfa.new()
            nfa.eps[t].extend([s2, loop_end])
            nfa.eps[t2].extend([s2, loop_end])
            t = loop_end
        else:
            end = nfa.new()
            for _ in range(hi - lo):  # optional copies, each may skip to the end
                s2, t2 = _build(nfa, child)
                nfa.eps[t].extend([s2, end])
                t = t2
            nfa.eps[t].append(end)
            t = end
        return s, t
    raise ValueError(kind)


def to_nfa(pattern: str) -> NFA:
    nfa = NFA()
    nfa.start, nfa.accept = _build(nfa, parse(pattern))
    return nfa


# ---------------------------------------------------------------- DFA


@dataclass
class DFA:
    """``trans[s][b]`` is the next state or −1 (dead). ``accept[s]`` marks final states.

    Every non-dead state is *live*: an accepting state is reachable from it. The constrained
    decoder relies on this, because a token that leads to a live state can always be completed.
    """

    trans: list[list[int]]
    accept: list[bool]
    start: int

    @property
    def n_states(self) -> int:
        return len(self.trans)

    def walk(self, s: int, data: bytes) -> int:
        for b in data:
            if s < 0:
                return -1
            s = self.trans[s][b]
        return s

    def fullmatch(self, data: bytes) -> bool:
        s = self.walk(self.start, data)
        return s >= 0 and self.accept[s]


def _closure(nfa: NFA, states) -> frozenset:
    stack, seen = list(states), set(states)
    while stack:
        for t in nfa.eps[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return frozenset(seen)


def determinize(nfa: NFA) -> DFA:
    start = _closure(nfa, [nfa.start])
    ids, sets, trans = {start: 0}, [start], []
    i = 0
    while i < len(sets):
        S = sets[i]
        moves = [(m, t) for s in S for m, t in nfa.trans[s]]
        row = [-1] * 256
        cache: dict[frozenset, int] = {}
        for b in range(256):
            bit = 1 << b
            targets = frozenset(t for m, t in moves if m & bit)
            if not targets:
                continue
            if targets not in cache:
                T = _closure(nfa, targets)
                if T not in ids:
                    ids[T] = len(sets)
                    sets.append(T)
                cache[targets] = ids[T]
            row[b] = cache[targets]
        trans.append(row)
        i += 1
    return DFA(trans, [nfa.accept in S for S in sets], 0)


def prune(dfa: DFA) -> DFA:
    """Drop states from which no accepting state is reachable, and unreachable states."""
    n = dfa.n_states
    rev: list[set[int]] = [set() for _ in range(n)]
    for s, row in enumerate(dfa.trans):
        for t in row:
            if t >= 0:
                rev[t].add(s)
    live = {s for s in range(n) if dfa.accept[s]}
    stack = list(live)
    while stack:
        for p in rev[stack.pop()]:
            if p not in live:
                live.add(p)
                stack.append(p)
    if dfa.start not in live:  # empty language: a single dead start state
        return DFA([[-1] * 256], [False], 0)
    reach, stack = {dfa.start}, [dfa.start]
    while stack:
        for t in dfa.trans[stack.pop()]:
            if t in live and t not in reach:
                reach.add(t)
                stack.append(t)
    keep = sorted(reach)
    remap = {s: i for i, s in enumerate(keep)}
    trans = [[remap.get(t, -1) for t in dfa.trans[s]] for s in keep]
    return DFA(trans, [dfa.accept[s] for s in keep], remap[dfa.start])


def minimize(dfa: DFA) -> DFA:
    """Moore's algorithm: split blocks by (accepting, block of each successor) until stable."""
    block = [int(a) for a in dfa.accept]
    while True:
        sig = {}
        new = []
        for s in range(dfa.n_states):
            key = (block[s], tuple(block[t] if t >= 0 else -1 for t in dfa.trans[s]))
            new.append(sig.setdefault(key, len(sig)))
        stable = len(sig) == len(set(block))
        block = new  # compact labels 0..k-1
        if stable:
            break
    n = len(set(block))
    trans = [[-1] * 256 for _ in range(n)]
    accept = [False] * n
    for s in range(dfa.n_states):
        b = block[s]
        accept[b] = dfa.accept[s]
        trans[b] = [block[t] if t >= 0 else -1 for t in dfa.trans[s]]
    return DFA(trans, accept, block[dfa.start])


def compile_regex(pattern: str) -> DFA:
    return minimize(prune(determinize(to_nfa(pattern))))
