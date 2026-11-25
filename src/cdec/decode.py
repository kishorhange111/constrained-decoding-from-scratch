"""Masked decoding: at every step, logits of tokens the grammar forbids are set to −∞.

The output is guaranteed to be in the language: every allowed token leads to a live DFA
state, and EOS is allowed only in accepting states. If an accepting state has no
continuation at all, decoding stops there.
"""

from __future__ import annotations

import torch

from .index import TokenIndex


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,  # (1, P)
    index: TokenIndex | None,
    max_new: int,
    eos_id: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Greedy (temperature 0) or sampled decoding, constrained if ``index`` is given."""
    out = model(prompt_ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1]
    state = index.dfa.start if index is not None else None
    toks: list[int] = []
    for _ in range(max_new):
        if index is not None:
            m = index.mask(state, logits.shape[-1], logits.device)
            if not m.any():
                break  # accepting state with nothing allowed after it, and no EOS id
            logits = logits.masked_fill(~m, float("-inf"))
        if temperature > 0:
            p = torch.softmax(logits.float() / temperature, -1)
            t = int(torch.multinomial(p, 1, generator=generator))
        else:
            t = int(logits.argmax())
        if t == eos_id:
            break
        toks.append(t)
        if index is not None:
            state = index.advance(state, t)
            if index.dfa.accept[state] and not index.has_continuation(state):
                break
        out = model(torch.tensor([[t]], device=prompt_ids.device), past_key_values=past)
        past, logits = out.past_key_values, out.logits[0, -1]
    return toks
