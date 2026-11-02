"""Built-in, seedable prompt suites.

Prompts are fixed strings (plus a generated long-context body) so two
machines running the same seed compare the same bytes. We do not pull
prompts from the network.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from llmbench.config import BenchConfig
from llmbench.exceptions import ConfigError

# Compact vocabulary for the long-context generator. Deterministic given seed.
_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "meter",
    "newton",
    "orbit",
    "photon",
    "quark",
    "radian",
    "silicon",
    "tensor",
    "volt",
    "watt",
    "xray",
    "yield",
    "zenith",
    "vector",
    "matrix",
    "field",
    "gradient",
    "laplacian",
    "curl",
    "flux",
    "ampere",
    "coulomb",
    "farad",
    "henry",
    "joule",
    "kelvin",
    "lumen",
    "mole",
    "pascal",
    "steradian",
    "algorithm",
    "compiler",
    "runtime",
    "kernel",
    "scheduler",
    "cache",
    "latency",
    "throughput",
    "token",
    "context",
    "window",
    "attention",
    "logits",
    "softmax",
    "residual",
    "quantize",
    "prefill",
    "decode",
    "stream",
    "batch",
    "concurrent",
]


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    suite: str
    text: str
    max_tokens: int | None = None


SHORT_CHAT: tuple[Prompt, ...] = (
    Prompt("chat-01", "short_chat", "In one sentence, what is a photon?"),
    Prompt("chat-02", "short_chat", "List three properties of an ideal gas."),
    Prompt("chat-03", "short_chat", "Explain time dilation to a high-school student."),
    Prompt("chat-04", "short_chat", "What is the difference between heat and temperature?"),
    Prompt("chat-05", "short_chat", "Summarize Kirchhoff's circuit laws."),
    Prompt("chat-06", "short_chat", "Give a short definition of entropy in statistical mechanics."),
    Prompt(
        "chat-07",
        "short_chat",
        "Why does a spinning ice skater speed up when they pull their arms in?",
    ),
    Prompt("chat-08", "short_chat", "What does it mean for a system to be in thermal equilibrium?"),
    Prompt("chat-09", "short_chat", "Name two differences between fission and fusion."),
    Prompt("chat-10", "short_chat", "Describe constructive interference in one paragraph."),
    Prompt("chat-11", "short_chat", "What is a local minimum in an optimization landscape?"),
    Prompt(
        "chat-12",
        "short_chat",
        "Translate this to simpler language: 'the Hamiltonian generates time evolution'.",
    ),
)

CODE: tuple[Prompt, ...] = (
    Prompt(
        "code-01",
        "code",
        "Write a Python function that returns the nth Fibonacci number iteratively. Include a docstring.",
        192,
    ),
    Prompt(
        "code-02",
        "code",
        "Explain this function and point out a bug:\n\ndef mean(xs):\n    return sum(xs) / len(xs) or 1\n",
        192,
    ),
    Prompt(
        "code-03",
        "code",
        "Write a typed Python function `percentile(xs: list[float], p: float) -> float` using linear interpolation.",
        256,
    ),
    Prompt(
        "code-04",
        "code",
        "Given a binary tree node class, write a function that returns the maximum depth. Python.",
        192,
    ),
    Prompt(
        "code-05",
        "code",
        "Write a bash one-liner to count unique source IPs in an nginx access log.",
        96,
    ),
    Prompt(
        "code-06",
        "code",
        "Refactor this into a context manager:\n\nf = open('x')\ndata = f.read()\nf.close()\n",
        128,
    ),
)


def load_suite(name: str, cfg: BenchConfig) -> list[Prompt]:
    if name == "short_chat":
        return list(SHORT_CHAT)
    if name == "code":
        return list(CODE)
    if name == "long_context":
        return [_long_context(cfg)]
    raise ConfigError(f"unknown suite {name!r}")


def split_suite(
    cfg: BenchConfig, suite: str, warmup: int, iterations: int
) -> tuple[list[Prompt], list[Prompt]]:
    """One shuffled sequence: warmup is the prefix, measured is the rest.

    Calling ``iter_prompts`` twice restarted at index 0, so the first
    measured trials were the warmup prompts (prefix-cache hits).
    ``long_context`` is a single body; each trial gets a unique *prefix*
    nonce so llama.cpp / Ollama / vLLM cannot reuse the KV cache.
    """

    total = warmup + iterations
    seq = prompt_sequence(cfg, suite, total)
    if suite == "long_context":
        seq = [_cache_bust(p, i, cfg.seed) for i, p in enumerate(seq)]
    return seq[:warmup], seq[warmup:]


def _long_context(cfg: BenchConfig) -> Prompt:
    rng = random.Random(cfg.seed)
    target = max(2_000, cfg.long_context_chars)
    parts: list[str] = [
        "You are given a long technical passage. After the passage, answer: "
        "what is the last numbered marker in the text?\n\n"
    ]
    n = 0
    while sum(len(p) for p in parts) < target:
        n += 1
        words = [rng.choice(_WORDS) for _ in range(48)]
        parts.append(f"[{n:04d}] " + " ".join(words) + ".\n")
    parts.append(f"\nQuestion: what is the last numbered marker? It should be [{n:04d}].")
    return Prompt("long-01", "long_context", "".join(parts), max_tokens=32)


def iter_prompts(cfg: BenchConfig, suite: str, count: int) -> list[Prompt]:
    """Cycle the suite in seed order. Prefer ``split_suite`` for a run."""

    return prompt_sequence(cfg, suite, count)


def prompt_sequence(cfg: BenchConfig, suite: str, count: int) -> list[Prompt]:
    pool = load_suite(suite, cfg)
    rng = random.Random(cfg.seed + _stable_int(suite))
    order = list(pool)
    rng.shuffle(order)
    if not order:
        raise ConfigError(f"suite {suite} has no prompts")
    return [order[i % len(order)] for i in range(count)]


def _cache_bust(prompt: Prompt, index: int, seed: int) -> Prompt:
    """Unique prefix so prefix-cache cannot reuse the long body."""

    nonce = hashlib.sha256(f"{seed}:{index}:{prompt.prompt_id}".encode()).hexdigest()[:16]
    header = f"[cache-bust trial={index} nonce={nonce}]\n\n"
    return Prompt(
        prompt_id=f"{prompt.prompt_id}-t{index:04d}",
        suite=prompt.suite,
        text=header + prompt.text,
        max_tokens=prompt.max_tokens,
    )


def _stable_int(text: str) -> int:
    # hashlib, not hash(), so PYTHONHASHSEED cannot scramble suite order.
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
