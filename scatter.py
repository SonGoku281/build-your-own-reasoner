"""
Build Your Own Reasoner — the complete experiment.

Samples N independent Chain-of-Thought chains per problem (temperature > 0),
parses the answer out of each chain, and lets the majority vote decide.
Compares this "self-consistency" strategy against greedy decoding (n=1, temp=0)
on a set of word problems with known answers.

Pipeline (every box is a component you can swap):
    problem
       │
       ▼
  GENERATOR ──n chains──▶ PARSER ──answers──▶ AGGREGATOR ──winner──▶ VERIFIER
  (ask_llm)               (regex)            (vote)        (vs ground truth)

Configuration comes from .env (see .env.example) — no hardcoded values here.

Usage:
    cp .env.example .env      # edit to taste
    python3 scatter.py
"""

import json
import re
import urllib.request
from collections import Counter

import config
from problems import PROBLEMS

# ── 1. GENERATOR — the only LLM component ────────────────────────────────
TEMPLATE = """{problem}

End your response with exactly: Answer: <number>.
Nothing after that line."""


def ask_llm(prompt: str, temperature: float) -> str:
    """One completion call against the configured endpoint."""
    payload = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": config.MAX_TOKENS,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


# ── 2. PARSER — extract the number from free-form prose ───────────────────
# Pattern: "Answer:", then anything (lazy), then capture the digits.
# Lazy .*? is critical: greedy .* would grab everything and leave (\d+)
# only the LAST single digit ("44" → "4"). Ask us how we know.
ANSWER_PATTERN = re.compile(r"Answer:.*?(\d+)")


def parse(answer: str):
    """Return the answer number as a string, or None if the model disobeyed."""
    match = ANSWER_PATTERN.search(answer)
    if match is None:
        return None
    return match.group(1)


# ── 3. AGGREGATOR — the vote ──────────────────────────────────────────────
def solve_one(problem: str, n: int, temperature: float) -> tuple:
    """Generate n chains, parse, vote. Return (winner, vote_breakdown)."""
    votes: Counter = Counter()
    for _ in range(n):
        answer = ask_llm(TEMPLATE.format(problem=problem), temperature)
        candidate = parse(answer)
        if candidate is not None:          # guard clause: skip garbage, don't crash
            votes[candidate] += 1
    if not votes:
        return None, {}
    return votes.most_common(1)[0][0], dict(votes)


# ── 4. VERIFIER — compare against ground truth (eval mode) ────────────────
def verify(candidate, ground_truth) -> bool:
    """Normalized comparison: '44' == '44.0' == '44 ' == 44."""
    if candidate is None:
        return False
    try:
        return float(candidate) == float(ground_truth)
    except ValueError:
        return str(candidate).strip() == str(ground_truth).strip()


# ── THE EXPERIMENT ────────────────────────────────────────────────────────
def main() -> None:
    print(f"model={config.MODEL} | endpoint={config.BASE_URL} | "
          f"SC: n={config.SC_SAMPLES} temp={config.SC_TEMPERATURE} | "
          f"greedy: temp={config.GREEDY_TEMPERATURE}\n")

    greedy_correct = sc_correct = 0
    for i, (problem, truth) in enumerate(PROBLEMS, 1):
        g_winner, _ = solve_one(problem, n=1, temperature=config.GREEDY_TEMPERATURE)
        s_winner, s_votes = solve_one(problem, n=config.SC_SAMPLES,
                                      temperature=config.SC_TEMPERATURE)
        g_ok, s_ok = verify(g_winner, truth), verify(s_winner, truth)
        greedy_correct += g_ok
        sc_correct += s_ok
        print(f"[{i:2d}] truth={truth:>4} | greedy={g_winner} {'✓' if g_ok else '✗'} | "
              f"SC={s_winner} {'✓' if s_ok else '✗'} votes={s_votes}")

    n = len(PROBLEMS)
    print(f"\ngreedy: {greedy_correct}/{n} = {greedy_correct/n:.0%}")
    print(f"SC:     {sc_correct}/{n} = {sc_correct/n:.0%}")


if __name__ == "__main__":
    main()
