# Build Your Own Reasoner

**A hands-on tutorial on Chain of Thought (CoT) and self-consistency — from first principles to a working implementation.**

You don't need a GPU, a research lab, or a big budget. You need a Python interpreter, any OpenAI-compatible LLM endpoint, and about an afternoon. By the end, you'll have written — line by line, understanding every character — a system that samples multiple reasoning chains and votes on the answer: the same core idea behind reasoning models like o1 and R1.

This tutorial was written the way the code was actually built: one concept at a time, with the bugs left in the story, because **the debugging is the education.**

---

## Table of contents

1. [What is Chain of Thought?](#1-what-is-chain-of-thought)
2. [Why one chain isn't enough](#2-why-one-chain-isnt-enough)
3. [The generator — sampling with temperature](#3-the-generator--sampling-with-temperature)
4. [The parser — regex, and the bugs you'll hit](#4-the-parser--regex-and-the-bugs-youll-hit)
5. [The aggregator + verifier — the vote](#5-the-aggregator--verifier--the-vote)
6. [The experiment — greedy vs. self-consistency](#6-the-experiment--greedy-vs-self-consistency)
7. [What this means in production](#7-what-this-means-in-production)
8. [Run it yourself](#8-run-it-yourself)

---

## 1. What is Chain of Thought?

### The machine underneath

An LLM is an **autocomplete machine**. It reads text and predicts the *next word* (technically, the next token). Every word it writes becomes part of the input for the next word. That's all it does — a long chain of single-word predictions.

Now ask it a math problem. It has two ways to answer:

1. **Blurt it out** — predict the answer word directly: `"What's 17 × 23?"` → `"391"`. The entire "computation" has to happen inside one word prediction. It's guessing, not computing.
2. **Show its work** — write out steps, one word at a time, and the answer falls out at the end.

**Chain of Thought = making the model write out its reasoning steps before the final answer.**

```
Problem: A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball.
         How much does the ball cost?

Without CoT:    "10 cents"                                    ← wrong, but fast
With CoT:       "Let the ball cost x. The bat costs x + 1.00.
                 x + (x + 1.00) = 1.10 → 2x = 0.10 → x = 0.05"  ← right
```

The simplest CoT prompt in existence is seven words: **"Let's think step by step."** That's it. That's the whole trick at its most basic.

### Why it works

Each intermediate step becomes *text the model can read* before predicting the next step. It's the difference between doing long division in your head vs. on paper: the paper isn't smarter than you — it **holds your working memory**.

### Why doesn't the model do this by default?

Because of how it's trained. The model is trained to predict the next word on a giant pile of internet text — and on the internet, questions are mostly followed by **answers**, not worked solutions. So statistically, the most likely next word after a question is the start of a direct answer. The model *can* reason (the ability is in its weights, from all the textbooks in its training data) — but nothing activates it by default. The prompt is the switch.

And "let's think step by step" works because the model has seen that exact phrase millions of times in training, always followed by step-by-step solutions. It's a statistical cue that flips the model into "textbook mode" — and once the first step is written, the chain self-sustains (each step conditions the next).

> **The one insight to internalize:** modern models often reason *by default* — they were trained to. The prompt version is the hack; training it in is the real fix (that's what o1/R1 did). But the *machinery* you'll build in this tutorial is what makes reasoning *dependable*, which matters regardless of the model.

---

## 2. Why one chain isn't enough

One chain is one **opinion from a smart-but-sloppy expert**. The model predicts probabilistically — at temperature > 0, the same question produces different chains each time, and some will be wrong. Trust one chain and you're betting everything on one roll of the dice.

But here's the beautiful pattern (this is **self-consistency**, Wang et al., 2022):

> Run the same problem 8 times. **Correct chains converge on the same answer** — there's only one right answer, so all correct paths reach it. **Wrong chains scatter** — wrongness is creative; each failure fails differently.

So when you count votes, the majority is right even though individual chains weren't:

```
Chain 1: 44      Chain 5: 44
Chain 2: 42      Chain 6: 44
Chain 3: 44      Chain 7: 44
Chain 4: 48      Chain 8: 44
                → 44 wins 6–1–1
```

The math backs this up (Condorcet's jury theorem, 1785): a group of voters each only slightly better than random becomes near-certain as the group grows. Your model is the jury; the chains are the jurors.

**The critical catch:** voting only works if the chains are actually *different*. At temperature 0, the model gives the same chain every time — one opinion repeated N times. **Temperature creates diversity; diversity makes voting meaningful.**

---

## 3. The generator — sampling with temperature

The generator is the only component that touches the LLM. Everything else is classical algorithms. This is the first thing you write — a function that calls an OpenAI-compatible endpoint:

```python
import json
import urllib.request

def ask_llm(prompt: str, temperature: float) -> str:
    payload = {
        "model": "gemma-4-12b-v2",          # any model your endpoint serves
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,          # 0.0 = greedy, 0.7+ = diverse
        "max_tokens": 8192,                  # reasoning models think before they answer!
        "stream": False,
    }
    req = urllib.request.Request(
        "http://localhost:8081/v1/chat/completions",   # llama.cpp / vLLM / any
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode()
    data = json.loads(body)
    return data["choices"][0]["message"]["content"]
```

### The experiment that teaches temperature

Run the same problem 5 times at `temperature=0.7` and print all 5 answers. Then change to `temperature=0.0` and run again.

- **At 0.7:** you get 5 *different* chains. Some right, some wrong — the scatter, live.
- **At 0.0:** you get 5 *identical* chains. Deterministic. Boring. Useless for voting.

This is the entire temperature mental model, proven in two runs:

| Temperature | Behavior | Voting value |
|---|---|---|
| 0.0 | Always the most-likely token; deterministic | None — one opinion × N |
| 0.7 | Weighted dice-rolling; diverse chains | The raw material of voting |

### The token budget trap (real-world gotcha)

Reasoning models emit **hidden thinking** (in OpenAI-compatible APIs, the `reasoning_content` field) *before* the visible answer — and that thinking counts against `max_tokens`. If `max_tokens` is too small (2048), the model spends its entire budget thinking and produces **an empty answer**. Symptoms: `finish_reason: "length"`, empty `content`, and a parser that crashes on `None`.

**Fix:** give the model room to think *and* answer. 8192 is a safe starting point. (You can measure your chains' actual token usage and tune this — that's the engineering habit.)

---

## 4. The parser — regex, and the bugs you'll hit

The model's output is prose with the answer buried inside it. You need the *number*, not the sentence. This is where ~30% of real-world effort goes.

### First bug: `find('Answer:', -1)` is not "search from the end"

```python
index = answer.find('Answer:', -1)   # WRONG
```

`str.find(sub, start)` — the second argument is a **starting position**, not a direction. `-1` as a start means "start at the last character," so the search window is 1 character wide, and a 7-character string like `Answer:` can never match. It returns `-1` (not found). The tool you want is **`rfind`** (search from the end). Or, better, skip manual indexing entirely and use regex:

```python
import re

def parse(answer: str):
    match = re.search(r'Answer:.*?(\d+)', answer)
    if match is None:
        return None          # model disobeyed — signal it, don't crash
    return match.group(1)    # '44' — just the number
```

### Second bug: greedy `.*` eats your answer

This is the most common regex mistake in the world, and it produces *silently wrong* data:

```python
re.search(r'Answer:.*(\d+)', "Answer: 44").group(1)   # → '4'  WRONG!
re.search(r'Answer:.*?(\d+)', "Answer: 44").group(1)  # → '44' RIGHT!
```

`.*` (greedy) matches *as much as possible*, then `(\d+)` gets whatever scraps are left — the **last single digit**. `.*?` (lazy) matches *as little as possible*, so `(\d+)` gets the whole number. The `?` is one character; the difference is `4` vs `44`.

> **Greedy `.*` grabs the whole plate and leaves the last bite. Lazy `.*?` takes only what it needs and lets the next part eat properly.**

### The `None` guard

`re.search` returns `None` when nothing matches. Calling `.group()` on `None` crashes with `AttributeError`. The parser must survive model disobedience — return `None`, and let the *voter* skip it:

```python
if match is None:
    return None
```

---

## 5. The aggregator + verifier — the vote

### The aggregator: count and pick the winner

```python
def solve_one(problem: str, n: int, temperature: float) -> tuple:
    """Generate n chains, parse, vote. Return (winner, vote_breakdown)."""
    votes = {}
    for _ in range(n):
        answer = ask_llm(TEMPLATE.format(problem=problem), temperature)
        candidate = parse(answer)
        if candidate is not None:                    # guard clause: skip garbage
            votes[candidate] = votes.get(candidate, 0) + 1
    if not votes:
        return None, {}
    return max(votes, key=votes.get), votes          # winner, breakdown
```

Two subtle things here:

- `votes.get(candidate, 0)` — the dict-counter idiom: "give me the current count, or 0 if new." Replaces a whole if/else.
- `max(votes, key=votes.get)` — the `key=` tells max *how to score* candidates. Without it, `max(dict, dict.get)` compares a dict to a function → `TypeError`. This exact bug crashed our first version.

### Normalization — the hidden trap

`"44"` and `"44.0"` and `"44 "` are *different strings* but the *same answer*. If you vote with raw strings, a true 6-vote majority can look like a 3-3 tie. **Normalize before comparing:**

```python
def verify(candidate, ground_truth) -> bool:
    if candidate is None:
        return False
    try:
        return float(candidate) == float(ground_truth)
    except ValueError:
        return candidate.strip() == ground_truth.strip()
```

The `try` survives non-numeric garbage (a model that votes a stray word). This is the **verifier** — in this tutorial it compares against known answers (eval mode). In production, it's replaced by something stronger — see [section 7](#7-what-this-means-in-production).

---

## 6. The experiment — greedy vs. self-consistency

The payoff. Same model, same 14 problems, only the decoding strategy differs:

| Strategy | Sampling | Cost per problem |
|---|---|---|
| greedy | n=1, temperature=0.0 | 1 call |
| self-consistency | n=5, temperature=0.9 | 5 calls |

```python
for i, (problem, truth) in enumerate(PROBLEMS, 1):
    g_winner, _ = solve_one(problem, n=1, temperature=0.0)
    s_winner, s_votes = solve_one(problem, n=5, temperature=0.9)
    g_ok, s_ok = verify(g_winner, truth), verify(s_winner, truth)
    ...
```

**Make a prediction before you run it.** That's what separates an experiment from a demo. Which strategy wins? Which problems are tricky?

### Reading the results

```
[ 1] truth=  84 | greedy=84 ✓ | SC=84 ✓ votes={'84': 5}
[ 4] truth=  15 | greedy=15 ✓ | SC=15 ✓ votes={'15': 5}
[ 9] truth=  42 | greedy=42 ✓ | SC=42 ✓ votes={'42': 5}
```

- `votes={'84': 5}` — perfect consensus, all five chains agree
- `votes={'44': 3, '42': 2}` — the scatter! Majority saved the day
- `votes={'44': 3, '42': 2}` where the winner is *wrong* — consensus on wrongness (rare; the failure mode voting can't fix)

### The model-size lesson

The interesting experiment isn't just one model — it's **two models of very different sizes**. Here's why: self-consistency fixes *mistakes*, and small models make more mistakes.

| Model | Expectation |
|---|---|
| Large (12B+) | Both greedy and SC nail easy problems → no gap. The problems are too easy; voting has nothing to fix. |
| Small (3-4B) | Greedy stumbles on multi-step problems → SC's majority vote recovers → **the gap appears**. |

This is the honest finding from running this tutorial: **voting buys the most exactly where the model is weakest.** On a strong model with easy problems, the delta is zero — which isn't a failure of the method, it's a measurement of the problem set. To see the method shine, give the small model hard problems.

---

## 7. What this means in production

The aggregator is not something you run on every request — it's a **tool you deploy selectively**, and it's only one rung of the verification ladder.

### The verifier ladder

In the lab, the verifier compares against a known answer. In production, **the system itself is the verifier** — and there's a ladder, from cheap to expensive:

1. **Execution** — make the answer *runnable* and run it. Code → tests. SQL → the database. Math → a calculator. The environment itself checks.
2. **Consensus** — the vote we built. The agreement of many chains *is* evidence; wrong answers scatter. Its output is **confidence** (votes ÷ total).
3. **Judgment** — another model call grades the chain (LLM-as-judge). Expensive, imperfect, but works on any question type.
4. **Abstention** — when nothing agrees: **don't answer.** "I'm not sure" beats a confident hallucination every time. Low confidence → escalate to a human.

Think about real-world systems: a customer-service agent that creates orders, books appointments, files claims. It doesn't verify answers against a key — it calls the system, reads the response, and checks the *resulting state*. That *is* verification against reality. The reasoner pattern is just a more disciplined version of the same instinct: **a system that checks its own work.**

### Where the aggregator belongs (and where it doesn't)

- ✅ **Interpretations** — an ambiguous customer email: parse it 5 ways, majority wins on what they actually want
- ✅ **Plans** — generate 3 delivery plans, score them, pick the best (this is the seed of tree-of-thoughts search)
- ✅ **Low-volume, high-stakes decisions** — a rare decision where being wrong is expensive
- ❌ **The hot path** — don't sample 8× on every step of a long workflow; you'll burn your budget in a week

**The production pattern is risk routing:** greedy by default; aggregate when the task is flagged high-risk or ambiguous; escalate when even the vote is unsure. The router is business rules + confidence thresholds.

### Cost engineering (if you do aggregate)

1. **Parallelize** — fire the N calls concurrently (thread pool, or a server-side `n` parameter). The real cost is tokens, not time.
2. **Adaptive sampling** — stop at 3 samples if 3 agree; only go to 8 on a split.
3. **Cheap samples, expensive judge** — small fast model generates candidates; a bigger model scores them.

### Agents are deterministic code around a stochastic model

The deepest reframe: **the LLM has no memory — your code builds its context fresh on every call.** The agent is deterministic code (the loop, the guards, the routing) calling a stochastic model (the judgment). This is why guard clauses, invariants, and plan-review gates matter more than prompt cleverness: **deterministic invariants first (cheap, certain), LLM judgment second (fuzzy), escalation last (safe).**

---

## 8. Run it yourself

### Requirements

- Python 3.8+ (stdlib only — no pip installs needed)
- Any OpenAI-compatible LLM endpoint. Examples:
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) server (`llama-server -m model.gguf --port 8081`)
  - [vLLM](https://github.com/vllm-project/vllm)
  - OpenAI / any hosted API (change the `base_url` and `model`)
- A model that can do arithmetic (a 7B+ instruct model works; reasoning models work great but need the bigger `max_tokens`)

### Configuration

All settings live in `.env` — nothing is hardcoded. Copy the example and edit:

```bash
cp .env.example .env
# edit: BASE_URL, MODEL, MAX_TOKENS, SC_SAMPLES, SC_TEMPERATURE, ...
```

| Setting | Default | What it controls |
|---|---|---|
| `BASE_URL` | `http://localhost:8081/v1` | Your OpenAI-compatible endpoint |
| `MODEL` | `gemma-4-12b-v2` | Which model to query |
| `MAX_TOKENS` | `8192` | Budget per completion — reasoning models think *before* they answer, and thinking counts against this |
| `SC_TEMPERATURE` | `0.9` | Diversity for the vote (must be > 0) |
| `SC_SAMPLES` | `5` | Chains per problem for the vote |
| `GREEDY_TEMPERATURE` | `0.0` | Deterministic baseline |

### Run

```bash
# 1. point .env at your endpoint (BASE_URL + MODEL)
# 2. run the scatter experiment (temperature lesson):
python3 -c "
from scatter import ask_llm
for i in range(5):
    print(ask_llm('2+2?', 0.7))"     # try 0.0 vs 0.7 — see the difference

# 3. run the full experiment:
python3 scatter.py
```

### Files

| File | What it is |
|---|---|
| `scatter.py` | The complete reasoner: generator, parser, aggregator, verifier, experiment |
| `config.py` | Dependency-free `.env` loader — every knob is configurable |
| `.env.example` | All configuration knobs, documented |
| `problems.py` | 14 word problems with exact answers (the eval set — extend it!) |

### Ideas to take it further

- **Harder problems** — add problems the model actually fails sometimes; that's where the SC-vs-greedy gap shows
- **Few-shot CoT** — add 2-3 worked examples to the prompt (free accuracy boost)
- **Tree of Thoughts** — instead of voting on whole chains, score *steps* and search (the next rung)
- **Train your own reasoner** — fine-tune a small model with RL on verifiable rewards (GRPO, R1-style) and watch the "aha moment" emerge in your own training logs

---

## Troubleshooting

This tutorial was built by actually hitting every bug below — they're the real failure modes, in the order you'll meet them.

### 1. Empty answers, `finish_reason: "length"`, parser crashes on `None`

**Symptom:** the model returns nothing (or gets cut off mid-sentence), and your parser throws `AttributeError: 'NoneType' object has no attribute 'group'`.

**Cause:** reasoning models emit hidden thinking (`reasoning_content`) *before* the visible answer — and that thinking counts against `max_tokens`. A budget of 2048 gets eaten by thinking; the answer never gets written.

**Fix:** raise `MAX_TOKENS` (8192 is a safe starting point). Measure your chains' actual usage and tune from there.

### 2. The vote says `4` when the answer is `44`

**Symptom:** silently wrong results — the most dangerous kind. No crash, just wrong.

**Cause:** greedy regex. `r'Answer:.*(\d+)'` — the `.*` matches as much as possible, so `(\d+)` is left only the *last single digit*. Lazy `.*?` fixes it: `r'Answer:.*?(\d+)'`.

### 3. `max(areas, areas.get)` → `TypeError`

**Cause:** `max(a, b)` compares two values; you passed a dict and a method. The `key=` argument tells max *how to score*: `max(areas, key=areas.get)`.

### 4. `find('Answer:', -1)` never matches

**Cause:** `str.find(sub, start)` — the second argument is a *starting position*, not "search from the end." `-1` as a start means "start at the last character." Use `rfind` for last-occurrence, or just use regex.

### 5. HTTP 502 / connection errors mid-run

**Symptom:** the experiment dies partway with `HTTPError: Bad Gateway`.

**Cause:** the serving process (llama.cpp / llama-swap / vLLM) restarted or reloaded mid-request — commonly because *you* edited its config, or the model was unloaded for VRAM. The 502 is the server saying "I was busy changing."

**Fix:** finish your config edits *before* starting the run; re-run after the server settles. (We hit this one live — the first run lost its last problem to a config reload.)

### 6. `re.search` matches but `group(1)` is the wrong number

**Symptom:** the parse returns a number from the reasoning text instead of the final answer.

**Cause:** the model wrote `Answer:` but the regex grabbed digits elsewhere. Make the prompt demand the answer on the final line, and keep the lazy pattern anchored to the marker. In production, constrain the output format (JSON / structured output) so parsing is trivial.

### 7. Same output every time, voting does nothing

**Cause:** `temperature=0.0` — greedy decoding is deterministic; N samples are N copies of the same opinion. The vote needs diversity: `SC_TEMPERATURE` must be > 0.

---

## How this tutorial was built

This repo started as a live tutoring session: one person learning Chain of Thought from zero, writing every line themselves, with an AI coach. The bugs in the Troubleshooting section weren't invented — they're the actual bugs hit during that session, including:

- the greedy-regex `4` vs `44` bug (found because the vote said "4")
- the `max_tokens` reasoning-budget trap (found because chains came back empty)
- the 502 on config reload (found because the experiment died on its last problem)
- the download-filename fiasco (three attempts before the right file name matched)

The tutorial keeps them because **the debugging is the education.** If you hit a bug not listed here, open an issue — the session that built this is proof that hitting bugs is the point.

---

## References

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al., 2022. The origin paper: few-shot rationales unlock math/commonsense reasoning in large models.
- **Large Language Models are Zero-Shot Reasoners** — Kojima et al., 2022. "Let's think step by step" — CoT without examples.
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** — Wang et al., 2022. The paper this tutorial implements: sample N chains, majority vote.
- **STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., 2022. Generate rationales, fine-tune on the correct ones.
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** — Yao et al., 2023. Search over steps, not just one chain.
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI, 2025. GRPO, the "aha moment," and the distilled reasoning models many people run locally.
- **Condorcet's jury theorem** (1785) — the mathematics behind why voting works: a group of slightly-better-than-random voters becomes near-certain as it grows.

---

## Acknowledgements

- Built and tested against [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap) serving local models (Gemma 4 12B and Qwen 3 4B) — no cloud API required for the whole tutorial.
- The eval problems are original (written for this tutorial) to avoid copyright issues with benchmark sets.

---

## License

MIT — use it, learn from it, build on it.
