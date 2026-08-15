import json
import urllib.request
import re
from problems import PROBLEMS

MODEL = "gemma-4-12b-v2"
TEMPLATE = """{problem}

End your response with exactly: Answer: <number>.
Nothing after that line."""

def ask_llm(prompt: str, temperature: float) -> str:
    payload = {
        "model": "gemma-4-12b-v2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False,
    }
    req = urllib.request.Request(
        "http://localhost:8081/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode()
    data = json.loads(body)
    return data["choices"][0]["message"]["content"]

def parse(answer: str):
    match = re.search(r'Answer:.*?(\d+)',answer)
    if match == None:
        return None
    return match.group(1)

def solve_one(problem: str,n: int,temperature: float) -> tuple:
    """Generate n chains, parse, vote. Return (winner, vote_breakdown)."""
    areas = {}
    for i in range(0, n):
        answer = ask_llm(TEMPLATE.format(problem=problem), temperature)
        area = parse(answer)
        if area == None:
            continue
        if area in areas:
            areas[area]+=1
        else:
            areas[area] = 1
    if not areas:
        return None,{}
    return max(areas,key=areas.get),areas

def verify(candidate: float, ground_truth: float) -> bool:
    if candidate is None:
        return False
    if float(candidate) == float(ground_truth):
        return True
    else:
        return False

greedy_correct = sc_correct = 0
for i, (problem, truth) in enumerate(PROBLEMS, 1):
    g_winner, _ = solve_one(problem, n=1, temperature=0.0)
    s_winner, s_votes = solve_one(problem, n=5, temperature=0.9)
    g_ok, s_ok = verify(g_winner, truth), verify(s_winner, truth)
    greedy_correct += g_ok
    sc_correct += s_ok
    print(f"[{i:2d}] truth={truth:>4} | greedy={g_winner} {'✓' if g_ok else '✗'} | "
          f"SC={s_winner} {'✓' if s_ok else '✗'} votes={s_votes}")

print(f"\ngreedy: {greedy_correct}/{len(PROBLEMS)} = {greedy_correct/len(PROBLEMS):.0%}")
print(f"SC:     {sc_correct}/{len(PROBLEMS)} = {sc_correct/len(PROBLEMS):.0%}")





    