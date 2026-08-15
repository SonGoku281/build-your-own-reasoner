"""
Environment-based configuration for the reasoner.

Loads settings from a .env file (if present) and environment variables.
No hardcoded values in the codebase — every knob lives in .env.

Usage:
    cp .env.example .env   # then edit .env
    python3 scatter.py

The loader is deliberately dependency-free (~15 lines of stdlib):
parse KEY=VALUE lines, expand ${VAR} references, and let real environment
variables win over .env (standard 12-factor behavior).
"""

import os
import re
from pathlib import Path

_ENV_FILE = Path(__file__).parent / ".env"
_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _load_dotenv(path: Path) -> None:
    """Parse KEY=VALUE lines from path into os.environ (without overwriting)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _expand(value: str) -> str:
    """Resolve ${OTHER_VAR} references inside a value."""
    return _VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


_load_dotenv(_ENV_FILE)


def get(key: str, default: str = "") -> str:
    """Read a config value with ${VAR} expansion and a fallback default."""
    return _expand(os.environ.get(key, default))


def get_int(key: str, default: int) -> int:
    return int(get(key, str(default)))


def get_float(key: str, default: float) -> float:
    return float(get(key, str(default)))


# ── The settings the reasoner uses ─────────────────────────────────────────
MODEL = get("MODEL", "gemma-4-12b-v2")              # which model to query
BASE_URL = get("BASE_URL", "http://localhost:8081/v1")  # OpenAI-compatible endpoint
MAX_TOKENS = get_int("MAX_TOKENS", 8192)            # thinking + answer budget
REQUEST_TIMEOUT = get_int("REQUEST_TIMEOUT", 600)   # seconds; first call cold-loads

# sampling
SC_TEMPERATURE = get_float("SC_TEMPERATURE", 0.9)   # diversity for the vote
SC_SAMPLES = get_int("SC_SAMPLES", 5)               # chains per problem (vote)
GREEDY_TEMPERATURE = get_float("GREEDY_TEMPERATURE", 0.0)  # deterministic baseline
