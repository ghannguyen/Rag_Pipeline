"""
Single .env loader for the backend.

- Real environment variables win over .env (e.g. `RETRIEVAL_TOP_K=15 uvicorn ...`).
- Safe to call on every request: edits to .env apply without a restart, because
  only keys that this loader set itself are updated or removed.
- Empty values (`KEY=`) count as unset, so `int(os.getenv(...))` keeps its default.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

# Keys this module put into os.environ, with the value it set.
_FROM_DOTENV: dict = {}


def parse_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip("'").strip('"')
            if key and value:
                values[key] = value
    return values


def load_env(path: Path = ENV_FILE) -> None:
    values = parse_env_file(path)

    # Remove keys we set earlier that were deleted/emptied in .env, unless
    # something else changed them in the meantime.
    for key in list(_FROM_DOTENV):
        if key not in values:
            if os.environ.get(key) == _FROM_DOTENV[key]:
                os.environ.pop(key, None)
            del _FROM_DOTENV[key]

    for key, value in values.items():
        owned = key in _FROM_DOTENV and os.environ.get(key) == _FROM_DOTENV[key]
        if key in os.environ and not owned:
            continue  # set by the real environment: it wins
        os.environ[key] = value
        _FROM_DOTENV[key] = value
