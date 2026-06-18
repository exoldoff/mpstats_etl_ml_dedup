from __future__ import annotations

import os
from pathlib import Path
import re


ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root for research helpers."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "research" / "dedup").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _unquote_value(value: str) -> str:
    if len(value) < 2 or value[0] not in {"'", '"'} or value[-1] != value[0]:
        return value
    body = value[1:-1]
    if value[0] == '"':
        replacements = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
        result: list[str] = []
        escaped = False
        for char in body:
            if escaped:
                result.append(replacements.get(char, char))
                escaped = False
            elif char == "\\":
                escaped = True
            else:
                result.append(char)
        if escaped:
            result.append("\\")
        return "".join(result)
    return body


def parse_dotenv(content: str) -> dict[str, str]:
    """Parse the small .env subset used by the research notebooks."""
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.match(key):
            continue
        value = _strip_inline_comment(raw_value.strip())
        values[key] = _unquote_value(value)
    return values


def load_research_dotenv(
    project_root: Path | None = None,
    *,
    env_path: Path | None = None,
    override: bool = False,
) -> Path | None:
    """Load repository .env into os.environ without overriding shell values."""
    path = env_path or (find_project_root(project_root) / ".env")
    if not path.exists():
        return None
    values = parse_dotenv(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return path
