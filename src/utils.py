"""Utility helpers for the project.

This module provides a small, dependency-free `load_dotenv` helper to
load KEY=VALUE pairs from a `.env` file into `os.environ`.

Example:
    from code.utils import load_dotenv
    load_dotenv('.env')

"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Optional

def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env and env.get("KAGGLE_KEY", "").startswith("KG"):
        env["KAGGLE_API_TOKEN"] = env["KAGGLE_KEY"]
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True, capture_output=True, check=False, env=env)


def slug_from_url(value: str) -> str:
    value = value.strip().rstrip("/")
    m = re.search(r"kaggle\.com/competitions/([^/?#]+)", value)
    if m:
        return m.group(1)
    return value.split("/")[-1]

def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value

def _workspace_for_challenge(slug: str | None) -> Path:
    return Path("runs") / slug

def write_python_code_to_file(content: str, filename: str, slug: str, append: bool = False):
    solution_dir = Path.cwd() / "runs" / slug / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)

    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        for block in code_blocks:
            block_text = block.strip()
            if not block_text:
                continue

            header_match = re.search(r"(?im)^\s*#\s*FILE:\s*(.+?)\s*$", block_text)
            relative_path = header_match.group(1).strip() if header_match else filename
            relative_path = relative_path.lstrip("./")
            if relative_path.startswith("solution/"):
                relative_path = relative_path[len("solution/"):]

            block_text = re.sub(r"(?im)^\s*#\s*FILE:\s*.+\s*$", "", block_text).strip()
            target = solution_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)

            if append:
                existing = target.read_text(encoding="utf-8") if target.exists() else ""
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                if existing and block_text and not existing.rstrip().endswith(block_text.rstrip()):
                    block_payload = "\n\n" + block_text if existing.strip() else block_text
                    target.write_text(existing + block_payload + "\n", encoding="utf-8")
                elif not existing.strip() and block_text:
                    target.write_text(block_text + "\n", encoding="utf-8")
                continue

            target.write_text(block_text + "\n", encoding="utf-8")
        return

    target = solution_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=True)
    target.write_text(content.strip() + "\n", encoding="utf-8")


def get_file_content(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def ensure_dirs(workspace: Path) -> None:
    for p in ["data", "solution", "reports", "scratch", "submissions"]:
        (workspace / p).mkdir(parents=True, exist_ok=True)


def load_dotenv(path: str = ".env", override: bool = False, encoding: str = "utf-8") -> None:
    """Load environment variables from a .env file.

    Behavior:
    - Ignores blank lines and lines starting with `#`.
    - Parses simple `KEY=VALUE` pairs. Values may be quoted with single
      or double quotes.
    - If `override` is False (default), existing `os.environ` values are
      preserved.

    This is intentionally lightweight to avoid an external dependency.
    """
    try:
        with open(path, encoding=encoding) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()

                if not key:
                    continue

                # Handle quoted values
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = _strip_quotes(val)
                else:
                    # Remove inline comments after an unquoted value
                    if "#" in val:
                        val = val.split("#", 1)[0].strip()

                if not override and key in os.environ:
                    continue

                os.environ[key] = val
    except FileNotFoundError:
        return


__all__ = ["load_dotenv"]
