import os
from pathlib import Path
from typing import Any

import yaml


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path) -> dict[str, Any]:
    load_env_file()
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
