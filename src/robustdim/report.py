import json
from pathlib import Path


class Tee:
    """Print each line to stdout and append it to a fresh log file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def save_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
