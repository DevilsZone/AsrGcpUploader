import json
from pathlib import Path
from typing import Any


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "next_index": 0,
            "processed_count": 0,
            "failed_count": 0,
        }

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2, ensure_ascii=False)

    temporary_path.replace(path)