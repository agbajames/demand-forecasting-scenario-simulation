"""Shared project utilities for paths, directories, and serialisation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory, including parents, if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_project_directories() -> None:
    """Create the standard project directories used by ingestion and EDA."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EDA_FIGURES_DIR, TABLES_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
        ensure_dir(directory)


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data with stable indentation and UTF-8 encoding."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    return output_path


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_filename(value: str, default: str = "file") -> str:
    """Convert arbitrary text into a conservative, portable file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or default
