"""Small cached report reads; raw CSVs and ML imports stay out of report pages."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, max_entries=48)
def _read(path: str, stamp: tuple[int, int], kind: str):
    file = Path(path)
    if kind == "json":
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    if kind == "csv":
        return pd.read_csv(file)
    return file.read_bytes()


class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        self.reports = root / "reports"
        self.issues: dict[str, str] = {}

    def read(self, name: str, kind: str = "json"):
        path = self.reports / name
        try:
            status = path.stat()
            return _read(str(path), (status.st_mtime_ns, status.st_size), kind)
        except FileNotFoundError:
            self.issues[name] = "Local artifact not found"
        except (OSError, ValueError, UnicodeError) as exc:
            self.issues[name] = f"Could not read local artifact ({type(exc).__name__})"
        return None

    def csv(self, name: str, required: tuple[str, ...] = ()):
        frame = self.read(name, "csv")
        if frame is not None and not set(required).issubset(frame.columns):
            self.issues[name] = "Required report columns are missing"
            return None
        return frame

    def bytes(self, name: str):
        return self.read(name, "bytes")


def nested(record, *keys):
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record
