from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path

from vdbbench.core.types import RunResult
from vdbbench.storage.schema import CREATE_TABLE_SQL, RESULTS_TABLE


class SQLiteStore:
    """Append-only results store. Runs accumulate across sessions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.commit()

    def save(self, result: RunResult) -> None:
        row = dataclasses.asdict(result)
        row["build_params"] = json.dumps(row["build_params"])
        row["search_params"] = json.dumps(row["search_params"])

        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(
            f"INSERT INTO {RESULTS_TABLE} ({cols}) VALUES ({placeholders})",
            row,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
