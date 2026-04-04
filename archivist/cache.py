from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


class SQLiteScoreCache:
    """Tiny SQLite cache for expensive directional scores.

    Keyed by (kind, model_name, key).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
                        CREATE TABLE IF NOT EXISTS scores_v2 (
                            kind TEXT NOT NULL,
                            model TEXT NOT NULL,
                            k TEXT NOT NULL,
                            score REAL NOT NULL,
                            PRIMARY KEY (kind, model, k)
                        )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_many(
        self,
        *,
        kind: str,
        model: str,
        keys: Iterable[str],
    ) -> dict[str, float]:
        keys = list(keys)
        if not keys:
            return {}

        # Chunk to avoid SQLite parameter limits.
        out: dict[str, float] = {}
        cur = self._conn.cursor()
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            rows = cur.execute(
                f"SELECT k,score FROM scores_v2 WHERE kind=? AND model=? AND k IN ({placeholders})",
                [kind, model, *chunk],
            ).fetchall()
            for k, s in rows:
                out[str(k)] = float(s)
        return out

    def set_many(
        self,
        *,
        kind: str,
        model: str,
        entries: Iterable[tuple[str, float]],
    ) -> None:
        rows = [(kind, model, str(k), float(s)) for k, s in entries]
        if not rows:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO scores_v2(kind,model,k,score) VALUES (?,?,?,?)",
            rows,
        )
        self._conn.commit()
