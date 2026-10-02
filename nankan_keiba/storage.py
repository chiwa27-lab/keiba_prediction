from __future__ import annotations

import csv
import sqlite3
from dataclasses import asdict, fields
from pathlib import Path

from .models import Payout, RaceInfo, RaceResult, ResultEntry

_SQL_TYPES = {int: "INTEGER", float: "REAL", str: "TEXT"}
_TABLES: dict[str, tuple[type, str]] = {
    "races": (RaceInfo, "PRIMARY KEY (race_id)"),
    "results": (ResultEntry, "FOREIGN KEY (race_id) REFERENCES races(race_id)"),
    "payouts": (Payout, "FOREIGN KEY (race_id) REFERENCES races(race_id)"),
}


def _column_type(annotation: object) -> str:
    text = str(annotation)
    for py_type, sql_type in _SQL_TYPES.items():
        if py_type.__name__ in text:
            return sql_type
    return "TEXT"


class RaceStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        for table, (model, constraint) in _TABLES.items():
            columns = ", ".join(f"{f.name} {_column_type(f.type)}" for f in fields(model))
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns}, {constraint})")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_results_race ON results(race_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_payouts_race ON payouts(race_id)")
        self.conn.commit()

    def has_race(self, race_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM races WHERE race_id = ?", (race_id,)).fetchone()
        return row is not None

    def save(self, result: RaceResult) -> None:
        race_id = result.race.race_id
        with self.conn:
            self.conn.execute("DELETE FROM results WHERE race_id = ?", (race_id,))
            self.conn.execute("DELETE FROM payouts WHERE race_id = ?", (race_id,))
            self.conn.execute("DELETE FROM races WHERE race_id = ?", (race_id,))
            self._insert("races", [asdict(result.race)])
            self._insert("results", [asdict(e) for e in result.entries])
            self._insert("payouts", [asdict(p) for p in result.payouts])

    def _insert(self, table: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        columns = list(rows[0])
        placeholders = ", ".join("?" for _ in columns)
        self.conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [tuple(row[c] for c in columns) for row in rows],
        )

    def count(self, table: str) -> int:
        if table not in _TABLES:
            raise ValueError(table)
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def export_csv(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for table in _TABLES:
            cursor = self.conn.execute(f"SELECT * FROM {table} ORDER BY race_id")
            path = out_dir / f"{table}.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([d[0] for d in cursor.description])
                writer.writerows(cursor)
            written.append(path)
        return written

    def close(self) -> None:
        self.conn.close()
