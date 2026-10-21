from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import nankan_keiba.predict as predict_module
from nankan_keiba.model import RaceModel
from nankan_keiba.predict import OUTPUT_COLUMNS, PredictionRun
from nankan_keiba.webapp import jst_today, last_race_date, load_actual_results, race_label, race_table, update_period

APP = str(Path(__file__).resolve().parents[1] / "nankan_keiba" / "app.py")


def _pred() -> pd.DataFrame:
    rows = [
        ["202642102101", "2026-10-21", "浦和", 1, "2歳", 1400, rank, num, f"馬{num}", f"J{num}", win, top3, odds, pop]
        for rank, num, win, top3, odds, pop in [
            (2, 1, 0.2, 0.5, 4.0, 2),
            (1, 2, 0.4, 0.7, 2.0, 1),
            (3, 3, 0.1, 0.3, 9.0, 3),
        ]
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def test_jst_today_uses_japan_date() -> None:
    assert jst_today(datetime(2026, 10, 20, 16, 0, tzinfo=timezone.utc)) == date(2026, 10, 21)


def test_race_label() -> None:
    assert race_label("202642102108") == "浦和 8R"
    assert race_label("202644102112") == "大井 12R"


def test_update_period() -> None:
    assert update_period(date(2026, 10, 18), date(2026, 10, 21)) == (date(2026, 10, 19), date(2026, 10, 20))
    assert update_period(date(2026, 10, 20), date(2026, 10, 21)) is None
    assert update_period(None, date(2026, 10, 21), default_days=10) == (date(2026, 10, 10), date(2026, 10, 20))


def test_race_table_orders_by_prediction_and_adds_results(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE races (race_id TEXT, race_date TEXT)")
        conn.execute("INSERT INTO races VALUES ('202642102101', '2026-10-21')")
        conn.execute("CREATE TABLE results (race_id TEXT, horse_number INT, finish_position INT, finish_status TEXT)")
        conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?)",
            [("202642102101", 1, 1, ""), ("202642102101", 2, None, "中止"), ("202642102101", 3, 2, "")],
        )
    assert last_race_date(db) == date(2026, 10, 21)
    actual = load_actual_results(db, ["202642102101"])

    table = race_table(_pred(), actual)
    assert list(table["馬番"]) == [2, 1, 3]
    assert list(table["印"]) == ["◎", "○", "▲"]
    assert list(table["着順"]) == ["中止", "1", "2"]
    assert table["勝率"].iloc[0] == pytest.approx(40.0)

    no_result = race_table(_pred(), load_actual_results(tmp_path / "missing.sqlite", ["202642102101"]))
    assert "着順" not in no_result.columns


def test_app_predicts_selected_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "data" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "meta.json").write_text("{}")
    model = RaceModel({}, [], False, {"trained_until": "2026-10-20"})
    calls: list[list[str]] = []

    def fake_run(client: object, db: Path, m: RaceModel, race_ids: list[str]) -> PredictionRun:
        calls.append(race_ids)
        return PredictionRun(_pred(), 0, 0)

    monkeypatch.setattr(RaceModel, "load", classmethod(lambda cls, path: model))
    monkeypatch.setattr(predict_module, "find_race_ids", lambda client, d, places=None: ["202642102101"])
    monkeypatch.setattr(predict_module, "run_prediction", fake_run)

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "浦和" in at.markdown[0].value
    at.button(key="predict").click().run()
    assert not at.exception
    assert calls == [["202642102101"]]
    assert at.subheader[0].value.startswith("浦和 1R")
    assert len(at.dataframe) == 1
