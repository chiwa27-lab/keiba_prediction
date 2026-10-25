from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from nankan_keiba.spat4 import (
    Spat4Settings,
    build_tickets,
    check,
    load_log,
    load_settings,
    record,
    save_settings,
)


def _pred() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["202642102101"] * 5 + ["202642102102"] * 3,
            "pred_rank": [2, 1, 3, 4, 5, 1, 2, 3],
            "horse_number": [1, 2, 3, 4, 5, 7, 8, 9],
        }
    )


def test_settings_roundtrip_and_validation(tmp_path: Path) -> None:
    assert not load_settings(tmp_path).complete
    save_settings(tmp_path, Spat4Settings("12345678", "1234", 3000))
    loaded = load_settings(tmp_path)
    assert loaded == Spat4Settings("12345678", "1234", 3000)
    assert loaded.complete
    with pytest.raises(ValueError):
        save_settings(tmp_path, Spat4Settings("abc", "1234"))
    with pytest.raises(ValueError):
        save_settings(tmp_path, Spat4Settings("1", "2", 0))


def test_build_tickets_only_selected_races_and_types() -> None:
    t = build_tickets(_pred(), ["202642102101"], ["単勝", "馬連", "3連単"])
    assert set(t["race_id"]) == {"202642102101"}
    assert t[t["券種"] == "単勝"]["買い目"].tolist() == ["2"]
    assert t[t["券種"] == "馬連"]["買い目"].tolist() == ["1-2", "2-3", "2-4"]
    assert "2→1→3" in t[t["券種"] == "3連単"]["買い目"].tolist()
    assert (t["金額"] == 100).all()
    assert build_tickets(_pred(), [], ["単勝"]).empty


def test_check_skips_duplicates_and_limits(tmp_path: Path) -> None:
    day = date(2026, 10, 25)
    t = build_tickets(_pred(), ["202642102101", "202642102102"], ["単勝", "複勝"])
    first = check(t, load_log(tmp_path, day), 300)
    assert first.total_yen == 400 and first.over_limit
    record(tmp_path, t.head(2), day, now=datetime(2026, 10, 25, 10, 0))
    again = check(t, load_log(tmp_path, day), 500)
    assert len(again.duplicates) == 2
    assert again.total_yen == 200 and again.used_today_yen == 200
    assert not again.over_limit
    assert load_log(tmp_path, date(2026, 10, 26)).empty
