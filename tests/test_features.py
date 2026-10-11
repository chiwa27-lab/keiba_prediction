from __future__ import annotations

import numpy as np
import pandas as pd

from nankan_keiba.features import BASE_FEATURES, build_features, class_group


def _row(
    race_id: str, horse: str, pos: int | None, *, status: str | None = None, jockey: str = "J1"
) -> dict[str, object]:
    return {
        "race_id": race_id,
        "race_date": f"{race_id[0:4]}-{race_id[6:8]}-{race_id[8:10]}",
        "place_code": race_id[4:6],
        "race_number": int(race_id[10:12]),
        "race_class": "サラ系一般 C1",
        "surface": "ダ",
        "distance": 1200,
        "weather": "晴",
        "track_condition": "良",
        "num_horses": 3,
        "finish_position": pos,
        "finish_status": status if status is not None else (str(pos) if pos is not None else None),
        "horse_number": int(horse[1:]),
        "horse_id": horse,
        "horse_name": horse,
        "jockey_id": jockey,
        "finish_time_sec": 70.0 + (pos or 0) * 0.5 if pos else None,
        "last_3f": 38.0 + (pos or 0) * 0.1 if pos else None,
    }


def _history() -> pd.DataFrame:
    rows = [
        _row("202644010101", "h1", 1, jockey="J1"),
        _row("202644010101", "h2", 2, jockey="J2"),
        _row("202644010101", "h3", 3, jockey="J3"),
        _row("202644010102", "h4", 1, jockey="J1"),
        _row("202644020101", "h1", 3, jockey="J1"),
        _row("202644020101", "h2", None, status="取消", jockey="J2"),
        _row("202644020101", "h3", 1, jockey="J3"),
        _row("202644020101", "h5", 2, jockey="J2"),
    ]
    return pd.DataFrame(rows)


def test_class_group() -> None:
    assert class_group("サラ系一般 B2B3") == "B"
    assert class_group("サラ系３歳 3歳") == "3歳"
    assert class_group(None) == "other"


def test_horse_features_use_only_past_races() -> None:
    f = build_features(_history()).set_index(["race_id", "horse_id"])
    first = f.loc[("202644010101", "h1")]
    assert first["horse_starts"] == 0 and np.isnan(first["prev_rel_finish"])

    second = f.loc[("202644020101", "h1")]
    assert second["horse_starts"] == 1
    assert second["horse_win_rate"] == 1.0
    assert second["prev_rel_finish"] == 1 / 3
    assert second["days_since_last"] == 31


def test_scratched_runs_are_not_counted() -> None:
    f = build_features(_history()).set_index(["race_id", "horse_id"])
    scratched = f.loc[("202644020101", "h2")]
    assert np.isnan(scratched["is_win"]) and np.isnan(scratched["is_top3"])
    assert scratched["horse_starts"] == 1


def test_jockey_stats_exclude_same_day() -> None:
    f = build_features(_history()).set_index(["race_id", "horse_id"])
    # J1 は 1/1 に 2 戦 2 勝。同日の 2 レース目にはまだ反映しない
    assert f.loc[("202644010102", "h4")]["jockey_starts"] == 0
    later = f.loc[("202644020101", "h1")]
    assert later["jockey_starts"] == 2 and later["jockey_win_rate"] == 1.0


def test_future_results_do_not_change_past_features() -> None:
    base = build_features(_history())
    changed_history = _history()
    changed_history.loc[changed_history["race_id"] == "202644020101", "finish_position"] = [1.0, np.nan, 3.0, 2.0]
    changed = build_features(changed_history)
    past = base["race_id"] != "202644020101"
    pd.testing.assert_frame_equal(
        base.loc[past, BASE_FEATURES].reset_index(drop=True),
        changed.loc[past, BASE_FEATURES].reset_index(drop=True),
    )


def test_target_rows_get_features_without_labels() -> None:
    target = pd.DataFrame([_row("202644030101", "h1", None), _row("202644030101", "h3", None)])
    target = target.drop(columns=["finish_position", "finish_status", "finish_time_sec", "last_3f"])
    f = build_features(pd.concat([_history(), target], ignore_index=True))
    rows = f[f["race_id"] == "202644030101"].set_index("horse_id")
    assert rows["is_top3"].isna().all()
    assert rows.loc["h1", "horse_starts"] == 2
    assert rows.loc["h3", "horse_win_rate"] == 0.5
    assert rows.loc["h1", "rel_finish_mean5"] == rows.loc["h3", "rel_finish_mean5"] == 2 / 3
    assert rows["rel_finish_mean5_race_rank"].tolist() == [0.75, 0.75]
