from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nankan_keiba import features as features_module
from nankan_keiba.features import (
    BASE_FEATURES,
    affiliation_group,
    build_features,
    class_group,
    corner_positions,
    course_layout,
    first_corner_positions,
)


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


def test_standard_time_diff_uses_only_prior_days(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features_module, "STANDARD_TIME_MIN_SAMPLES", 1)
    rows = [
        _row("202644010101", "h1", 1),
        _row("202644010101", "h2", 2),
        _row("202644020101", "h1", 2),
        _row("202644020101", "h2", 1),
        _row("202644030101", "h1", None),
        _row("202644030101", "h2", None),
    ]
    df = build_features(pd.DataFrame(rows)).set_index(["race_id", "horse_id"])

    assert np.isnan(df.loc[("202644010101", "h1"), "std_diff"])
    standard = (70.5 + 71.0) / 2
    assert df.loc[("202644020101", "h1"), "std_diff"] == pytest.approx((71.0 - standard) / 1.2)
    assert np.isnan(df.loc[("202644020101", "h1"), "prev_std_diff"])
    assert df.loc[("202644030101", "h1"), "prev_std_diff"] == pytest.approx((71.0 - standard) / 1.2)
    assert df.loc[("202644030101", "h2"), "prev_std_diff"] == pytest.approx((70.5 - standard) / 1.2)
    assert (
        df.loc[("202644030101", "h2"), "std_diff_mean5_race_rank"]
        < df.loc[("202644030101", "h1"), "std_diff_mean5_race_rank"]
    )


def test_corner_positions_share_position_within_parentheses() -> None:
    assert corner_positions("(4,5),8,(1,6,7),2-3") == {4: 1, 5: 1, 8: 3, 1: 4, 6: 4, 7: 4, 2: 7, 3: 8}
    assert first_corner_positions('{"4コーナー": "2,1", "3コーナー": "1,(2,3)"}') == {1: 1, 2: 2, 3: 2}
    assert first_corner_positions(None) == {}


def test_early_position_pop_response_and_place_rates_use_only_past() -> None:
    history = _history()
    corners = {"202644010101": '{"3コーナー": "3,(1,2)"}', "202644020101": '{"3コーナー": "1,5,3"}'}
    history["corner_passing"] = history["race_id"].map(corners)
    history["popularity"] = [3, 1, 2, 1, 1, None, 3, 2]
    f = build_features(history).set_index(["race_id", "horse_id"])

    assert np.isnan(f.loc[("202644010101", "h1"), "prev_early_pos"])
    later = f.loc[("202644020101", "h1")]
    assert later["prev_early_pos"] == pytest.approx(2 / 3)
    assert f.loc[("202644020101", "h3"), "prev_early_pos"] == pytest.approx(1 / 3)
    # 1 番人気で 2 着 → log2(1/2)、3 番人気で 1 着 → log2(3)
    assert f.loc[("202644020101", "h2"), "prev_pop_response"] == pytest.approx(-1.0)
    assert later["prev_pop_response"] == pytest.approx(np.log2(3))
    # J1 は 1/1 に 3 番人気 1 着・1 番人気 1 着
    assert later["jockey_pop_response"] == pytest.approx(np.log2(3) / 2)
    assert np.isnan(f.loc[("202644010102", "h4"), "jockey_pop_response"])
    assert later["place_win_rate"] == 1.0
    assert later["jockey_place_starts"] == 2 and later["jockey_place_win_rate"] == 1.0
    assert f.loc[("202644020101", "h3"), "early_pos_mean5_race_rank"] < later["early_pos_mean5_race_rank"]


def test_course_layout_and_affiliation_group() -> None:
    assert course_layout("45", 1400) == (300.0, 1200.0)
    assert course_layout("44", 1200) == (386.0, 1600.0)
    assert course_layout("44", 1600) == (286.0, 1400.0)
    assert np.isnan(course_layout("30", 1400)[0])
    assert affiliation_group("大井", "44") == "home"
    assert affiliation_group("川崎", "44") == "nankan"
    assert affiliation_group("美浦", "44") == affiliation_group("栗東", "45") == "jra"
    assert affiliation_group("高知", "44") == "other_nar"
    assert affiliation_group(None, "44") == affiliation_group("", "44") == "unknown"


def test_jra_features() -> None:
    history = _history()
    history["trainer_affiliation"] = ["大井", "美浦", "栗東", None, "大井", "美浦", "川崎", "高知"]
    history.loc[history["horse_id"] == "h3", "jockey_id"] = "01088"
    f = build_features(history).set_index(["race_id", "horse_id"])
    assert f.loc[("202644010101", "h1"), "race_jra_count"] == 2
    assert f.loc[("202644010102", "h4"), "race_jra_count"] == 0
    assert f.loc[("202644010102", "h4"), "affiliation_group"] == "unknown"
    assert f.loc[("202644020101", "h3"), "affiliation_group"] == "nankan"
    assert f.loc[("202644010101", "h3"), "jra_jockey"] == 1
    assert f.loc[("202644010101", "h1"), "jra_jockey"] == 0


def test_course_front_advantage_uses_only_prior_days(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features_module, "COURSE_BIAS_MIN_SAMPLES", 1)
    rows = []
    for race_id, corner in (("202645010101", "1,2,3,4,5,6"), ("202645020101", "6,5,4,3,2,1")):
        for n in range(1, 7):
            row = _row(race_id, f"h{n}", n)
            row["num_horses"] = 6
            row["corner_passing"] = json.dumps({"3コーナー": corner})
            rows.append(row)
    f = build_features(pd.DataFrame(rows)).set_index(["race_id", "horse_id"])

    assert np.isnan(f.loc[("202645010101", "h1"), "course_front_adv"])
    # 1/1 は前 1/3（h1, h2）が 3 着内 2/2、後ろ 1/3（h5, h6）が 0/2。1/2 の結果はまだ使わない
    day2 = f.loc[("202645020101", "h1")]
    assert day2["course_front_adv"] == pytest.approx(1.0)
    assert day2["early_pos_mean5"] == pytest.approx(1 / 6)
    assert day2["style_fit"] == pytest.approx(0.5 - 1 / 6)
    assert f.loc[("202645020101", "h6"), "style_fit"] == pytest.approx(-0.5)
    assert day2["race_front_runners"] == 1


def test_same_day_bias_uses_only_earlier_races(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features_module, "DAY_BIAS_MIN_SAMPLES", 1)
    rows = []
    # 1R は前に行った馬（馬番の小さい馬）が上位、2R は後ろの馬が上位、3R は予想対象
    for race_id, corner, order in (
        ("202645010101", "1,2,3,4,5,6", [1, 2, 3, 4, 5, 6]),
        ("202645010102", "1,2,3,4,5,6", [6, 5, 4, 3, 2, 1]),
        ("202645010103", "", [None] * 6),
    ):
        for n, pos in enumerate(order, start=1):
            row = _row(race_id, f"h{n}{race_id[-1]}", pos)
            row["horse_number"] = n
            row["num_horses"] = 6
            row["corner_passing"] = json.dumps({"3コーナー": corner}) if corner else None
            rows.append(row)
    f = build_features(pd.DataFrame(rows)).set_index(["race_id", "horse_number"])

    first = f.loc[("202645010101", 1)]
    assert first["day_prior_races"] == 0 and np.isnan(first["day_front_adv"])
    second = f.loc[("202645010102", 1)]
    # 1R だけ: 前 1/3（1, 2 番）が 3 着内 2/2、後ろ 1/3（5, 6 番）が 0/2。2R の結果は使わない
    assert second["day_prior_races"] == 1
    assert second["day_front_adv"] == pytest.approx(1.0)
    assert second["day_inner_adv"] == pytest.approx(1.0)
    assert second["day_winner_early_pos"] == pytest.approx(1 / 6)
    third = f.loc[("202645010103", 1)]
    assert third["day_prior_races"] == 2
    assert third["day_front_adv"] == pytest.approx(0.0)
    assert third["day_winner_early_pos"] == pytest.approx((1 / 6 + 1) / 2)
    assert third["day_inner_fit"] == pytest.approx(0.0)
    other_place = _row("202644010101", "x1", None)
    assert (
        build_features(pd.DataFrame([*rows, other_place])).set_index("race_id").loc["202644010101", "day_prior_races"]
        == 0
    )


def test_pedigree_features_use_only_prior_days() -> None:
    history = _history()
    history["sire_id"] = history["horse_id"].map({"h1": "S1", "h3": "S1", "h4": "S1", "h5": "S1"})
    history["damsire_id"] = "D1"
    history["dam_id"] = history["horse_id"].map({"h1": "M1", "h5": "M1"})
    history["sire_line"] = "Halo系"
    target = _row("202644030101", "h9", None)
    target.update(sire_id="S1", damsire_id="D1", dam_id="M1", sire_line="Halo系")
    f = build_features(pd.concat([history, pd.DataFrame([target])], ignore_index=True))
    f = f.set_index(["race_id", "horse_id"])

    # 1/1 の産駒（h1, h3, h4）は当日なので 0 戦扱い
    assert f.loc[("202644010101", "h1"), "sire_starts"] == 0
    # 1/2 時点: 1/1 の S1 産駒は 3 戦 2 勝・3 着内 3
    day2 = f.loc[("202644020101", "h5")]
    assert day2["sire_starts"] == 3
    assert day2["sire_win_rate"] == pytest.approx(2 / 3)
    assert day2["sire_dist_top3_rate"] == pytest.approx(1.0)
    assert day2["dam_starts"] == 1 and day2["dam_top3_rate"] == pytest.approx(1.0)
    # 初出走の馬にも血統の成績が付く（1/2 の取消は数えない）
    debut = f.loc[("202644030101", "h9")]
    assert debut["horse_starts"] == 0
    assert debut["sire_starts"] == 6
    assert debut["sire_win_rate"] == pytest.approx(3 / 6)
    assert debut["damsire_starts"] == 7
    assert debut["sire_line"] == "Halo系"
    assert debut["sire_wet_top3_rate"] == pytest.approx(1.0)
