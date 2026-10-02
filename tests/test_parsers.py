import json
from pathlib import Path

import pytest

from nankan_keiba.parsers import ParseError, parse_calendar, parse_race_ids, parse_result

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_calendar_filters_nankan_places() -> None:
    kaisai = parse_calendar(load("calendar.html"))
    assert [(k.kaisai_id, k.kaisai_date, k.place_name) for k in kaisai] == [
        ("2025420903", "20250903", "浦和"),
        ("2025440901", "20250901", "大井"),
        ("2025450902", "20250902", "川崎"),
    ]


def test_parse_calendar_with_place_subset() -> None:
    assert [k.place_code for k in parse_calendar(load("calendar.html"), ["44"])] == ["44"]


def test_parse_race_ids_for_kaisai() -> None:
    assert parse_race_ids(load("race_list.html"), "2025440901") == ["202544090101", "202544090102"]


def test_parse_result_race_info() -> None:
    race = parse_result(load("result.html"), "202544090111").race
    assert race.race_date == "2025-09-01"
    assert race.place_name == "大井"
    assert race.race_number == 11
    assert race.race_name == "テスト記念"
    assert race.grade == "重賞"
    assert (race.post_time, race.surface, race.distance, race.direction) == ("20:10", "ダ", 1600, "右")
    assert (race.weather, race.track_condition) == ("曇", "稍重")
    assert (race.kai, race.nichi, race.num_horses) == (5, 2, 3)
    assert race.race_class == "サラ系一般 C1"
    assert race.prize_money == "100.0,40.0,25.0"
    assert json.loads(race.corner_passing or "") == {"3コーナー": "1,2", "4コーナー": "(1,2)"}
    assert json.loads(race.lap_times or "")["sections"] == ["12.0", "12.5"]


def test_parse_result_entries() -> None:
    entries = parse_result(load("result.html"), "202544090111").entries
    assert len(entries) == 3
    first, second, scratched = entries
    assert first.finish_position == 1
    assert first.horse_id == "2021100001"
    assert first.horse_name == "テストホースA"
    assert (first.sex, first.age, first.weight_carried) == ("牝", 4, 54.0)
    assert (first.jockey_id, first.jockey_name) == ("05380", "騎手A")
    assert (first.finish_time, first.finish_time_sec) == ("1:40.2", 100.2)
    assert first.margin is None
    assert (first.popularity, first.win_odds, first.last_3f) == (1, 2.5, 38.1)
    assert (first.trainer_affiliation, first.trainer_id, first.trainer_name) == ("大井", "05667", "調教師A")
    assert (first.horse_weight, first.horse_weight_diff) == (470, 2)
    assert second.margin == "1.1/2"
    assert second.horse_weight_diff == -4
    assert scratched.finish_position is None
    assert scratched.finish_status == "取消"
    assert scratched.finish_time_sec is None
    assert scratched.popularity is None
    assert scratched.win_odds is None
    assert scratched.horse_weight is None


def test_parse_result_payouts() -> None:
    payouts = parse_result(load("result.html"), "202544090111").payouts
    assert [(p.bet_type, p.combination, p.payout_yen, p.popularity) for p in payouts] == [
        ("単勝", "1", 250, 1),
        ("複勝", "1", 110, 1),
        ("複勝", "2", 1230, 2),
        ("馬単", "1-2", 12340, 3),
    ]


def test_parse_result_without_table_raises() -> None:
    with pytest.raises(ParseError):
        parse_result("<html><body>まだ結果はありません</body></html>", "202544090111")
