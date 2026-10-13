from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nankan_keiba.config import race_result_url
from nankan_keiba.features import build_features
from nankan_keiba.model import train
from nankan_keiba.parsers import parse_result, parse_shutuba
from nankan_keiba.predict import fetch_same_day_results, format_predictions, predict_cards

from .test_model import _synthetic_history

FIXTURES = Path(__file__).parent / "fixtures"


def test_predict_cards_excludes_scratched_and_uses_history() -> None:
    history, payouts = _synthetic_history()
    history.loc[history["horse_id"] == "h1", "horse_id"] = "2021100001"
    model, _ = train(build_features(history), payouts, valid_from=date(2026, 2, 20))
    card = parse_shutuba((FIXTURES / "shutuba.html").read_text(encoding="utf-8"), "202645101201")

    pred = predict_cards(model, history, [card])

    assert pred["horse_number"].tolist() in ([1, 2], [2, 1])
    assert abs(pred["win_prob"].sum() - 1.0) < 1e-9
    assert set(pred["place_name"]) == {"川崎"}
    text = format_predictions(pred)
    assert "川崎1R テスト特別" in text and "テストホースC" not in text


class StubClient:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.requested: list[tuple[str, bool]] = []

    def get(self, url: str, *, use_cache: bool = True) -> str:
        self.requested.append((url, use_cache))
        return self.pages.get(url, "<html></html>")


def test_fetch_same_day_results_stops_at_unfinished_race() -> None:
    result_html = (FIXTURES / "result.html").read_text(encoding="utf-8")
    client = StubClient({race_result_url("202645101201"): result_html, race_result_url("202645101203"): result_html})

    results = fetch_same_day_results(client, ["202645101205"], known_race_ids=["202645101202"])  # type: ignore[arg-type]

    # 1R は取得、2R は保存済み、3R は取得、4R は未確定なのでそこで止まる（5R は予想対象）
    assert [r.race.race_id for r in results] == ["202645101201", "202645101203"]
    assert [url for url, _ in client.requested] == [race_result_url(f"20264510120{n}") for n in (1, 3, 4)]
    assert all(not use_cache for _, use_cache in client.requested)


def test_predict_cards_with_same_day_results_and_pedigrees() -> None:
    history, payouts = _synthetic_history()
    model, _ = train(build_features(history), payouts, valid_from=date(2026, 2, 20))
    card = parse_shutuba((FIXTURES / "shutuba.html").read_text(encoding="utf-8"), "202645101202")
    earlier = parse_result((FIXTURES / "result.html").read_text(encoding="utf-8"), "202645101201")
    pedigrees = pd.DataFrame(
        [
            {"horse_id": e.horse_id, "sire_id": "S1", "sire_line": "Halo系", "dam_id": None, "damsire_id": "D1"}
            for e in card.entries
        ]
    )

    pred = predict_cards(model, history, [card], same_day=[earlier], pedigrees=pedigrees)

    assert pred["horse_number"].tolist() in ([1, 2], [2, 1])
    assert abs(pred["win_prob"].sum() - 1.0) < 1e-9
