from __future__ import annotations

from datetime import date
from pathlib import Path

from nankan_keiba.features import build_features
from nankan_keiba.model import train
from nankan_keiba.parsers import parse_shutuba
from nankan_keiba.predict import format_predictions, predict_cards

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
