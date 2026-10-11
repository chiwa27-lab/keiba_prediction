from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nankan_keiba.features import build_features
from nankan_keiba.model import RaceModel, log_loss, roc_auc, train


def _synthetic_history(days: int = 60, races_per_day: int = 4, field: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n_horses = 120
    ability = rng.normal(size=n_horses)
    rows, payouts = [], []
    start = date(2026, 1, 5)
    for d in range(days):
        day = start + timedelta(days=d)
        for r in range(1, races_per_day + 1):
            race_id = f"{day:%Y}44{day:%m%d}{r:02d}"
            horses = rng.choice(n_horses, size=field, replace=False)
            score = ability[horses] + rng.normal(scale=0.7, size=field)
            order = np.argsort(-score)
            for pos, idx in enumerate(order, start=1):
                h = int(horses[idx])
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": day.isoformat(),
                        "place_code": "44",
                        "race_number": r,
                        "race_class": "サラ系一般 C1",
                        "surface": "ダ",
                        "distance": 1200,
                        "weather": "晴",
                        "track_condition": "良",
                        "num_horses": field,
                        "finish_position": pos,
                        "finish_status": str(pos),
                        "horse_number": int(idx) + 1,
                        "horse_id": f"h{h}",
                        "horse_name": f"h{h}",
                        "jockey_id": f"j{h % 10}",
                        "finish_time_sec": 72.0 + pos * 0.3,
                        "last_3f": 38.0 + pos * 0.1,
                        "popularity": pos,
                        "win_odds": 1.5 * pos,
                    }
                )
                if pos == 1:
                    payouts.append({"race_id": race_id, "horse_number": int(idx) + 1, "payout_yen": 300})
    return pd.DataFrame(rows), pd.DataFrame(payouts)


def test_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert abs(log_loss(y, np.array([0.5, 0.5, 0.5, 0.5])) - np.log(2)) < 1e-12


def test_train_evaluate_save_and_predict(tmp_path: Path) -> None:
    history, payouts = _synthetic_history()
    features = build_features(history)
    model, metrics = train(features, payouts, valid_from=date(2026, 2, 20))

    assert metrics["valid_from"] == "2026-02-20"
    assert float(metrics["top3_auc"]) > 0.6  # type: ignore[arg-type]
    model.save(tmp_path)
    loaded = RaceModel.load(tmp_path)

    race = features[features["race_id"] == features["race_id"].iloc[-1]]
    pred = loaded.predict(race)
    assert sorted(pred["pred_rank"]) == list(range(1, len(race) + 1))
    assert abs(pred["win_prob"].sum() - 1.0) < 1e-9
    assert pred["top3_prob"].between(0, 1).all()
    np.testing.assert_allclose(pred["win_prob"], model.predict(race)["win_prob"])
