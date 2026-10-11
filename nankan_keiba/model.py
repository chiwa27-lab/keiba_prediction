"""LightGBM による着順予想モデルの学習・評価・推論."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from .features import CATEGORICAL_FEATURES, feature_columns

logger = logging.getLogger(__name__)

TARGETS = {"win": "is_win", "top3": "is_top3"}

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 1,
}
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100


@dataclass
class RaceModel:
    boosters: dict[str, lgb.Booster]
    features: list[str]
    use_odds: bool
    meta: dict[str, object] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """各馬の勝率（レース内で合計 1）・3 着内確率・予想順位を返す."""
        x = frame[self.features]
        out = frame[["race_id", "horse_number", "horse_name"]].copy()
        win_raw = np.asarray(self.boosters["win"].predict(x), dtype=float)
        out["top3_prob"] = np.asarray(self.boosters["top3"].predict(x), dtype=float)
        out["win_prob"] = win_raw / pd.Series(win_raw, index=out.index).groupby(out["race_id"]).transform("sum")
        out["pred_rank"] = out.groupby("race_id")["win_prob"].rank(ascending=False, method="first").astype(int)
        return out.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, booster in self.boosters.items():
            booster.save_model(str(model_dir / f"model_{name}.txt"))
        meta = {**self.meta, "features": self.features, "use_odds": self.use_odds}
        (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> RaceModel:
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_dir}（先に train を実行してください）")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        boosters = {name: lgb.Booster(model_file=str(model_dir / f"model_{name}.txt")) for name in TARGETS}
        return cls(
            boosters=boosters, features=list(meta.pop("features")), use_odds=bool(meta.pop("use_odds")), meta=meta
        )


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    return features[features["is_top3"].notna()].reset_index(drop=True)


def default_valid_from(rows: pd.DataFrame, valid_ratio: float = 0.2) -> date:
    dates = np.sort(rows["race_date"].dt.date.unique())
    if len(dates) < 2:
        raise ValueError("学習データの開催日が少なすぎます")
    return date.fromisoformat(str(dates[min(len(dates) - 1, int(len(dates) * (1 - valid_ratio)))]))


def train(
    features: pd.DataFrame,
    win_payouts: pd.DataFrame,
    *,
    valid_from: date | None = None,
    use_odds: bool = False,
    params: dict[str, object] | None = None,
) -> tuple[RaceModel, dict[str, object]]:
    """時系列で検証してから、全期間で学習し直したモデルと検証結果を返す."""
    rows = training_rows(features)
    valid_from = valid_from or default_valid_from(rows)
    is_valid = rows["race_date"].dt.date >= valid_from
    train_rows, valid_rows = rows[~is_valid], rows[is_valid]
    if train_rows.empty or valid_rows.empty:
        raise ValueError(f"検証開始日 {valid_from} で学習・検証データを分けられません")
    cols = feature_columns(use_odds)
    params = {**DEFAULT_PARAMS, **(params or {})}
    categorical = [c for c in CATEGORICAL_FEATURES if c in cols]
    logger.info("学習 %d 頭 / 検証 %d 頭（検証開始日 %s）", len(train_rows), len(valid_rows), valid_from)

    best_rounds: dict[str, int] = {}
    valid_boosters: dict[str, lgb.Booster] = {}
    for name, label in TARGETS.items():
        dtrain = lgb.Dataset(train_rows[cols], train_rows[label], categorical_feature=categorical)
        dvalid = lgb.Dataset(valid_rows[cols], valid_rows[label], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        valid_boosters[name] = booster
        best_rounds[name] = max(booster.best_iteration, 1)

    valid_model = RaceModel(valid_boosters, cols, use_odds)
    metrics = evaluate(valid_model, valid_rows, win_payouts)
    metrics.update(
        {
            "valid_from": valid_from.isoformat(),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "best_rounds": best_rounds,
        }
    )

    final_boosters = {}
    for name, label in TARGETS.items():
        dall = lgb.Dataset(rows[cols], rows[label], categorical_feature=categorical)
        rounds = int(best_rounds[name] * len(rows) / len(train_rows))
        final_boosters[name] = lgb.train(params, dall, num_boost_round=max(rounds, 1))
    meta: dict[str, object] = {
        "trained_until": rows["race_date"].max().date().isoformat(),
        "trained_from": rows["race_date"].min().date().isoformat(),
        "metrics": metrics,
    }
    return RaceModel(final_boosters, cols, use_odds, meta), metrics


def evaluate(model: RaceModel, rows: pd.DataFrame, win_payouts: pd.DataFrame) -> dict[str, object]:
    pred = model.predict(rows)
    merged = pred.merge(
        rows[["race_id", "horse_number", "is_win", "is_top3", "popularity"]],
        on=["race_id", "horse_number"],
        how="left",
    )
    metrics: dict[str, object] = {
        "races": int(merged["race_id"].nunique()),
        "win_logloss": log_loss(merged["is_win"].to_numpy(), merged["win_prob"].to_numpy()),
        "win_auc": roc_auc(merged["is_win"].to_numpy(), merged["win_prob"].to_numpy()),
        "top3_logloss": log_loss(merged["is_top3"].to_numpy(), merged["top3_prob"].to_numpy()),
        "top3_auc": roc_auc(merged["is_top3"].to_numpy(), merged["top3_prob"].to_numpy()),
    }
    metrics["model_top_pick"] = _pick_stats(merged[merged["pred_rank"] == 1], win_payouts)
    favorites = merged.sort_values(["race_id", "popularity"]).groupby("race_id").head(1)
    metrics["favorite_baseline"] = _pick_stats(favorites[favorites["popularity"].notna()], win_payouts)
    return metrics


def _pick_stats(picks: pd.DataFrame, win_payouts: pd.DataFrame) -> dict[str, float]:
    """1 レース 1 頭を単勝 100 円で買った場合の的中率・回収率."""
    if picks.empty:
        return {"races": 0, "win_hit_rate": float("nan"), "top3_rate": float("nan"), "win_return_rate": float("nan")}
    paid = picks.merge(win_payouts, on=["race_id", "horse_number"], how="left")
    return {
        "races": len(picks),
        "win_hit_rate": float(picks["is_win"].mean()),
        "top3_rate": float(picks["is_top3"].mean()),
        "win_return_rate": float(paid["payout_yen"].fillna(0).sum() / (100 * len(picks))),
    }


def log_loss(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))


def roc_auc(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    yy = np.asarray(y, dtype=float)
    n_pos = int(yy.sum())
    n_neg = len(yy) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(p, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
