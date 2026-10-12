"""LightGBM による着順予想モデルの学習・評価・推論."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
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
OBJECTIVES = ("lambdarank", "binary")
RANK_BOOSTER = "rank"

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
RANK_PARAMS: dict[str, object] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [3],
    "lambdarank_truncation_level": 10,
}
CALIBRATION_RATIO = 0.2
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100


@dataclass
class RaceModel:
    boosters: dict[str, lgb.Booster]
    features: list[str]
    use_odds: bool
    meta: dict[str, object] = field(default_factory=dict)
    objective: str = "binary"
    temperature: float = 1.0
    place_discount: float = 1.0

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """各馬の勝率（レース内で合計 1）・3 着内確率・予想順位を返す."""
        x = frame[self.features]
        out = frame[["race_id", "horse_number", "horse_name"]].copy()
        if self.objective == "lambdarank":
            score = np.asarray(self.boosters[RANK_BOOSTER].predict(x), dtype=float)
            out["win_prob"] = race_softmax(score * self.temperature, out["race_id"])
            out["top3_prob"] = race_top3(out["win_prob"].to_numpy(), out["race_id"], self.place_discount)
        else:
            win_raw = np.asarray(self.boosters["win"].predict(x), dtype=float)
            out["top3_prob"] = np.asarray(self.boosters["top3"].predict(x), dtype=float)
            out["win_prob"] = win_raw / pd.Series(win_raw, index=out.index).groupby(out["race_id"]).transform("sum")
        out["pred_rank"] = out.groupby("race_id")["win_prob"].rank(ascending=False, method="first").astype(int)
        return out.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, booster in self.boosters.items():
            booster.save_model(str(model_dir / f"model_{name}.txt"))
        meta = {
            **self.meta,
            "features": self.features,
            "use_odds": self.use_odds,
            "objective": self.objective,
            "temperature": self.temperature,
            "place_discount": self.place_discount,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> RaceModel:
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_dir}（先に train を実行してください）")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        objective = str(meta.pop("objective", "binary"))
        names = [RANK_BOOSTER] if objective == "lambdarank" else list(TARGETS)
        boosters = {name: lgb.Booster(model_file=str(model_dir / f"model_{name}.txt")) for name in names}
        return cls(
            boosters=boosters,
            features=list(meta.pop("features")),
            use_odds=bool(meta.pop("use_odds")),
            meta=meta,
            objective=objective,
            temperature=float(meta.pop("temperature", 1.0)),
            place_discount=float(meta.pop("place_discount", 1.0)),
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
    objective: str = "lambdarank",
    params: dict[str, object] | None = None,
) -> tuple[RaceModel, dict[str, object]]:
    """時系列で検証してから、全期間で学習し直したモデルと検証結果を返す."""
    if objective not in OBJECTIVES:
        raise ValueError(f"未対応の objective です: {objective}")
    rows = training_rows(features)
    valid_from = valid_from or default_valid_from(rows)
    is_valid = rows["race_date"].dt.date >= valid_from
    train_rows, valid_rows = rows[~is_valid], rows[is_valid]
    if train_rows.empty or valid_rows.empty:
        raise ValueError(f"検証開始日 {valid_from} で学習・検証データを分けられません")
    cols = feature_columns(use_odds)
    is_rank = objective == "lambdarank"
    params = {**DEFAULT_PARAMS, **(RANK_PARAMS if is_rank else {}), **(params or {})}
    categorical = [c for c in CATEGORICAL_FEATURES if c in cols]
    names = [RANK_BOOSTER] if is_rank else list(TARGETS)
    logger.info("学習 %d 頭 / 検証 %d 頭（検証開始日 %s）", len(train_rows), len(valid_rows), valid_from)

    best_rounds: dict[str, int] = {}
    valid_boosters: dict[str, lgb.Booster] = {}
    for name in names:
        dtrain = _dataset(train_rows, cols, name, categorical)
        dvalid = _dataset(valid_rows, cols, name, categorical, reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        valid_boosters[name] = booster
        best_rounds[name] = max(booster.best_iteration, 1)

    calibration = (
        _calibrate(train_rows, cols, params, categorical, best_rounds[RANK_BOOSTER]) if is_rank else (1.0, 1.0)
    )
    valid_model = RaceModel(
        valid_boosters, cols, use_odds, objective=objective, temperature=calibration[0], place_discount=calibration[1]
    )
    metrics = evaluate(valid_model, valid_rows, win_payouts)
    metrics.update(
        {
            "valid_from": valid_from.isoformat(),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "objective": objective,
            "best_rounds": best_rounds,
            "temperature": calibration[0],
            "place_discount": calibration[1],
        }
    )

    final_boosters = {}
    for name in names:
        rounds = int(best_rounds[name] * len(rows) / len(train_rows))
        final_boosters[name] = lgb.train(
            params, _dataset(rows, cols, name, categorical), num_boost_round=max(rounds, 1)
        )
    final_calibration = (
        _calibrate(rows, cols, params, categorical, int(best_rounds[RANK_BOOSTER] * len(rows) / len(train_rows)))
        if is_rank
        else (1.0, 1.0)
    )
    meta: dict[str, object] = {
        "trained_until": rows["race_date"].max().date().isoformat(),
        "trained_from": rows["race_date"].min().date().isoformat(),
        "metrics": metrics,
    }
    return (
        RaceModel(
            final_boosters,
            cols,
            use_odds,
            meta,
            objective=objective,
            temperature=final_calibration[0],
            place_discount=final_calibration[1],
        ),
        metrics,
    )


def relevance(rows: pd.DataFrame) -> npt.NDArray[np.int_]:
    """LambdaRank の関連度: 1 着 3、2 着 2、3 着 1、それ以外 0."""
    pos = rows["finish_position"].to_numpy(dtype=float)
    return np.select([pos == 1, pos == 2, pos == 3], [3, 2, 1], 0)


def _dataset(
    rows: pd.DataFrame,
    cols: list[str],
    name: str,
    categorical: list[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    if name == RANK_BOOSTER:
        group = rows.groupby("race_id", sort=False).size().to_numpy()
        return lgb.Dataset(
            rows[cols], relevance(rows), group=group, categorical_feature=categorical, reference=reference
        )
    return lgb.Dataset(rows[cols], rows[TARGETS[name]], categorical_feature=categorical, reference=reference)


def _calibrate(
    rows: pd.DataFrame, cols: list[str], params: dict[str, object], categorical: list[str], rounds: int
) -> tuple[float, float]:
    """直近 ``CALIBRATION_RATIO`` の開催日を除いて学習し直し、除いた期間で温度と 2・3 着の割引係数を合わせる."""
    calib_from = default_valid_from(rows, CALIBRATION_RATIO)
    is_calib = rows["race_date"].dt.date >= calib_from
    fit_rows, calib_rows = rows[~is_calib], rows[is_calib]
    rounds = max(int(rounds * len(fit_rows) / len(rows)), 1)
    booster = lgb.train(params, _dataset(fit_rows, cols, RANK_BOOSTER, categorical), num_boost_round=rounds)
    score = np.asarray(booster.predict(calib_rows[cols]), dtype=float)
    race_id = calib_rows["race_id"]
    temperature = fit_temperature(score, race_id, calib_rows["is_win"].to_numpy(dtype=float))
    win_prob = race_softmax(score * temperature, race_id)
    discount = fit_place_discount(win_prob, race_id, calib_rows["is_top3"].to_numpy(dtype=float))
    return temperature, discount


def race_softmax(score: npt.ArrayLike, race_id: pd.Series) -> npt.NDArray[np.float64]:
    """レースごとに softmax を取り、合計 1 の勝率にする."""
    s = pd.Series(np.asarray(score, dtype=float), index=race_id.index)
    e = np.exp(s - s.groupby(race_id).transform("max"))
    return np.asarray(e / e.groupby(race_id).transform("sum"), dtype=float)


def fit_temperature(score: npt.ArrayLike, race_id: pd.Series, is_win: npt.ArrayLike) -> float:
    """勝ち馬の対数尤度が最大になる softmax の温度（スコアに掛ける係数）を求める."""
    sc = np.asarray(score, dtype=float)
    win = np.asarray(is_win, dtype=float)

    def neg_loglik(log_t: float) -> float:
        return -_bernoulli_loglik(win, race_softmax(sc * np.exp(log_t), race_id), positive_only=True)

    return float(np.exp(_golden_min(neg_loglik, -4.0, 3.0)))


def fit_place_discount(win_prob: npt.ArrayLike, race_id: pd.Series, is_top3: npt.ArrayLike) -> float:
    """3 着内の対数尤度が最大になる、2・3 着の確率計算に使う割引係数を求める."""
    p = np.asarray(win_prob, dtype=float)
    top3 = np.asarray(is_top3, dtype=float)
    return _golden_min(lambda d: -_bernoulli_loglik(top3, race_top3(p, race_id, d)), 0.2, 1.5)


def _bernoulli_loglik(y: npt.NDArray[np.float64], p: npt.NDArray[np.float64], positive_only: bool = False) -> float:
    pp = np.clip(p, 1e-15, 1 - 1e-15)
    ll = y * np.log(pp) if positive_only else y * np.log(pp) + (1 - y) * np.log(1 - pp)
    return float(np.sum(ll))


def _golden_min(f: Callable[[float], float], lo: float, hi: float, iterations: int = 40) -> float:
    ratio = (np.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = f(b)
    return float((lo + hi) / 2)


def race_top3(win_prob: npt.ArrayLike, race_id: pd.Series, discount: float = 1.0) -> npt.NDArray[np.float64]:
    p = np.asarray(win_prob, dtype=float)
    out = np.empty(len(p))
    for idx in race_id.groupby(race_id.to_numpy(), sort=False).indices.values():
        out[idx] = plackett_luce_top3(p[idx], discount)
    return out


def plackett_luce_top3(win_prob: npt.ArrayLike, discount: float = 1.0) -> npt.NDArray[np.float64]:
    """勝率から Plackett-Luce モデルで各馬の 3 着内確率を求める.

    2・3 着の抽選には勝率を ``discount`` 乗して正規化した強さを使う（1 なら素の Plackett-Luce）。
    """
    p = np.asarray(win_prob, dtype=float)
    if len(p) <= 3:
        return np.ones(len(p))
    p = p / p.sum()
    q = p**discount
    q = q / q.sum()
    n = len(p)
    eye = np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        # first_second[j, i]: j が 1 着、i が 2 着になる確率
        first_second = np.where(eye, 0.0, p[:, None] * q[None, :] / (1 - q)[:, None])
        rest = 1 - q[:, None] - q[None, :]
        third = first_second[:, :, None] * q[None, None, :] / rest[:, :, None]
    distinct = ~(eye[:, :, None] | eye[:, None, :] | eye[None, :, :])
    third = np.where(distinct, np.nan_to_num(third), 0.0)
    return np.asarray(np.clip(p + first_second.sum(axis=0) + third.sum(axis=(0, 1)), 0.0, 1.0), dtype=float)


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
