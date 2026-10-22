from __future__ import annotations

from datetime import date

import pandas as pd

from nankan_keiba.model import RaceModel
from nankan_keiba.simulate import leaks_training, payout_table, settle, tickets


def test_tickets_follow_predicted_marks() -> None:
    t = tickets([5, 2, 9, 1, 7])
    assert t["単勝"] == {(5,)}
    assert t["複勝"] == {(5,)}
    assert t["ワイド"] == t["馬連"] == {(2, 5), (5, 9), (1, 5)}
    assert t["3連単"] == {(5, 2, 9), (5, 2, 1), (5, 2, 7), (5, 9, 2), (5, 9, 1), (5, 9, 7)}
    # ◎-○-▲ は ◎○軸と ◎▲軸で重なるので 1 点
    assert t["3連複"] == {(2, 5, 9), (1, 2, 5), (2, 5, 7), (1, 5, 9), (5, 7, 9)}


def test_tickets_small_field() -> None:
    t = tickets([3, 1])
    assert t["ワイド"] == {(1, 3)}
    assert t["3連複"] == set()
    assert tickets([])["単勝"] == set()


def test_payout_table_normalizes_unordered_combinations() -> None:
    payouts = pd.DataFrame(
        [("r", "馬連", "13-6", 570), ("r", "3連単", "13-6-11", 3840), ("r", "枠連", "3-7", 400)],
        columns=["race_id", "bet_type", "combination", "payout_yen"],
    )
    table = payout_table(payouts)
    assert table[("r", "馬連")] == {(6, 13): 570}
    assert table[("r", "3連単")] == {(13, 6, 11): 3840}
    assert ("r", "枠連") not in table


def test_settle_counts_hits_and_skips_unsold_bet_types() -> None:
    pred = pd.DataFrame(
        {
            "race_id": ["r1"] * 5 + ["r2"] * 4,
            "race_date": ["2026-10-21"] * 9,
            "horse_number": [13, 6, 11, 2, 4, 1, 2, 3, 4],
            "pred_rank": [1, 2, 3, 4, 5, 1, 2, 3, 4],
        }
    )
    payouts = pd.DataFrame(
        [
            ("r1", "単勝", "13", 460),
            ("r1", "複勝", "13", 120),
            ("r1", "複勝", "6", 110),
            ("r1", "複勝", "11", 120),
            ("r1", "ワイド", "6-13", 260),
            ("r1", "ワイド", "11-13", 300),
            ("r1", "ワイド", "6-11", 200),
            ("r1", "馬連", "6-13", 570),
            ("r1", "3連複", "6-11-13", 570),
            ("r1", "3連単", "13-6-11", 3840),
            ("r2", "単勝", "3", 800),
        ],
        columns=["race_id", "bet_type", "combination", "payout_yen"],
    )
    summary, details = settle(pred, payouts)
    by_bet = summary.set_index("券種")
    assert by_bet.loc["単勝", "レース数"] == 2
    assert by_bet.loc["単勝", "投資額"] == 200
    assert by_bet.loc["単勝", "払戻額"] == 460
    assert by_bet.loc["単勝", "的中レース数"] == 1
    assert by_bet.loc["複勝", "レース数"] == 1
    assert by_bet.loc["ワイド", "投資額"] == 300
    assert by_bet.loc["ワイド", "払戻額"] == 560
    assert by_bet.loc["3連複", "購入点数"] == 5
    assert by_bet.loc["3連単", "購入点数"] == 6
    assert by_bet.loc["3連単", "回収率"] == 3840 / 600 * 100
    total = by_bet.loc["合計"]
    assert total["投資額"] == details["投資額"].sum() == 100 + 100 + 100 + 300 + 300 + 500 + 600
    assert total["払戻額"] == 460 + 120 + 560 + 570 + 570 + 3840
    assert pd.isna(total["的中レース数"])
    assert details.loc[details["券種"] == "ワイド", "的中"].iloc[0] == "6-13 / 11-13"


def test_leaks_training() -> None:
    model = RaceModel({}, [], False, {"trained_until": "2026-10-20"})
    assert leaks_training(model, date(2026, 10, 20))
    assert not leaks_training(model, date(2026, 10, 21))
