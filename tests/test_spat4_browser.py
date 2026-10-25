import pandas as pd
import pytest

from nankan_keiba.spat4_browser import MAX_TICKETS, Spat4BrowserError, Ticket, plan


def _rows(rows):
    return pd.DataFrame(rows, columns=["race_id", "券種", "買い目", "金額"])


def test_plan_groups_by_race_and_maps_bet_types():
    rows = _rows(
        [
            ["202645102512", "3連単", "3→1→7", 100],
            ["202642102508", "単勝", "5", 100],
            ["202642102508", "ワイド", "2-5", 100],
            ["202645102512", "馬連", "1-3", 100],
        ]
    )
    assert plan(rows) == [
        Ticket("浦和", 8, 1, (5,)),
        Ticket("浦和", 8, 7, (2, 5)),
        Ticket("川崎", 12, 9, (3, 1, 7)),
        Ticket("川崎", 12, 5, (1, 3)),
    ]


def test_plan_rejects_more_than_spat4_limit():
    rows = _rows([["202644102501", "単勝", str(i % 16 + 1), 100] for i in range(MAX_TICKETS + 1)])
    with pytest.raises(Spat4BrowserError, match="120"):
        plan(rows)
