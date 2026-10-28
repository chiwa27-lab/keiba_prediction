import pandas as pd
import pytest

from nankan_keiba.spat4_browser import MAX_TICKETS, Spat4BrowserError, Ticket, _verify_confirmation, plan


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


class _FakePage:
    def __init__(self, rows):
        self.rows = rows

    def evaluate(self, _js):
        return self.rows


def test_verify_confirmation_accepts_matching_tickets_in_any_order():
    tickets = [Ticket("大井", 1, 9, (2, 1, 3)), Ticket("大井", 1, 5, (1, 3))]
    page = _FakePage([["大井 1R", "馬複", "1-3", "100円"], ["大井 1R", "三連単", "2-1-3", "100円"]])
    _verify_confirmation(page, tickets)


def test_verify_confirmation_rejects_wrong_order_for_trifecta():
    tickets = [Ticket("大井", 1, 9, (2, 1, 3))]
    page = _FakePage([["大井 1R", "三連単", "1-2-3", "100円"]])
    with pytest.raises(Spat4BrowserError, match="一致しません"):
        _verify_confirmation(page, tickets)
