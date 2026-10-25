"""ご自身の PC の Chrome で SPAT4 を開き、買い目を入力して最終確認画面で止める（投票ボタンは押さない）."""

from __future__ import annotations

import pandas as pd

from .spat4 import Spat4Settings

SPAT4_URL = "https://www.spat4.jp/keiba/pc"


class Spat4BrowserError(RuntimeError):
    pass


def enter_votes(settings: Spat4Settings, pin: str, ticket_rows: pd.DataFrame) -> None:
    raise Spat4BrowserError("SPAT4 の画面への入力はまだ準備中です")
