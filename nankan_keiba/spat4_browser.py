"""ご自身の PC の Chrome で SPAT4 を開き、買い目を入力して最終確認画面で止める（投票ボタンは押さない）."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import NANKAN_PLACES
from .spat4 import Spat4Settings

SPAT4_URL = "https://www.spat4.jp/keiba/pc"
CDP_PORT = 9333
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"
MAX_TICKETS = 120
# SPAT4 の通常投票画面の式別ボタン（#_Inp_shiki_N）
SHIKI_IDS = {"単勝": 1, "複勝": 2, "馬連": 5, "ワイド": 7, "3連複": 8, "3連単": 9}
TIMEOUT_MS = 15_000


class Spat4BrowserError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ticket:
    place: str
    race_no: int
    shiki: int
    horses: tuple[int, ...]


def plan(ticket_rows: pd.DataFrame) -> list[Ticket]:
    """買い目の表を、SPAT4 の画面で押すボタンの順に並べ替える（レースごとにまとめる）."""
    if len(ticket_rows) > MAX_TICKETS:
        raise Spat4BrowserError(f"SPAT4 で一度に入力できるのは {MAX_TICKETS} 点までです（今回 {len(ticket_rows)} 点）")
    out = []
    for row in ticket_rows.sort_values("race_id", kind="stable").itertuples(index=False):
        race_id = str(row.race_id)
        bet_type = str(row.券種)
        if bet_type not in SHIKI_IDS:
            raise Spat4BrowserError(f"{bet_type} は SPAT4 への入力に対応していません")
        horses = tuple(int(n) for n in str(row.買い目).replace("→", "-").split("-"))
        out.append(
            Ticket(
                place=NANKAN_PLACES[race_id[4:6]],
                race_no=int(race_id[10:12]),
                shiki=SHIKI_IDS[bet_type],
                horses=horses,
            )
        )
    return out


def _find_chrome() -> str:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", "")),
    ]
    for base in candidates:
        exe = base / "Google" / "Chrome" / "Application" / "chrome.exe"
        if exe.is_file():
            return str(exe)
    for name in ("google-chrome", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    raise Spat4BrowserError("Google Chrome が見つかりません。インストールしてください")


def _cdp_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{CDP_URL}/json/version", timeout=2):
            return True
    except OSError:
        return False


def _ensure_chrome(data_dir: Path) -> None:
    """SPAT4 用の Chrome（普段の Chrome とは別のプロファイル）を開く。開いていればそのまま使う."""
    if _cdp_ready():
        return
    profile = data_dir / "spat4_profile"
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(  # noqa: S603
        [
            _find_chrome(),
            f"--remote-debugging-port={CDP_PORT}",
            "--remote-allow-origins=http://127.0.0.1",
            f"--user-data-dir={profile}",
            "--no-first-run",
            SPAT4_URL,
        ]
    )
    for _ in range(30):
        if _cdp_ready():
            return
        time.sleep(0.5)
    raise Spat4BrowserError("SPAT4 用の Chrome を起動できませんでした")


def _check_dialog(page: Any) -> None:
    """SPAT4 の画面内のお知らせ（締切など）が出ていたら閉じて、内容をエラーにする."""
    dialog = page.locator("div.dialog").first
    if not dialog.is_visible():
        return
    message = " ".join(dialog.locator(".msg p").first.inner_text().split())
    for button in ("#Common_alertYes", "#Common_dialogNo"):
        if page.locator(button).is_visible():
            page.click(button)
            break
    raise Spat4BrowserError(f"SPAT4: {message}")


def _cart_count(page: Any) -> int:
    return int(page.inner_text("#sumKensu").strip() or 0)


def _wait_cart(page: Any, expected: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _check_dialog(page)
        if _cart_count(page) == expected:
            return
        page.wait_for_timeout(200)
    raise Spat4BrowserError(
        f"買い目の数が合いません（SPAT4 側 {_cart_count(page)} 件、入力したい件数 {expected} 件）。"
        "SPAT4 の画面で買い目を全削除してから、やり直してください"
    )


def _login(page: Any, settings: Spat4Settings) -> None:
    if not page.locator("#MEMBERNUMR").is_visible():
        return
    page.fill("#MEMBERNUMR", settings.member_number)
    page.fill("#MEMBERIDR", settings.user_id)
    page.click("a.loginbtn")
    page.wait_for_load_state()
    if page.locator("#MEMBERNUMR").is_visible():
        raise Spat4BrowserError("SPAT4 にログインできませんでした。加入者番号と利用者IDを確かめてください")


def _vote_page(context: Any, settings: Spat4Settings) -> Any:
    for page in context.pages:
        if "HANDLERR=L000S" in page.url:
            return page
    menu = next((p for p in context.pages if "spat4.jp/keiba/pc" in p.url), None)
    if menu is None:
        menu = context.new_page()
        menu.goto(SPAT4_URL)
    _login(menu, settings)
    with context.expect_page(timeout=TIMEOUT_MS) as info:
        menu.click("a[target=spat4_LiveVote] span.btn_live")
    page = info.value
    page.wait_for_selector("#Bet1", state="visible", timeout=TIMEOUT_MS)
    return page


def _select_race(page: Any, ticket: Ticket) -> None:
    place = page.locator("span[id^=_Inp_jo_]", has_text=ticket.place)
    if place.count() == 0:
        raise Spat4BrowserError(f"SPAT4 で今日の{ticket.place}が発売されていません")
    place.first.click()
    page.wait_for_timeout(500)
    _check_dialog(page)
    race = page.locator(f"#_Inp_race_{ticket.race_no:02d}")
    if race.count() == 0 or "inactive" in (race.get_attribute("class") or ""):
        raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R は発売していません（締切後など）")
    race.click()
    page.wait_for_timeout(500)
    _check_dialog(page)


def _enter_ticket(page: Any, ticket: Ticket) -> None:
    shiki = page.locator(f"#_Inp_shiki_{ticket.shiki}")
    if shiki.count() == 0:
        raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R でこの券種は発売していません")
    shiki.click()
    page.wait_for_timeout(300)
    for column, horse in enumerate(ticket.horses, start=1):
        row = page.locator(f"#ent_tbl tr.entry:has(span.num-um{horse})")
        if row.count() == 0:
            raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R に {horse} 番の馬がいません")
        row.locator(f"span.td-no{column}").click()


def enter_votes(
    settings: Spat4Settings,
    pin: str,
    ticket_rows: pd.DataFrame,
    data_dir: Path,
) -> None:
    """買い目を SPAT4 に入力し、暗証番号と合計金額まで入れて最終確認画面で止める.

    「投票する」ボタンは押さない。暗証番号はこの関数の中でだけ使い、保存もログ出力もしない。
    """
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    tickets = plan(ticket_rows)
    if not tickets:
        raise Spat4BrowserError("入力する買い目がありません")
    total_yen = 100 * len(tickets)
    _ensure_chrome(data_dir)
    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            context = browser.contexts[0]
            page = _vote_page(context, settings)
            page.set_default_timeout(TIMEOUT_MS)
            page.bring_to_front()
            _check_dialog(page)
            if not page.locator("#Bet2").is_visible():
                raise Spat4BrowserError(
                    "SPAT4 の投票画面が買い目選択の状態ではありません。画面を閉じてからやり直してください"
                )
            if _cart_count(page) != 0:
                raise Spat4BrowserError("SPAT4 の画面にすでに選んだ買い目があります。全削除してからやり直してください")
            current = None
            for n, ticket in enumerate(tickets, start=1):
                key = (ticket.place, ticket.race_no)
                if key != current:
                    _select_race(page, ticket)
                    current = key
                _enter_ticket(page, ticket)
                _wait_cart(page, n)
            page.click("#gotoBuy")
            page.wait_for_selector("#Buy1", state="visible")
            _check_dialog(page)
            page.fill("#Buy1 input[name=buy_all]", "1")
            page.click("#all_set")
            page.wait_for_timeout(500)
            shown_total = page.inner_text("#Buy2 .sum .n2").replace(",", "").strip()
            if shown_total != str(total_yen):
                raise Spat4BrowserError(f"合計金額が合いません（SPAT4 側 {shown_total} 円、予定 {total_yen} 円）")
            page.click("#gotoCfm-buy")
            page.wait_for_selector("#Cfm", state="visible")
            _check_dialog(page)
            page.fill("#Cfm input[name=cfm_ansho]", pin)
            page.fill("#Cfm input[name=cfm_amount]", str(total_yen))
            page.bring_to_front()
    except PlaywrightError as exc:
        raise Spat4BrowserError(f"SPAT4 の画面を操作できませんでした: {type(exc).__name__}") from None
