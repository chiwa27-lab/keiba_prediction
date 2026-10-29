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
SPAT4_SHIKI_NAMES = {1: "単勝", 2: "複勝", 5: "馬複", 7: "ワイド", 8: "三連複", 9: "三連単"}
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
    # Chrome は相対パスのプロファイルを開けないため、フルパスで渡す
    profile = (data_dir / "spat4_profile").resolve()
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


VOTE_LINK = "a[target=spat4_LiveVote] span.btn_live"


def _find_menu(context: Any, timeout_ms: int = 0) -> Any:
    """ログイン済みのメニュー画面（通常投票ボタンがあるタブ）を探す。ログインすると別のタブに開くことがある."""
    from playwright.sync_api import Error as PlaywrightError

    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for page in context.pages:
            try:
                if "spat4.jp" in page.url and page.locator(VOTE_LINK).count():
                    return page
            except PlaywrightError:
                pass  # ログイン直後など、画面の切り替え中は読めない
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)


def _login(context: Any, settings: Spat4Settings) -> Any:
    page = next((p for p in context.pages if "spat4.jp/keiba/pc" in p.url), None)
    if page is None:
        page = context.new_page()
        page.goto(SPAT4_URL)
    if not page.locator("#MEMBERNUMR").is_visible():
        raise Spat4BrowserError(
            "SPAT4 のメニュー画面が見つかりません。SPAT4 用の Chrome でログインしてからやり直してください"
        )
    # 「かんたんログイン」設定中は入力欄が無効になり、ログインボタンだけで入れる
    if page.locator("#MEMBERNUMR").is_enabled():
        if not settings.complete:
            raise Spat4BrowserError(
                "設定で加入者番号と利用者IDを保存するか、SPAT4 用の Chrome でログインしてからやり直してください"
            )
        page.fill("#MEMBERNUMR", settings.member_number)
        page.fill("#MEMBERIDR", settings.user_id)
    # ボタンの <a> は幅0で描画されていて普通のクリックが届かないため、click イベントを直接送る
    page.dispatch_event("a.loginbtn", "click")
    menu = _find_menu(context, TIMEOUT_MS)
    if menu is None:
        raise Spat4BrowserError("SPAT4 にログインできませんでした。加入者番号と利用者IDを確かめてください")
    return menu


def _vote_page(context: Any, settings: Spat4Settings) -> Any:
    for page in context.pages:
        if "HANDLERR=L000S" in page.url:
            return page
    menu = _find_menu(context) or _login(context, settings)
    with context.expect_page(timeout=TIMEOUT_MS) as info:
        menu.click(VOTE_LINK)
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


def _is_current(page: Any, selector: str) -> bool:
    return "current" in (page.locator(selector).get_attribute("class") or "").split()


def _select_shiki(page: Any, ticket: Ticket) -> None:
    """券種ボタンを目的の1つだけ選んだ状態にする。単勝と複勝は同時に選べるトグルになっている."""
    target = f"#_Inp_shiki_{ticket.shiki}"
    if page.locator(target).count() == 0 or "inactive" in (page.locator(target).get_attribute("class") or ""):
        raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R でこの券種は発売していません")
    if not _is_current(page, target):
        page.click(target)
        page.wait_for_timeout(300)
    for other in (1, 2):
        selector = f"#_Inp_shiki_{other}"
        if other != ticket.shiki and page.locator(selector).count() and _is_current(page, selector):
            page.click(selector)
            page.wait_for_timeout(300)
    if not _is_current(page, target):
        raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R の券種を選べませんでした")


def _enter_ticket(page: Any, ticket: Ticket, expected: int) -> None:
    """1点ずつ馬番を選び、「セットして投票を続ける」で確定する（同じレースの同じ券種でも1点ずつ）."""
    _select_shiki(page, ticket)
    for column, horse in enumerate(ticket.horses, start=1):
        row = page.locator(f"#ent_tbl tr.entry:has(span.num-um{horse})")
        if row.count() == 0:
            raise Spat4BrowserError(f"{ticket.place} {ticket.race_no}R に {horse} 番の馬がいません")
        row.locator(f"span.td-no{column}").click()
    _wait_cart(page, expected)
    page.click("#gotoNext")
    page.wait_for_selector("#gotoNext.inactive", state="attached")
    _wait_cart(page, expected)


CFM_ROWS_JS = """() => [...document.querySelectorAll('#Cfm tr')].filter(tr => tr.querySelector('td.kaime')).map(tr => [
    tr.querySelector('td.race p').textContent.replace(/\\s+/g, ' ').trim(),
    tr.querySelector('td.shikibetsu').textContent.trim(),
    [...tr.querySelectorAll('td.kaime [class*=num-um-s]')].map(e => e.className.match(/num-um-s(\\d+)/)[1]).join('-'),
    tr.querySelector('td.kingaku').textContent.trim()])"""


def _ticket_key(
    place: str, race_no: int, shiki_name: str, horses: tuple[int, ...]
) -> tuple[str, int, str, tuple[int, ...]]:
    return place, race_no, shiki_name, horses if shiki_name == "三連単" else tuple(sorted(horses))


def _verify_confirmation(page: Any, tickets: list[Ticket]) -> None:
    """確認画面の買い目が、入力したかった買い目とすべて一致するか確かめる."""
    shown = []
    for race, shiki_name, horses, amount in page.evaluate(CFM_ROWS_JS):
        place, _, race_no = race.partition(" ")
        if amount.replace(",", "") != "100円" or not horses:
            raise Spat4BrowserError(f"確認画面の買い目が想定と違います（{race} {shiki_name} {amount}）")
        shown.append(_ticket_key(place, int(race_no.rstrip("R")), shiki_name, tuple(int(h) for h in horses.split("-"))))
    expected = [_ticket_key(t.place, t.race_no, SPAT4_SHIKI_NAMES[t.shiki], t.horses) for t in tickets]
    if sorted(shown) != sorted(expected):
        raise Spat4BrowserError(
            "確認画面の買い目が入力したかった買い目と一致しません。投票せずに、SPAT4 の画面で買い目を全削除してください"
        )


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
                _enter_ticket(page, ticket, n)
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
            _verify_confirmation(page, tickets)
            page.fill("#Cfm input[name=cfm_ansho]", pin)
            page.fill("#Cfm input[name=cfm_amount]", str(total_yen))
            page.bring_to_front()
    except PlaywrightError as exc:
        raise Spat4BrowserError(
            f"SPAT4 の画面を操作できませんでした: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        ) from None
