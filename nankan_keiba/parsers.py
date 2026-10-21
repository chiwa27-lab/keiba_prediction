from __future__ import annotations

import json
import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

from .config import NANKAN_PLACES
from .models import Kaisai, Payout, Pedigree, RaceInfo, RaceResult, ResultEntry

_KAISAI_LINK_RE = re.compile(r"kaisai_date=(\d{8})&(?:amp;)?kaisai_id=(\d{10})")
_RESULT_LINK_RE = re.compile(r"result\.html\?race_id=(\d{12})")
_SHUTUBA_LINK_RE = re.compile(r"shutuba\.html\?race_id=(\d{12})")
_SCRATCH_STATUSES = ("取消", "除外")
_ID_IN_PATH_RE = {
    "horse": re.compile(r"/horse/(\w+)"),
    "jockey": re.compile(r"/jockey/(?:result/recent/)?(\w+)"),
    "trainer": re.compile(r"/trainer/(?:result/recent/)?(\w+)"),
}


class ParseError(Exception):
    pass


def parse_calendar(html: str, place_codes: Iterable[str] = NANKAN_PLACES) -> list[Kaisai]:
    """月間カレンダーから対象競馬場の開催（kaisai_id）を抽出する."""
    codes = set(place_codes)
    found: dict[str, Kaisai] = {}
    for kaisai_date, kaisai_id in _KAISAI_LINK_RE.findall(html):
        place_code = kaisai_id[4:6]
        if place_code in codes and kaisai_id[6:] == kaisai_date[4:]:
            found[kaisai_id] = Kaisai(kaisai_id, kaisai_date, place_code, NANKAN_PLACES.get(place_code, place_code))
    return sorted(found.values(), key=lambda k: k.kaisai_id)


def parse_race_ids(html: str, kaisai_id: str) -> list[str]:
    """レース一覧から指定開催の結果ページ race_id を抽出する."""
    return sorted({rid for rid in _RESULT_LINK_RE.findall(html) if rid.startswith(kaisai_id)})


def parse_shutuba_ids(html: str, kaisai_date: str, place_codes: Iterable[str] = NANKAN_PLACES) -> list[str]:
    """レース一覧から、指定日・対象競馬場の race_id を抽出する（発走後で結果ページへのリンクになったレースも含む）."""
    codes = set(place_codes)
    return sorted(
        rid
        for rid in {*_SHUTUBA_LINK_RE.findall(html), *_RESULT_LINK_RE.findall(html)}
        if rid[4:6] in codes and rid[0:4] == kaisai_date[0:4] and rid[6:10] == kaisai_date[4:8]
    )


def parse_shutuba(html: str, race_id: str) -> RaceResult:
    """出馬表を、着順が空の RaceResult として読み取る.

    出走予定の馬は ``finish_status`` が空文字、取消・除外の馬はその文字列になる。
    ``win_odds`` / ``popularity`` は netkeiba の予想オッズ・人気（確定値ではない）。
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.ShutubaTable")
    if table is None:
        raise ParseError(f"{race_id}: 出馬表が見つかりません（ページ構造変更の可能性）")
    header_row = table.select_one("thead tr") or table.find("tr")
    headers = (
        [re.sub(r"\s+", "", th.get_text("", strip=True)) for th in header_row.find_all("th")]
        if isinstance(header_row, Tag)
        else []
    )

    def col(name: str) -> int | None:
        return next((i for i, h in enumerate(headers) if h.startswith(name)), None)

    idx = {
        "waku": col("枠"),
        "umaban": col("馬番"),
        "horse": col("馬名"),
        "sexage": col("性齢"),
        "kinryo": col("斤量"),
        "jockey": col("騎手"),
        "trainer": col("厩舎"),
        "weight": col("馬体重"),
        "odds": col("予想オッズ"),
        "ninki": col("人気"),
    }
    entries: list[ResultEntry] = []
    for row in table.select("tr.HorseList"):
        cells = row.find_all("td", recursive=False)

        def cell(key: str, cells: list[Tag] = cells) -> Tag | None:
            i = idx[key]
            return cells[i] if i is not None and i < len(cells) else None

        def value(key: str) -> str | None:
            c = cell(key)
            return _text(c) or None if c is not None else None

        status = ""
        row_text = " ".join(_text(c) for c in cells if not c.find("select"))
        classes = row.get("class") or []
        for word in _SCRATCH_STATUSES:
            if word in row_text:
                status = word
        if not status and "Cancel" in classes:
            status = "取消"

        sex, age = _sex_age(value("sexage"))
        affiliation, trainer_name = _trainer(cell("trainer"))
        horse_weight, horse_weight_diff = _horse_weight(value("weight"))
        horse_cell, jockey_cell, trainer_cell = cell("horse"), cell("jockey"), cell("trainer")
        entries.append(
            ResultEntry(
                race_id=race_id,
                finish_position=None,
                finish_status=status,
                bracket_number=_to_int(value("waku")),
                horse_number=_to_int(value("umaban")),
                horse_id=_link_id(horse_cell, "horse") if horse_cell is not None else None,
                horse_name=_text(horse_cell.select_one(".HorseName") or horse_cell) if horse_cell is not None else "",
                sex=sex,
                age=age,
                weight_carried=_to_float(value("kinryo")),
                jockey_id=_link_id(jockey_cell, "jockey") if jockey_cell is not None else None,
                jockey_name=_text(jockey_cell) or None,
                finish_time=None,
                finish_time_sec=None,
                margin=None,
                popularity=_to_int(value("ninki")) if not status else None,
                win_odds=_to_float(value("odds")) if not status else None,
                last_3f=None,
                trainer_affiliation=affiliation,
                trainer_id=_link_id(trainer_cell, "trainer") if trainer_cell is not None else None,
                trainer_name=trainer_name,
                horse_weight=horse_weight,
                horse_weight_diff=horse_weight_diff,
            )
        )
    if not entries:
        raise ParseError(f"{race_id}: 出走馬が見つかりません")
    return RaceResult(race=_parse_race_info(soup, race_id), entries=entries)


def parse_pedigree(html: str, horse_id: str) -> Pedigree:
    """5 代血統表から父・父の父・母・母の父と、父の系統（例: 'Mr. Prospector系'）を読み取る."""
    table = BeautifulSoup(html, "lxml").select_one("table.blood_table")
    if table is None:
        raise ParseError(f"{horse_id}: 血統表が見つかりません（ページ構造変更の可能性）")
    parents = table.find_all("td", rowspan="16")
    if len(parents) < 2:
        raise ParseError(f"{horse_id}: 血統表の父・母が読み取れません")
    sire, dam = parents[0], parents[1]
    sire_id, sire_name = _pedigree_horse(sire)
    sire_sire_id, sire_sire_name = _pedigree_horse(sire.find_next_sibling("td"))
    dam_id, dam_name = _pedigree_horse(dam)
    damsire_id, damsire_name = _pedigree_horse(dam.find_next_sibling("td"))
    sire_line = next((s for s in sire.stripped_strings if s.endswith("系")), None)
    return Pedigree(
        horse_id=horse_id,
        sire_id=sire_id,
        sire_name=sire_name,
        sire_line=sire_line,
        sire_sire_id=sire_sire_id,
        sire_sire_name=sire_sire_name,
        dam_id=dam_id,
        dam_name=dam_name,
        damsire_id=damsire_id,
        damsire_name=damsire_name,
    )


def _pedigree_horse(cell: object) -> tuple[str | None, str | None]:
    if not isinstance(cell, Tag):
        return None, None
    link = cell.find("a", href=_ID_IN_PATH_RE["horse"])
    if not isinstance(link, Tag):
        return None, None
    return _link_id(cell, "horse"), _text(link) or None


def _sex_age(text: str | None) -> tuple[str | None, int | None]:
    m = re.match(r"(\D+)(\d+)", text or "")
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _trainer(cell: Tag | None) -> tuple[str | None, str | None]:
    if cell is None:
        return None, None
    label = cell.find("span")
    link = cell.find("a")
    affiliation = _text(label) or None if isinstance(label, Tag) else None
    name = _text(link) or None if isinstance(link, Tag) else None
    return affiliation, name


def _horse_weight(text: str | None) -> tuple[int | None, int | None]:
    m = re.match(r"(\d+)\s*(?:\(\s*([-+]?\d+)\s*\))?", text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def parse_result(html: str, race_id: str) -> RaceResult:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table#All_Result_Table")
    if table is None:
        raise ParseError(f"{race_id}: 結果テーブルが見つかりません（未確定・中止・ページ構造変更の可能性）")
    entries = _parse_entries(table, race_id)
    if not entries:
        raise ParseError(f"{race_id}: 着順データが空です")
    race = _parse_race_info(soup, race_id)
    payouts = _parse_payouts(soup, race_id)
    return RaceResult(race=race, entries=entries, payouts=payouts)


def _text(node: Tag | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.search(r"[-+]?\d[\d,]*", value)
    return int(m.group(0).replace(",", "")) if m else None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(m.group(0)) if m else None


def _time_to_sec(value: str) -> float | None:
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", value.strip())
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _link_id(cell: Tag, kind: str) -> str | None:
    link = cell.find("a", href=_ID_IN_PATH_RE[kind])
    if not isinstance(link, Tag):
        return None
    m = _ID_IN_PATH_RE[kind].search(str(link.get("href", "")))
    return m.group(1) if m else None


def _parse_entries(table: Tag, race_id: str) -> list[ResultEntry]:
    header_row = table.select_one("thead tr") or table.find("tr")
    if not isinstance(header_row, Tag):
        return []
    headers = [re.sub(r"\s+", "", th.get_text("", strip=True)) for th in header_row.find_all("th")]

    def col(name: str) -> int | None:
        for i, h in enumerate(headers):
            if h.startswith(name):
                return i
        return None

    idx = {
        "rank": col("着順"),
        "waku": col("枠"),
        "umaban": col("馬番"),
        "horse": col("馬名"),
        "sexage": col("性齢"),
        "kinryo": col("斤量"),
        "jockey": col("騎手"),
        "time": col("タイム"),
        "margin": col("着差"),
        "ninki": col("人気"),
        "odds": col("単勝"),
        "last3f": col("後3F"),
        "trainer": col("厩舎"),
        "weight": col("馬体重"),
    }

    entries: list[ResultEntry] = []
    body_rows = table.select("tbody tr") or table.find_all("tr")[1:]
    for row in body_rows:
        cells = row.find_all("td", recursive=False)
        if len(cells) < len(headers):
            continue

        def cell(key: str, cells: list[Tag] = cells) -> Tag | None:
            i = idx[key]
            return cells[i] if i is not None else None

        def value(key: str) -> str | None:
            c = cell(key)
            return _text(c) or None if c is not None else None

        rank_text = value("rank") or ""
        sex, age = _sex_age(value("sexage"))

        horse_cell = cell("horse")
        jockey_cell = cell("jockey")
        trainer_cell = cell("trainer")
        affiliation, trainer_name = _trainer(trainer_cell)
        horse_weight, horse_weight_diff = _horse_weight(value("weight"))

        finish_time = value("time")
        entries.append(
            ResultEntry(
                race_id=race_id,
                finish_position=int(rank_text) if rank_text.isdigit() else None,
                finish_status=rank_text,
                bracket_number=_to_int(value("waku")),
                horse_number=_to_int(value("umaban")),
                horse_id=_link_id(horse_cell, "horse") if horse_cell is not None else None,
                horse_name=_text(horse_cell),
                sex=sex,
                age=age,
                weight_carried=_to_float(value("kinryo")),
                jockey_id=_link_id(jockey_cell, "jockey") if jockey_cell is not None else None,
                jockey_name=_text(jockey_cell) or None,
                finish_time=finish_time,
                finish_time_sec=_time_to_sec(finish_time) if finish_time else None,
                margin=value("margin"),
                popularity=_to_int(value("ninki")),
                win_odds=_to_float(value("odds")),
                last_3f=_to_float(value("last3f")),
                trainer_affiliation=affiliation,
                trainer_id=_link_id(trainer_cell, "trainer") if trainer_cell is not None else None,
                trainer_name=trainer_name,
                horse_weight=horse_weight,
                horse_weight_diff=horse_weight_diff,
            )
        )
    return entries


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> RaceInfo:
    place_code = race_id[4:6]
    race_date = f"{race_id[0:4]}-{race_id[6:8]}-{race_id[8:10]}"

    name_node = soup.select_one(".RaceName")
    grade = None
    race_name = ""
    if isinstance(name_node, Tag):
        grade_node = name_node.select_one("[class*=Icon_Grade]")
        if isinstance(grade_node, Tag):
            grade = _text(grade_node) or None
            grade_node.extract()
        race_name = _text(name_node)

    data01 = _text(soup.select_one(".RaceData01"))
    post_time = _search(r"(\d{1,2}:\d{2})\s*発走", data01)
    course = re.search(r"(芝|ダ|障)\s*(\d+)m", data01)
    direction = _search(r"\(([^)]*)\)", data01)
    weather = _known(_search(r"天候\s*:\s*([^\s/]+)", data01))
    track_condition = _known(_search(r"馬場\s*:\s*([^\s/]+)", data01))

    kai = nichi = num_horses = None
    prize_money = None
    class_parts: list[str] = []
    data02 = soup.select_one(".RaceData02")
    if isinstance(data02, Tag):
        for span in data02.find_all("span"):
            t = _text(span)
            if not t:
                continue
            if m := re.fullmatch(r"(\d+)回", t):
                kai = int(m.group(1))
            elif m := re.fullmatch(r"(\d+)日目", t):
                nichi = int(m.group(1))
            elif m := re.fullmatch(r"(\d+)頭", t):
                num_horses = int(m.group(1))
            elif t.startswith("本賞金"):
                prize_money = t.split(":", 1)[-1].replace("万円", "").replace("、", ",").strip()
            elif t == NANKAN_PLACES.get(place_code):
                continue
            else:
                class_parts.append(t)

    return RaceInfo(
        race_id=race_id,
        race_date=race_date,
        place_code=place_code,
        place_name=NANKAN_PLACES.get(place_code, place_code),
        race_number=int(race_id[10:12]),
        race_name=race_name,
        grade=grade,
        post_time=post_time,
        surface=course.group(1) if course else None,
        distance=int(course.group(2)) if course else None,
        direction=direction,
        weather=weather,
        track_condition=track_condition,
        kai=kai,
        nichi=nichi,
        race_class=" ".join(class_parts) or None,
        num_horses=num_horses,
        prize_money=prize_money,
        corner_passing=_parse_corners(soup),
        lap_times=_parse_laps(soup),
    )


def _known(value: str | None) -> str | None:
    """発表前の '−' などは欠損扱いにする."""
    return value if value and re.search(r"\w", value) else None


def _search(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def _br_split(node: Tag | None) -> list[str]:
    if node is None:
        return []
    for br in node.find_all("br"):
        br.replace_with("\n")
    return [s.strip() for s in node.get_text().split("\n") if s.strip()]


def _parse_payouts(soup: BeautifulSoup, race_id: str) -> list[Payout]:
    payouts: list[Payout] = []
    for row in soup.select("table.Payout_Detail_Table tr"):
        bet_type = _text(row.find("th"))
        result_cell = row.select_one("td.Result")
        if not bet_type or result_cell is None:
            continue
        uls = result_cell.find_all("ul")
        if uls:
            combos = ["-".join(t for li in ul.find_all("li") if (t := _text(li))) for ul in uls]
        else:
            combos = [t for span in result_cell.find_all("span") if (t := _text(span))]
        amounts = _br_split(row.select_one("td.Payout"))
        ninkis = _br_split(row.select_one("td.Ninki"))
        for i, combo in enumerate(combos):
            if not combo:
                continue
            payouts.append(
                Payout(
                    race_id=race_id,
                    bet_type=bet_type,
                    combination=combo,
                    payout_yen=_to_int(amounts[i]) if i < len(amounts) else None,
                    popularity=_to_int(ninkis[i]) if i < len(ninkis) else None,
                )
            )
    return payouts


def _parse_corners(soup: BeautifulSoup) -> str | None:
    corners: dict[str, str] = {}
    for row in soup.select("table.Corner_Num tr"):
        name = _text(row.find("th"))
        td = row.find("td")
        if name and isinstance(td, Tag):
            corners[name.replace(" ", "")] = td.get_text("", strip=True)
    return json.dumps(corners, ensure_ascii=False) if corners else None


def _parse_laps(soup: BeautifulSoup) -> str | None:
    table = soup.select_one("table.Race_HaronTime")
    if not isinstance(table, Tag):
        return None
    distances = [_text(th) for th in table.select("tr.Header th")]
    rows = [[_text(td) for td in tr.find_all("td")] for tr in table.select("tr.HaronTime")]
    if not distances or not rows:
        return None
    laps = {
        "distances": distances,
        "cumulative": rows[0],
        "sections": rows[1] if len(rows) > 1 else [],
    }
    return json.dumps(laps, ensure_ascii=False)
