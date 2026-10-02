from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta

from .client import NotFoundError, PoliteHttpClient
from .config import NANKAN_PLACES, calendar_url, race_list_url, race_result_url
from .models import Kaisai
from .parsers import ParseError, parse_calendar, parse_race_ids, parse_result
from .storage import RaceStore

logger = logging.getLogger(__name__)


@dataclass
class CollectStats:
    kaisai: int = 0
    races_found: int = 0
    races_saved: int = 0
    races_skipped: int = 0
    failed_race_ids: list[str] = field(default_factory=list)


def default_period(today: date) -> tuple[date, date]:
    """前日までの過去 1 年間."""
    end = today - timedelta(days=1)
    return end - timedelta(days=364), end


def iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _month_end(year: int, month: int) -> date:
    first_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first_next - timedelta(days=1)


class Collector:
    def __init__(
        self,
        client: PoliteHttpClient,
        store: RaceStore,
        *,
        place_codes: Iterable[str] = NANKAN_PLACES,
        today: date | None = None,
        refetch: bool = False,
    ) -> None:
        self.client = client
        self.store = store
        self.place_codes = tuple(place_codes)
        self.today = today or date.today()
        self.refetch = refetch

    def find_kaisai(self, start: date, end: date) -> list[Kaisai]:
        kaisai: list[Kaisai] = []
        for year, month in iter_months(start, end):
            url = calendar_url(year, month)
            # 当月以降のカレンダーは更新されうるのでキャッシュしない
            html = self.client.get(url, use_cache=_month_end(year, month) < self.today)
            for k in parse_calendar(html, self.place_codes):
                if start.strftime("%Y%m%d") <= k.kaisai_date <= end.strftime("%Y%m%d"):
                    kaisai.append(k)
        logger.info("%s〜%s の対象開催: %d 日分", start, end, len(kaisai))
        return kaisai

    def find_race_ids(self, kaisai: Kaisai) -> list[str]:
        url = race_list_url(kaisai.kaisai_date)
        race_ids = parse_race_ids(self.client.get(url), kaisai.kaisai_id)
        if not race_ids:
            self.client.invalidate(url)
            logger.warning("%s %s: レースが見つかりません", kaisai.kaisai_date, kaisai.place_name)
        return race_ids

    def collect_race(self, race_id: str) -> bool:
        url = race_result_url(race_id)
        try:
            result = parse_result(self.client.get(url, use_cache=not self.refetch), race_id)
        except (ParseError, NotFoundError) as exc:
            self.client.invalidate(url)
            logger.warning("%s", exc)
            return False
        self.store.save(result)
        return True

    def collect(self, start: date, end: date) -> CollectStats:
        if end >= self.today:
            end = self.today - timedelta(days=1)
            logger.info("未確定のレースを避けるため終了日を %s に調整しました", end)
        if start > end:
            raise ValueError(f"開始日 {start} が終了日 {end} より後です")

        stats = CollectStats()
        for kaisai in self.find_kaisai(start, end):
            stats.kaisai += 1
            race_ids = self.find_race_ids(kaisai)
            stats.races_found += len(race_ids)
            for race_id in race_ids:
                if not self.refetch and self.store.has_race(race_id):
                    stats.races_skipped += 1
                    continue
                if self.collect_race(race_id):
                    stats.races_saved += 1
                else:
                    stats.failed_race_ids.append(race_id)
            logger.info(
                "%s %s 完了（累計: 保存 %d / スキップ %d / 失敗 %d, リクエスト %d）",
                kaisai.kaisai_date,
                kaisai.place_name,
                stats.races_saved,
                stats.races_skipped,
                len(stats.failed_race_ids),
                self.client.request_count,
            )
        return stats
