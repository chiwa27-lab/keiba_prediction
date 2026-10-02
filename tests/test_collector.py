from __future__ import annotations

from datetime import date
from pathlib import Path

from nankan_keiba.collector import Collector, default_period, iter_months
from nankan_keiba.config import calendar_url, race_list_url, race_result_url
from nankan_keiba.storage import RaceStore

FIXTURES = Path(__file__).parent / "fixtures"


class StubClient:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.requested: list[tuple[str, bool]] = []
        self.invalidated: list[str] = []
        self.request_count = 0

    def get(self, url: str, *, use_cache: bool = True) -> str:
        self.requested.append((url, use_cache))
        self.request_count += 1
        return self.pages.get(url, "<html></html>")

    def invalidate(self, url: str) -> None:
        self.invalidated.append(url)


def test_default_period_is_one_year_until_yesterday() -> None:
    assert default_period(date(2026, 10, 2)) == (date(2025, 10, 2), date(2026, 10, 1))


def test_iter_months_crosses_year() -> None:
    assert list(iter_months(date(2025, 11, 15), date(2026, 2, 1))) == [(2025, 11), (2025, 12), (2026, 1), (2026, 2)]


def test_collect_saves_races_and_skips_existing(tmp_path: Path) -> None:
    result_html = (FIXTURES / "result.html").read_text(encoding="utf-8")
    pages = {
        calendar_url(2025, 9): (FIXTURES / "calendar.html").read_text(encoding="utf-8"),
        race_list_url("20250901"): (FIXTURES / "race_list.html").read_text(encoding="utf-8"),
        race_result_url("202544090101"): result_html,
    }
    client = StubClient(pages)
    store = RaceStore(tmp_path / "db.sqlite")
    collector = Collector(client, store, place_codes=["44"], today=date(2025, 10, 1))  # type: ignore[arg-type]

    stats = collector.collect(date(2025, 9, 1), date(2025, 9, 1))

    assert (stats.kaisai, stats.races_found, stats.races_saved) == (1, 2, 1)
    assert stats.failed_race_ids == ["202544090102"]
    assert client.invalidated == [race_result_url("202544090102")]
    assert (calendar_url(2025, 9), True) in client.requested
    assert (store.count("races"), store.count("results"), store.count("payouts")) == (1, 3, 4)

    client.requested.clear()
    stats = collector.collect(date(2025, 9, 1), date(2025, 9, 1))
    assert stats.races_skipped == 1
    assert race_result_url("202544090101") not in [url for url, _ in client.requested]

    csv_paths = store.export_csv(tmp_path / "csv")
    assert [p.name for p in csv_paths] == ["races.csv", "results.csv", "payouts.csv"]
    store.close()


def test_collect_does_not_cache_current_month_calendar_and_clamps_end(tmp_path: Path) -> None:
    client = StubClient({})
    store = RaceStore(tmp_path / "db.sqlite")
    collector = Collector(client, store, today=date(2025, 9, 15))  # type: ignore[arg-type]
    collector.collect(date(2025, 9, 1), date(2025, 9, 30))
    assert client.requested == [(calendar_url(2025, 9), False)]
    store.close()
