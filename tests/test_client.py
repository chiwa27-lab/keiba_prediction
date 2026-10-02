from __future__ import annotations

from pathlib import Path

import pytest
import requests

from nankan_keiba.client import (
    AccessDeniedError,
    FetchError,
    PoliteHttpClient,
    RequestBudgetExceeded,
)


class FakeResponse:
    def __init__(self, status: int, text: str = "", headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"


class FakeSession:
    def __init__(self, routes: dict[str, list[FakeResponse]]) -> None:
        self.routes = routes
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        queue = self.routes[url]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


ROBOTS = "https://nar.netkeiba.com/robots.txt"
PAGE = "https://nar.netkeiba.com/race/result.html?race_id=202544090101"


def make_client(tmp_path: Path, routes: dict[str, list[FakeResponse]], **kwargs: object) -> tuple:
    session = FakeSession(routes)
    clock = FakeClock()
    client = PoliteHttpClient(
        tmp_path,
        interval=3.0,
        jitter=0.0,
        session=session,  # type: ignore[arg-type]
        sleep=clock.sleep,
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, session, clock


def test_rejects_too_short_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PoliteHttpClient(tmp_path, interval=0.5)


def test_rejects_other_hosts(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, {})
    with pytest.raises(ValueError):
        client.get("https://example.com/")
    with pytest.raises(ValueError):
        client.get("http://nar.netkeiba.com/")
    assert session.calls == []


def test_throttles_and_caches(tmp_path: Path) -> None:
    page2 = PAGE.replace("01", "02")
    client, session, clock = make_client(
        tmp_path,
        {ROBOTS: [FakeResponse(404)], PAGE: [FakeResponse(200, "<p>1</p>")], page2: [FakeResponse(200, "<p>2</p>")]},
    )
    assert client.get(PAGE) == "<p>1</p>"
    assert client.get(page2) == "<p>2</p>"
    assert client.get(PAGE) == "<p>1</p>"
    assert session.calls == [ROBOTS, PAGE, page2]
    assert clock.sleeps == [3.0, 3.0]
    assert client.cache_hits == 1


def test_respects_robots_disallow_and_crawl_delay(tmp_path: Path) -> None:
    robots = "User-agent: *\nCrawl-delay: 10\nDisallow: /race/\n"
    client, session, _ = make_client(tmp_path, {ROBOTS: [FakeResponse(200, robots)]})
    with pytest.raises(AccessDeniedError):
        client.get(PAGE)
    assert session.calls == [ROBOTS]
    assert client.interval == 10.0


def test_aborts_on_403(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, {ROBOTS: [FakeResponse(404)], PAGE: [FakeResponse(403)]})
    with pytest.raises(AccessDeniedError):
        client.get(PAGE)


def test_retries_with_retry_after(tmp_path: Path) -> None:
    client, session, clock = make_client(
        tmp_path,
        {
            ROBOTS: [FakeResponse(404)],
            PAGE: [FakeResponse(429, headers={"Retry-After": "120"}), FakeResponse(200, "ok")],
        },
    )
    assert client.get(PAGE) == "ok"
    assert session.calls == [ROBOTS, PAGE, PAGE]
    assert 120.0 in clock.sleeps


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, {ROBOTS: [FakeResponse(404)], PAGE: [FakeResponse(503)]}, max_retries=2)
    with pytest.raises(FetchError):
        client.get(PAGE)
    assert session.calls.count(PAGE) == 3


def test_retries_connection_errors(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, {ROBOTS: [FakeResponse(404)], PAGE: [FakeResponse(200, "ok")]})
    original_get = session.get
    failures = iter([True])

    def flaky_get(url: str, timeout: float) -> FakeResponse:
        if url == PAGE and next(failures, False):
            session.calls.append(url)
            raise requests.ConnectionError("boom")
        return original_get(url, timeout)

    session.get = flaky_get  # type: ignore[method-assign]
    assert client.get(PAGE) == "ok"
    assert session.calls.count(PAGE) == 2


def test_request_budget(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, {ROBOTS: [FakeResponse(404)], PAGE: [FakeResponse(200, "ok")]}, max_requests=1)
    with pytest.raises(RequestBudgetExceeded):
        client.get(PAGE)
