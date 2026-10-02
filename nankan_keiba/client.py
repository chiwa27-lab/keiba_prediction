from __future__ import annotations

import gzip
import hashlib
import logging
import random
import time
import urllib.robotparser
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .config import (
    ALLOWED_HOSTS,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_JITTER_SEC,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_USER_AGENT,
    MAX_BACKOFF_SEC,
    MAX_CONSECUTIVE_FAILURES,
    MIN_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class FetchError(Exception):
    pass


class NotFoundError(FetchError):
    pass


class AccessDeniedError(FetchError):
    """403 / robots.txt による拒否。収集全体を中止すべきエラー."""


class RequestBudgetExceeded(FetchError):
    pass


class PoliteHttpClient:
    """サーバ負荷に配慮した HTTP クライアント.

    - 許可ホスト以外へのアクセスを拒否
    - robots.txt を確認し、Disallow / Crawl-delay に従う
    - 直列実行 + 最小リクエスト間隔（ジッター付き）
    - 429/5xx は Retry-After・指数バックオフで再試行し、連続失敗で中止
    - 403 は即時中止
    - 取得済みページはローカルキャッシュし、再リクエストしない
    - 1 回の実行あたりのリクエスト上限
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        interval: float = DEFAULT_INTERVAL_SEC,
        jitter: float = DEFAULT_JITTER_SEC,
        max_requests: int | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        max_retries: int = DEFAULT_MAX_RETRIES,
        allowed_hosts: frozenset[str] = ALLOWED_HOSTS,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval < MIN_INTERVAL_SEC:
            raise ValueError(f"interval は {MIN_INTERVAL_SEC} 秒以上にしてください（指定値: {interval}）")
        self.cache_dir = cache_dir
        self.interval = interval
        self.jitter = max(0.0, jitter)
        self.max_requests = max_requests
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.allowed_hosts = allowed_hosts
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ja,en;q=0.5",
            }
        )
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._consecutive_failures = 0
        self.request_count = 0
        self.cache_hits = 0

    def get(self, url: str, *, use_cache: bool = True) -> str:
        host = self._check_host(url)
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            self.cache_hits += 1
            return gzip.decompress(cache_path.read_bytes()).decode("utf-8")

        self._check_robots(host, url)
        text = self._fetch(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(gzip.compress(text.encode("utf-8")))
        return text

    def invalidate(self, url: str) -> None:
        self._cache_path(url).unlink(missing_ok=True)

    def _check_host(self, url: str) -> str:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.scheme != "https" or host not in self.allowed_hosts:
            raise ValueError(f"許可されていない URL です: {url}")
        return host

    def _cache_path(self, url: str) -> Path:
        parts = urlsplit(url)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / (parts.hostname or "unknown") / digest[:2] / f"{digest}.html.gz"

    def _check_robots(self, host: str, url: str) -> None:
        parser = self._robots.get(host)
        if parser is None:
            parser = self._load_robots(host)
            self._robots[host] = parser
            delay = parser.crawl_delay(self.user_agent)
            if delay is not None and float(delay) > self.interval:
                logger.info("robots.txt の Crawl-delay=%s に合わせて間隔を延長します", delay)
                self.interval = float(delay)
        if not parser.can_fetch(self.user_agent, url):
            raise AccessDeniedError(f"robots.txt により取得が禁止されています: {url}")

    def _load_robots(self, host: str) -> urllib.robotparser.RobotFileParser:
        robots_url = f"https://{host}/robots.txt"
        parser = urllib.robotparser.RobotFileParser(robots_url)
        response = self._request(robots_url)
        if response.status_code in (401, 403):
            parser.parse(["User-agent: *", "Disallow: /"])
        elif 400 <= response.status_code < 500:
            parser.parse([])
        elif response.status_code >= 500:
            raise FetchError(f"robots.txt を取得できません（HTTP {response.status_code}）。安全のため中止します")
        else:
            parser.parse(response.text.splitlines())
        logger.info("robots.txt 確認: %s (HTTP %d)", robots_url, response.status_code)
        return parser

    def _fetch(self, url: str) -> str:
        attempt = 0
        while True:
            try:
                response = self._request(url)
            except requests.RequestException as exc:
                error: str = f"{type(exc).__name__}: {exc}"
                retry_after = None
            else:
                if response.status_code == 200:
                    self._consecutive_failures = 0
                    return _decode(response)
                if response.status_code == 404:
                    self._consecutive_failures = 0
                    raise NotFoundError(f"HTTP 404: {url}")
                if response.status_code in (401, 403):
                    raise AccessDeniedError(
                        f"HTTP {response.status_code}: アクセスが拒否されました。収集を中止します: {url}"
                    )
                if response.status_code not in RETRYABLE_STATUS:
                    raise FetchError(f"HTTP {response.status_code}: {url}")
                error = f"HTTP {response.status_code}"
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))

            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise FetchError(f"連続 {self._consecutive_failures} 回失敗したため中止します（最後のエラー: {error}）")
            if attempt >= self.max_retries:
                raise FetchError(f"{self.max_retries} 回再試行しましたが失敗しました: {url} ({error})")
            backoff = min(MAX_BACKOFF_SEC, max(retry_after or 0.0, self.interval * (2 ** (attempt + 1))))
            logger.warning("%s の取得に失敗 (%s)。%.0f 秒後に再試行します", url, error, backoff)
            self._sleep(backoff)
            attempt += 1

    def _request(self, url: str) -> requests.Response:
        if self.max_requests is not None and self.request_count >= self.max_requests:
            raise RequestBudgetExceeded(f"リクエスト上限 ({self.max_requests}) に達しました")
        self._wait_turn()
        self.request_count += 1
        logger.debug("GET %s", url)
        try:
            return self.session.get(url, timeout=self.timeout)
        finally:
            self._last_request_at = self._clock()

    def _wait_turn(self) -> None:
        if self._last_request_at is None:
            return
        wait = self.interval + random.uniform(0.0, self.jitter) - (self._clock() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)


def _decode(response: requests.Response) -> str:
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding
    return response.text


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    if value.strip().isdigit():
        return float(value.strip())
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None
