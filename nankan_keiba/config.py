from __future__ import annotations

NAR_HOST = "nar.netkeiba.com"
NAR_BASE_URL = f"https://{NAR_HOST}"
ALLOWED_HOSTS = frozenset({NAR_HOST})

# netkeiba の地方競馬場コード（race_id の 5〜6 桁目）
NANKAN_PLACES: dict[str, str] = {
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
}

DEFAULT_USER_AGENT = "nankan-keiba-collector/0.1 (personal use; low-frequency polite crawler)"
DEFAULT_INTERVAL_SEC = 3.0
MIN_INTERVAL_SEC = 2.0
DEFAULT_JITTER_SEC = 1.0
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_MAX_RETRIES = 3
MAX_BACKOFF_SEC = 600.0
MAX_CONSECUTIVE_FAILURES = 5


def calendar_url(year: int, month: int) -> str:
    return f"{NAR_BASE_URL}/top/calendar.html?year={year}&month={month}"


def race_list_url(kaisai_date: str) -> str:
    return f"{NAR_BASE_URL}/top/race_list_sub.html?kaisai_date={kaisai_date}"


def race_result_url(race_id: str) -> str:
    return f"{NAR_BASE_URL}/race/result.html?race_id={race_id}"
