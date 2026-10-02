from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .client import AccessDeniedError, FetchError, PoliteHttpClient, RequestBudgetExceeded
from .collector import Collector, default_period
from .config import DEFAULT_INTERVAL_SEC, DEFAULT_JITTER_SEC, DEFAULT_USER_AGENT, MIN_INTERVAL_SEC, NANKAN_PLACES
from .storage import RaceStore

logger = logging.getLogger("nankan_keiba")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nankan-keiba",
        description="南関東競馬のレース結果を netkeiba から低負荷で収集します",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="レース結果を収集して SQLite に保存")
    collect.add_argument("--start", type=_parse_date, help="開始日 YYYY-MM-DD（既定: 終了日の 1 年前）")
    collect.add_argument("--end", type=_parse_date, help="終了日 YYYY-MM-DD（既定: 前日）")
    collect.add_argument(
        "--places",
        nargs="+",
        choices=sorted(NANKAN_PLACES),
        default=sorted(NANKAN_PLACES),
        help="競馬場コード 42=浦和 43=船橋 44=大井 45=川崎（既定: 全て）",
    )
    collect.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    collect.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    collect.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"リクエスト間隔（秒, 最小 {MIN_INTERVAL_SEC}）",
    )
    collect.add_argument("--jitter", type=float, default=DEFAULT_JITTER_SEC, help="間隔に加えるランダム秒数の上限")
    collect.add_argument(
        "--max-requests",
        type=int,
        default=1500,
        help="1 回の実行で送るリクエスト数の上限（0 で無制限）",
    )
    collect.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    collect.add_argument("--refetch", action="store_true", help="保存済み・キャッシュ済みのレース結果も取得し直す")

    export = sub.add_parser("export", help="SQLite の内容を CSV に書き出す")
    export.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    export.add_argument("--out", type=Path, default=Path("data/csv"))
    return parser


def _run_collect(args: argparse.Namespace) -> int:
    today = date.today()
    default_start, default_end = default_period(today)
    end = args.end or default_end
    start = args.start or (end - (default_end - default_start))

    client = PoliteHttpClient(
        args.cache_dir,
        interval=args.interval,
        jitter=args.jitter,
        max_requests=args.max_requests or None,
        user_agent=args.user_agent,
    )
    store = RaceStore(args.db)
    collector = Collector(client, store, place_codes=args.places, today=today, refetch=args.refetch)
    exit_code = 0
    try:
        stats = collector.collect(start, end)
        logger.info(
            "完了: 開催 %d / レース %d（新規保存 %d, 保存済み %d, 失敗 %d）",
            stats.kaisai,
            stats.races_found,
            stats.races_saved,
            stats.races_skipped,
            len(stats.failed_race_ids),
        )
        if stats.failed_race_ids:
            logger.warning("取得に失敗した race_id: %s", ", ".join(stats.failed_race_ids))
    except RequestBudgetExceeded as exc:
        logger.warning("%s。再実行すると続きから収集します", exc)
    except AccessDeniedError as exc:
        logger.error("%s", exc)
        exit_code = 2
    except FetchError as exc:
        logger.error("%s。時間をおいて再実行してください", exc)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("中断しました。再実行すると続きから収集します")
        exit_code = 130
    finally:
        logger.info(
            "リクエスト %d 件 / キャッシュ利用 %d 件 / DB: races=%d results=%d payouts=%d",
            client.request_count,
            client.cache_hits,
            store.count("races"),
            store.count("results"),
            store.count("payouts"),
        )
        store.close()
    return exit_code


def _run_export(args: argparse.Namespace) -> int:
    store = RaceStore(args.db)
    try:
        for path in store.export_csv(args.out):
            logger.info("書き出し: %s", path)
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "collect":
        return _run_collect(args)
    return _run_export(args)


if __name__ == "__main__":
    sys.exit(main())
