from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import webbrowser
from datetime import date
from pathlib import Path

from .client import AccessDeniedError, FetchError, PoliteHttpClient, RequestBudgetExceeded
from .collector import Collector, collect_pedigrees, default_period
from .config import DEFAULT_INTERVAL_SEC, DEFAULT_JITTER_SEC, DEFAULT_USER_AGENT, MIN_INTERVAL_SEC, NANKAN_PLACES
from .features import build_features, load_history, load_win_payouts
from .model import OBJECTIVES, RaceModel
from .model import train as train_model
from .predict import find_race_ids, format_predictions, run_prediction
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

    pedigree = sub.add_parser("pedigree", help="保存済みの出走馬の血統（父・母・母の父）を収集して SQLite に保存")
    pedigree.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    pedigree.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    pedigree.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"リクエスト間隔（秒, 最小 {MIN_INTERVAL_SEC}）",
    )
    pedigree.add_argument("--jitter", type=float, default=DEFAULT_JITTER_SEC, help="間隔に加えるランダム秒数の上限")
    pedigree.add_argument(
        "--max-requests",
        type=int,
        default=1500,
        help="1 回の実行で送るリクエスト数の上限（0 で無制限）",
    )
    pedigree.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    export = sub.add_parser("export", help="SQLite の内容を CSV に書き出す")
    export.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    export.add_argument("--out", type=Path, default=Path("data/csv"))

    train = sub.add_parser("train", help="保存済みのレース結果で予想モデルを学習")
    train.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    train.add_argument("--model-dir", type=Path, default=Path("data/model"))
    train.add_argument("--valid-from", type=_parse_date, help="検証期間の開始日 YYYY-MM-DD（既定: 直近 20%% の開催日）")
    train.add_argument("--use-odds", action="store_true", help="単勝オッズ・人気も特徴量に使う")
    train.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="binary",
        help="binary: 1 着・3 着内を別々に 2 値分類（既定） / lambdarank: レース内の順位を直接学習",
    )

    predict = sub.add_parser("predict", help="出馬表を取得して着順を予想")
    target = predict.add_mutually_exclusive_group(required=True)
    target.add_argument("--date", type=_parse_date, help="開催日 YYYY-MM-DD（対象競馬場の全レース）")
    target.add_argument("--race-id", nargs="+", help="race_id（12 桁）")
    predict.add_argument(
        "--places",
        nargs="+",
        choices=sorted(NANKAN_PLACES),
        default=sorted(NANKAN_PLACES),
        help="--date 指定時の競馬場コード（既定: 全て）",
    )
    predict.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    predict.add_argument("--model-dir", type=Path, default=Path("data/model"))
    predict.add_argument("--out", type=Path, help="予想結果の CSV（既定: data/predictions/<日付 or race_id>.csv）")
    predict.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    predict.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    predict.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    web = sub.add_parser("web", help="ブラウザで予想する画面を起動")
    web.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    web.add_argument("--model-dir", type=Path, default=Path("data/model"))
    web.add_argument("--odds-model-dir", type=Path, default=Path("data/model_odds"))
    web.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    web.add_argument("--port", type=int, default=8501)
    web.add_argument("--no-browser", action="store_true", help="起動時にブラウザを開かない")
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


def _run_pedigree(args: argparse.Namespace) -> int:
    client = PoliteHttpClient(
        args.cache_dir,
        interval=args.interval,
        jitter=args.jitter,
        max_requests=args.max_requests or None,
        user_agent=args.user_agent,
    )
    store = RaceStore(args.db)
    horse_ids = store.horses_without_pedigree()
    logger.info("血統が未取得の馬: %d 頭", len(horse_ids))
    exit_code = 0
    try:
        saved, failed = collect_pedigrees(client, store, horse_ids)
        logger.info("完了: 保存 %d 頭（うち読み取れなかった馬 %d 頭）", saved, failed)
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
        logger.info("リクエスト %d 件 / DB: pedigrees=%d", client.request_count, store.count("pedigrees"))
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


def _run_train(args: argparse.Namespace) -> int:
    features = build_features(load_history(args.db))
    model, metrics = train_model(
        features,
        load_win_payouts(args.db),
        valid_from=args.valid_from,
        use_odds=args.use_odds,
        objective=args.objective,
    )
    model.save(args.model_dir)
    logger.info("検証結果:\n%s", json.dumps(metrics, ensure_ascii=False, indent=2))
    logger.info("モデルを保存しました: %s", args.model_dir)
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    model = RaceModel.load(args.model_dir)
    client = PoliteHttpClient(args.cache_dir, interval=args.interval, user_agent=args.user_agent)
    try:
        if args.date:
            race_ids = find_race_ids(client, args.date.strftime("%Y%m%d"), args.places)
            if not race_ids:
                logger.error("%s の対象レースが見つかりません", args.date)
                return 1
        else:
            race_ids = args.race_id
        pred = run_prediction(client, args.db, model, race_ids).predictions
    except AccessDeniedError as exc:
        logger.error("%s", exc)
        return 2
    except FetchError as exc:
        logger.error("%s", exc)
        return 1

    out = args.out or Path("data/predictions") / f"{args.date or race_ids[0]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    print(format_predictions(pred))
    logger.info("予想結果を保存しました: %s", out)
    return 0


def _run_web(args: argparse.Namespace) -> int:
    app = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "localhost",
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        *("--db", str(args.db), "--model-dir", str(args.model_dir)),
        *("--odds-model-dir", str(args.odds_model_dir), "--cache-dir", str(args.cache_dir)),
    ]
    url = f"http://localhost:{args.port}"
    print(f"ブラウザで {url} を開いてください（終了するときは Ctrl+C）")
    if not args.no_browser:
        threading.Timer(3.0, webbrowser.open, args=(url,)).start()
    return subprocess.call(command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "collect":
        return _run_collect(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "pedigree":
        return _run_pedigree(args)
    if args.command == "web":
        return _run_web(args)
    return _run_export(args)


if __name__ == "__main__":
    sys.exit(main())
