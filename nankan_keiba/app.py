"""南関東競馬の着順予想 Web 画面（Streamlit）. 起動: ``nankan-keiba web``."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import streamlit as st

from nankan_keiba.client import AccessDeniedError, FetchError, PoliteHttpClient, RequestBudgetExceeded
from nankan_keiba.collector import Collector, collect_pedigrees
from nankan_keiba.features import build_features, load_history, load_win_payouts
from nankan_keiba.model import RaceModel
from nankan_keiba.model import train as train_model
from nankan_keiba.predict import find_race_ids, run_prediction
from nankan_keiba.storage import RaceStore
from nankan_keiba.webapp import jst_today, last_race_date, load_actual_results, race_label, race_table, update_period

MODEL_LABELS = {"plain": "オッズなし（おすすめ）", "odds": "オッズあり（出馬表の予想オッズを使用）"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/nankan.sqlite"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/model"))
    parser.add_argument("--odds-model-dir", type=Path, default=Path("data/model_odds"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_known_args(sys.argv[1:])[0]


@st.cache_resource
def _job_lock() -> threading.Lock:
    """netkeiba へのアクセスを伴う処理は、画面を複数開いても 1 つずつ実行する."""
    return threading.Lock()


@st.cache_resource
def _client(cache_dir: str) -> PoliteHttpClient:
    return PoliteHttpClient(Path(cache_dir))


@st.cache_resource
def _load_model(model_dir: str, mtime: float) -> RaceModel:
    return RaceModel.load(Path(model_dir))


@st.cache_data(ttl=300, show_spinner="開催を確認しています…")
def _race_ids(kaisai_date: str, cache_dir: str) -> list[str]:
    with _job_lock():
        return find_race_ids(_client(cache_dir), kaisai_date)


def _model(model_dir: Path) -> RaceModel | None:
    meta = model_dir / "meta.json"
    return _load_model(str(model_dir), meta.stat().st_mtime) if meta.exists() else None


def _update_data(args: argparse.Namespace) -> str:
    today = jst_today()
    period = update_period(last_race_date(args.db), today)
    client = PoliteHttpClient(args.cache_dir, max_requests=1500)
    store = RaceStore(args.db)
    try:
        lines = []
        if period:
            stats = Collector(client, store, today=today).collect(*period)
            lines.append(f"{period[0]}〜{period[1]} のレース {stats.races_saved} 件を保存しました")
        else:
            lines.append("レース結果は前日まで取り込み済みです")
        saved, _ = collect_pedigrees(client, store, store.horses_without_pedigree())
        lines.append(f"血統 {saved} 頭を保存しました")
        return "。".join(lines)
    finally:
        store.close()


def _retrain(args: argparse.Namespace) -> None:
    features = build_features(load_history(args.db))
    payouts = load_win_payouts(args.db)
    for model_dir, use_odds in ((args.model_dir, False), (args.odds_model_dir, True)):
        model, _ = train_model(features, payouts, use_odds=use_odds)
        model.save(model_dir)


def _run_job(label: str, job: str, args: argparse.Namespace) -> None:
    lock = _job_lock()
    if not lock.acquire(blocking=False):
        st.sidebar.warning("ほかの処理を実行中です。終わってからもう一度押してください")
        return
    try:
        with st.sidebar.status(label, expanded=True) as status:
            if job == "update":
                st.write(_update_data(args))
            else:
                _retrain(args)
                st.write("モデルを保存しました")
            status.update(label="完了しました", state="complete")
    except RequestBudgetExceeded:
        st.sidebar.warning("1 回のリクエスト上限に達しました。もう一度押すと続きから取り込みます")
    except (AccessDeniedError, FetchError) as exc:
        st.sidebar.error(f"netkeiba から取得できませんでした（{exc}）。時間をおいて試してください")
    except FileNotFoundError as exc:
        st.sidebar.error(f"{exc}。先に「前日までの結果と血統を取り込む」を押してください")
    finally:
        lock.release()


def _sidebar(args: argparse.Namespace) -> None:
    st.sidebar.header("データとモデル")
    last = last_race_date(args.db)
    st.sidebar.write(f"保存済みのレース結果: {last or 'なし'} まで")
    model = _model(args.model_dir)
    if model is not None:
        metrics = model.meta.get("metrics")
        auc = metrics.get("win_auc") if isinstance(metrics, dict) else None
        st.sidebar.write(f"モデルの学習データ: {model.meta.get('trained_until', '不明')} まで")
        if isinstance(auc, float):
            st.sidebar.caption(f"検証期間の勝率AUC: {auc:.3f}")
    update_help = "DB に無い日から前日までを低負荷で取得します（数分〜）"
    if st.sidebar.button("前日までの結果と血統を取り込む", help=update_help):
        _run_job("取り込んでいます…", "update", args)
    if st.sidebar.button("モデルを学習し直す", help="保存済みの全データで 2 つのモデルを学習します（数分）"):
        _run_job("学習しています…", "train", args)


def _predict(args: argparse.Namespace, model: RaceModel, race_ids: list[str]) -> None:
    lock = _job_lock()
    if not lock.acquire(blocking=False):
        st.warning("ほかの処理を実行中です。終わってからもう一度押してください")
        return
    try:
        with st.spinner("出馬表などを取得して予想しています…（1〜2 分かかります）"):
            run = run_prediction(_client(str(args.cache_dir)), args.db, model, race_ids)
        st.session_state["result"] = (run.predictions, load_actual_results(args.db, race_ids), run.same_day_races)
    except (AccessDeniedError, FetchError) as exc:
        st.error(f"netkeiba から取得できませんでした（{exc}）。時間をおいて試してください")
    except FileNotFoundError as exc:
        st.error(f"{exc}。サイドバーの「前日までの結果と血統を取り込む」を押してください")
    finally:
        lock.release()


def _show_result() -> None:
    result = st.session_state.get("result")
    if result is None:
        return
    pred, actual, same_day_races = result
    if pred.empty:
        st.info("予想できる出走馬がいませんでした")
        return
    if same_day_races:
        st.caption(f"当日の前のレースの結果 {same_day_races} レース分を、馬場傾向として使いました")
    st.download_button(
        "予想を CSV でダウンロード",
        pred.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"prediction_{pred['race_date'].iloc[0]}.csv",
        mime="text/csv",
    )
    if not actual.empty:
        st.caption(
            "結果が出ているレースは着順も表示しています。過去の日付はその日の結果も学習に使っているため、実際より当たって見えます。"
        )
    for race_id, race in pred.groupby("race_id", sort=True):
        head = race.iloc[0]
        st.subheader(f"{race_label(str(race_id))}　{head['race_name']}（{int(head['distance'])}m）")
        st.dataframe(
            race_table(race, actual),
            hide_index=True,
            width="stretch",
            column_config={
                "勝率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "3着内率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "オッズ": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="南関東競馬 着順予想", page_icon="🏇", layout="wide")
    st.title("南関東競馬 着順予想")
    st.caption("netkeiba の公開ページから低負荷で取得したデータで予想します。データと予想は個人利用に限ります。")
    _sidebar(args)

    kaisai = st.date_input("開催日", value=jst_today(), format="YYYY/MM/DD")
    try:
        race_ids = _race_ids(kaisai.strftime("%Y%m%d"), str(args.cache_dir))
    except (AccessDeniedError, FetchError) as exc:
        st.error(f"開催を確認できませんでした（{exc}）")
        return
    if not race_ids:
        st.info("この日は南関東（浦和・船橋・大井・川崎）の開催がありません")
        return
    places = sorted({race_label(r).split()[0] for r in race_ids})
    st.write(f"開催: {'・'.join(places)}（{len(race_ids)} レース）")

    mode = st.radio(
        "予想するレース",
        ["全レース（朝の予想）", "レースを選ぶ（発走直前の予想）"],
        horizontal=True,
        help="レースを選ぶと、その日の前のレースの結果から馬場傾向を計算して使います（全レースのときは使いません）。",
    )
    selected = race_ids
    if mode.startswith("レース"):
        selected = st.multiselect("レース", race_ids, default=race_ids[:1], format_func=race_label)
    model_key = st.radio("モデル", list(MODEL_LABELS), format_func=MODEL_LABELS.__getitem__, horizontal=True)
    model = _model(args.model_dir if model_key == "plain" else args.odds_model_dir)
    if model is None:
        st.error("学習済みモデルがありません。サイドバーの「モデルを学習し直す」を押してください")
        return

    if st.button("予想する", type="primary", disabled=not selected, key="predict"):
        _predict(args, model, list(selected))
    _show_result()


main()
