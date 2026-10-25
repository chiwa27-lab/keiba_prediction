"""南関東競馬の着順予想 Web 画面（Streamlit）. 起動: ``nankan-keiba web``."""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import date
from pathlib import Path

import streamlit as st

from nankan_keiba import spat4, spat4_browser
from nankan_keiba.client import AccessDeniedError, FetchError, PoliteHttpClient, RequestBudgetExceeded
from nankan_keiba.collector import Collector, collect_pedigrees
from nankan_keiba.features import build_features, load_history, load_win_payouts
from nankan_keiba.model import RaceModel
from nankan_keiba.model import train as train_model
from nankan_keiba.predict import find_race_ids, run_prediction
from nankan_keiba.simulate import BET_TYPES, STAKE_YEN, leaks_training, run_simulation
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
        latest = max(date(int(r[0:4]), int(r[6:8]), int(r[8:10])) for r in race_ids)
        st.session_state["result"] = (
            run.predictions,
            load_actual_results(args.db, race_ids),
            run.same_day_races,
            leaks_training(model, latest),
        )
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
    pred, actual, same_day_races, leaked = result
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
        note = "この日の結果もモデルの学習に使っているため、実際より当たって見えます。" if leaked else ""
        st.caption(f"結果が出ているレースは着順も表示しています。{note}")
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
    predict_tab, simulate_tab, vote_tab = st.tabs(["予想", "回収率の答え合わせ", "SPAT4 投票"])
    with predict_tab:
        _prediction_page(args)
    with simulate_tab:
        _simulation_page(args)
    with vote_tab:
        _vote_page(args)


def _prediction_page(args: argparse.Namespace) -> None:
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


def _simulation_page(args: argparse.Namespace) -> None:
    st.caption(
        "DB に保存済みのレース結果と払戻を使い、予想どおりに 1 点 100 円で買った場合の回収率を計算します"
        "（netkeiba にはアクセスしません）。同じ日の前のレースの結果は使う、発走直前の予想と同じ条件です。"
    )
    last = last_race_date(args.db)
    default = last or jst_today()
    period = st.date_input("期間", value=(default, default), format="YYYY/MM/DD", key="sim_period")
    if not isinstance(period, tuple) or len(period) != 2:
        st.info("開始日と終了日を選んでください")
        return
    start, end = period
    model_key = st.radio(
        "モデル", list(MODEL_LABELS), format_func=MODEL_LABELS.__getitem__, horizontal=True, key="sim_model"
    )
    model = _model(args.model_dir if model_key == "plain" else args.odds_model_dir)
    if model is None:
        st.error("学習済みモデルがありません。サイドバーの「モデルを学習し直す」を押してください")
        return
    if st.button("答え合わせする", type="primary", key="simulate"):
        try:
            with st.spinner("保存済みの結果で予想し直しています…（1〜2 分かかります）"):
                sim = run_simulation(args.db, model, start, end)
        except FileNotFoundError as exc:
            st.error(f"{exc}。サイドバーの「前日までの結果と血統を取り込む」を押してください")
            return
        st.session_state["simulation"] = (sim, start, end, leaks_training(model, end))
    _show_simulation()


def _show_simulation() -> None:
    result = st.session_state.get("simulation")
    if result is None:
        return
    sim, start, end, leaked = result
    if sim.summary.empty:
        st.info(
            "この期間のレース結果が DB にありません。" "サイドバーの「前日までの結果と血統を取り込む」を押してください"
        )
        return
    st.subheader(f"{start:%Y/%m/%d}〜{end:%Y/%m/%d}（{sim.details['race_id'].nunique()} レース）")
    if leaked:
        st.warning("この期間はモデルの学習データに含まれているため、実際より成績が良く出ます")
    st.dataframe(
        sim.summary,
        hide_index=True,
        width="stretch",
        column_config={
            "投資額": st.column_config.NumberColumn(format="%d 円"),
            "払戻額": st.column_config.NumberColumn(format="%d 円"),
            "回収率": st.column_config.NumberColumn(format="%.1f%%"),
            "的中率": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        "◎○▲△ はモデルの予想 1〜4 位。3 連複は ◎・○ と ◎・▲ の 2 頭軸からの流し（重なる買い目は 1 点）、"
        "3 連単は 1 着 ◎・2 着 ○ か ▲・3 着 残り全頭です。"
    )
    details = sim.details.assign(レース=sim.details["race_id"].map(race_label))
    st.download_button(
        "レース別の収支を CSV でダウンロード",
        details.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"simulation_{start:%Y%m%d}_{end:%Y%m%d}.csv",
        mime="text/csv",
    )
    with st.expander("レース別の収支"):
        st.dataframe(
            details[["race_date", "レース", "券種", "購入点数", "投資額", "払戻額", "的中"]],
            hide_index=True,
            width="stretch",
        )


def _spat4_settings_form(data_dir: Path, settings: spat4.Spat4Settings) -> None:
    with st.expander("SPAT4 の設定", expanded=not settings.complete), st.form("spat4_settings"):
        st.caption("この PC の data フォルダにだけ保存します。暗証番号は保存しません。")
        member = st.text_input("加入者番号", value=settings.member_number, type="password")
        user_id = st.text_input("利用者ID", value=settings.user_id, type="password")
        limit = st.number_input(
            "1日の上限金額（円）", min_value=STAKE_YEN, step=STAKE_YEN, value=settings.daily_limit_yen
        )
        if st.form_submit_button("保存する"):
            try:
                spat4.save_settings(data_dir, spat4.Spat4Settings(member.strip(), user_id.strip(), int(limit)))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success("保存しました")
                st.rerun()


def _vote_page(args: argparse.Namespace) -> None:
    data_dir = args.db.parent
    settings = spat4.load_settings(data_dir)
    st.caption(
        "予想の買い目を SPAT4 の画面に入力し、最終確認画面で止めます。"
        "内容を確かめて、投票ボタンはご自身で押してください。1 点 100 円です。"
    )
    _spat4_settings_form(data_dir, settings)
    result = st.session_state.get("result")
    if result is None or result[0].empty:
        st.info("先に「予想」タブで予想してください")
        return
    pred = result[0]
    race_ids = sorted(pred["race_id"].unique())
    races = st.multiselect("買うレース", race_ids, default=race_ids[:1], format_func=race_label, key="vote_races")
    bet_types = st.multiselect(
        "券種",
        list(BET_TYPES),
        default=["単勝"],
        format_func=lambda b: f"{b}（{BET_TYPES[b]}）",
        key="vote_bet_types",
    )
    rows = spat4.build_tickets(pred, list(races), list(bet_types))
    today = jst_today()
    checked = spat4.check(rows, spat4.load_log(data_dir, today), settings.daily_limit_yen)
    if not checked.duplicates.empty:
        st.warning(f"今日すでに入力した買い目 {len(checked.duplicates)} 点は除きました")
    if checked.new_tickets.empty:
        st.info("入力する買い目がありません")
        return
    shown = checked.new_tickets.assign(レース=checked.new_tickets["race_id"].map(race_label))
    st.dataframe(shown[["レース", "券種", "買い目", "金額"]], hide_index=True, width="stretch")
    st.write(
        f"合計 {len(checked.new_tickets)} 点 {checked.total_yen:,} 円"
        f"（今日の入力済み {checked.used_today_yen:,} 円 / 上限 {checked.limit_yen:,} 円）"
    )
    if checked.over_limit:
        st.error("1日の上限金額を超えます。レースや券種を減らすか、設定で上限を変えてください")
    if not settings.complete:
        st.error("「SPAT4 の設定」で加入者番号と利用者IDを保存してください")
    too_many = len(checked.new_tickets) > spat4_browser.MAX_TICKETS
    if too_many:
        st.error(f"SPAT4 で一度に入力できるのは {spat4_browser.MAX_TICKETS} 点までです。レースや券種を減らしてください")
    pin = st.text_input("暗証番号（保存しません）", type="password", max_chars=4, key="spat4_pin")
    ready = settings.complete and not checked.over_limit and not too_many and pin.isdigit() and len(pin) == 4
    if st.button("SPAT4 の画面に入力する", type="primary", disabled=not ready, key="spat4_fill"):
        try:
            spat4_browser.enter_votes(settings, pin, checked.new_tickets, data_dir)
        except spat4_browser.Spat4BrowserError as exc:
            st.error(str(exc))
        else:
            spat4.record(data_dir, checked.new_tickets, today)
            st.success(
                "入力しました。暗証番号と合計金額も入力済みです。"
                "開いた Chrome の確認画面で内容を確かめ、投票ボタンを押してください"
            )


main()
