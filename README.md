# nankan-keiba

南関東競馬（大井・川崎・船橋・浦和）の予想用データを集めるためのツールです。
第一段階として、[netkeiba 地方競馬](https://nar.netkeiba.com/) から **過去のレース結果**（着順・オッズ・タイム・払戻・コーナー通過順・ラップ）をサーバーに負担をかけないよう取得し、SQLite / CSV に保存します。

## 利用規約・マナーについて

netkeiba の[利用規約](https://www.netkeiba.com/info/kiyaku.html)（2026-10 時点で確認）では主に以下が定められています。本ツールはこれらを守る前提で作っています。

| 規約 | 内容 | 本ツールでの対応 |
| --- | --- | --- |
| 第14条 | 私的利用の範囲を超える複製・販売・出版の禁止。不特定多数が見られる状態に置く行為も禁止 | **個人の予想研究専用**。取得データ・キャッシュ (`data/`) は `.gitignore` 済み。公開リポジトリやWebへアップロードしないでください |
| 第15条 | 営利目的での利用・その準備の禁止 | 予想の販売・有料配信などには使わないでください |
| 第16条(12) | サービス設備の運営に支障を与える行為の禁止 | 下記の負荷対策を実装 |

robots.txt は 2026-10 時点で存在しません（HTTP 404）が、実行のたびに確認し、今後設置された場合は `Disallow` と `Crawl-delay` に従います。

### 負荷対策（`nankan_keiba/client.py`）

- **直列実行のみ**（並列リクエストなし）。リクエスト間隔は既定 3 秒 + 0〜1 秒のランダム。2 秒未満は設定不可
- **取得済みページは gzip でローカルキャッシュ**し、同じ URL へは二度とアクセスしない。DB に保存済みのレースは結果ページ自体を取りに行かない（中断後は続きから再開）
- 429 / 5xx・通信エラーは `Retry-After` または指数バックオフ（最大 10 分）で最大 3 回まで再試行。**連続 5 回失敗で中止**
- **403 を受けたら即中止**（アクセス制限のサインとして扱う）
- 1 回の実行あたりのリクエスト上限（既定 1,500 件）
- アクセス先は `https://nar.netkeiba.com` のみに制限。ログインや有料コンテンツには一切アクセスしない
- 結果が確定していない当日以降のレースは取得しない

## セットアップ

Python 3.10 以上。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 使い方

```bash
# 前日までの過去 1 年分（南関東 4 場すべて）を収集
python -m nankan_keiba collect

# 期間・競馬場を指定（42=浦和 43=船橋 44=大井 45=川崎）
python -m nankan_keiba collect --start 2026-04-01 --end 2026-09-30 --places 44 45

# CSV に書き出し（Excel で開けるよう UTF-8 BOM 付き）
python -m nankan_keiba export --out data/csv
```

過去 1 年分は約 250 開催日・約 3,000 レースで、リクエスト数は約 3,300 件、所要時間は 3〜4 時間程度です。
既定の上限 1,500 件で一旦止まるので、**数回に分けて（時間をおいて）再実行**してください。保存済みのレースはスキップされます。

主なオプション:

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--interval` | 3.0 | リクエスト間隔（秒、2.0 以上） |
| `--max-requests` | 1500 | 1 回の実行のリクエスト上限（0 で無制限） |
| `--db` | `data/nankan.sqlite` | 保存先 SQLite |
| `--cache-dir` | `data/cache` | HTML キャッシュ |
| `--refetch` | off | 保存済みのレースも取り直す（キャッシュも無視） |

## 取得の流れ

1. 月間カレンダー `top/calendar.html?year=&month=` から南関東 4 場の開催日（`kaisai_id`）を抽出
2. 開催日ごとのレース一覧 `top/race_list_sub.html?kaisai_date=` から `race_id` を抽出
3. 各レースの結果ページ `race/result.html?race_id=` を解析して保存

`race_id` は `YYYY` + 競馬場コード(2) + `MMDD` + レース番号(2)（例: `202544090111` = 2025年 9月1日 大井 11R）。

## データ構造（SQLite / CSV）

**races**（1 レース 1 行）: `race_id, race_date, place_code, place_name, race_number, race_name, grade, post_time, surface, distance, direction, weather, track_condition, kai, nichi, race_class, num_horses, prize_money, corner_passing(JSON), lap_times(JSON)`

**results**（1 頭 1 行）: `race_id, finish_position, finish_status(取消/除外/中止 等を含む生値), bracket_number, horse_number, horse_id, horse_name, sex, age, weight_carried, jockey_id, jockey_name, finish_time, finish_time_sec, margin, popularity, win_odds, last_3f, trainer_affiliation, trainer_id, trainer_name, horse_weight, horse_weight_diff`

**payouts**（1 払戻 1 行）: `race_id, bet_type, combination, payout_yen, popularity`
（`combination` は `-` 区切り。馬単・枠単・3連単は着順どおりの並び）

`horse_id` / `jockey_id` / `trainer_id` は netkeiba の ID なので、今後の血統・馬柱情報の取得でキーとして使えます。

## 開発

```bash
pytest
ruff check . && ruff format --check .
mypy nankan_keiba
```

テストはネットワークに接続せず、`tests/fixtures/` の擬似 HTML で動きます。
