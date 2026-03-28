# 日本個別株推薦システム（平均分散最適化ベース）

Markowitz の平均分散最適化（Mean-Variance Optimization, MVO）理論に基づき、東京証券取引所上場の日本個別株ポートフォリオを最適化・推薦するシステム。

---

## 目次

1. [概要](#概要)
2. [必要環境](#必要環境)
3. [インストール](#インストール)
4. [クイックスタート](#クイックスタート)
5. [ディレクトリ構成](#ディレクトリ構成)
6. [機能詳細](#機能詳細)
7. [API リファレンス](#api-リファレンス)
8. [設定](#設定)
9. [テスト](#テスト)
10. [Docker による起動](#docker-による起動)
11. [免責事項](#免責事項)

---

## 概要

### 解決する問題

個人投資家が分散投資ポートフォリオを構築するには、多数の銘柄間の相関関係を考慮した数理最適化の知識が必要です。本システムはその計算を自動化し、ユーザーのリスク許容度・投資金額に応じた最適な銘柄配分を推薦します。

### アーキテクチャ

```
Streamlit Dashboard  ←→  FastAPI Backend  ←→  PostgreSQL
                               ↕
                    yfinance / J-Quants API
```

### 最適化手法

| 手法 | 数理定式化 | 用途 |
|---|---|---|
| 最大シャープレシオ | SOCP 変換（cvxpy / CLARABEL） | リスク中〜高 |
| 最小分散 | 二次計画（QP） | リスク低 |
| 目標リターン最小リスク | QP + 等式制約 | リターン指定 |
| リスクパリティ | 非線形最適化（SLSQP） | リスク均等化 |

---

## 必要環境

- Python 3.11 以上
- PostgreSQL 16（Docker 利用時は不要）
- インターネット接続（株価データ取得に使用）

---

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd mvopt_recsys

# 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt          # 本番のみ
pip install -r requirements-dev.txt      # 開発・テスト含む

# または pyproject.toml から
pip install -e ".[dev]"
```

---

## クイックスタート

### 1. 環境変数を設定

```bash
cp .env.example .env
```

`.env` を編集して必要な値を設定（J-Quants API キーは任意。なければ yfinance でデータ取得）:

```env
JQUANTS_MAIL=your_email@example.com
JQUANTS_PASSWORD=your_password
DATABASE_URL=postgresql://mvopt:mvopt@localhost:5432/mvopt
```

### 2. Streamlit ダッシュボードを起動（DB 不要）

```bash
streamlit run app/dashboard.py
```

ブラウザで `http://localhost:8501` を開く。

### 3. FastAPI サーバーを起動

```bash
uvicorn src.api.main:app --reload
```

API ドキュメントは `http://localhost:8000/docs` で確認できる。

---

## ディレクトリ構成

```
mvopt_recsys/
├── docs/
│   ├── requirements.md        # 要件定義書
│   └── design.md              # 設計書
├── src/
│   ├── config.py              # 設定管理（pydantic-settings）
│   ├── data/
│   │   ├── models.py          # SQLAlchemy ORM モデル
│   │   ├── database.py        # DB 接続・セッション管理
│   │   ├── repository.py      # DB アクセス層
│   │   ├── fetchers/
│   │   │   ├── base.py        # フェッチャー抽象基底クラス
│   │   │   ├── yfinance.py    # yfinance クライアント
│   │   │   └── jquants.py     # J-Quants API クライアント
│   │   └── processors/
│   │       └── price.py       # 株価データ前処理・クレンジング
│   ├── analytics/
│   │   ├── returns.py         # リターン計算（MHR / 幾何平均 / CAPM / James-Stein）
│   │   ├── covariance.py      # 共分散推定（標本 / Ledoit-Wolf / OAS / EWMA）
│   │   └── risk_metrics.py    # リスク指標（SR / VaR / CVaR / MDD / リスク寄与度）
│   ├── optimization/
│   │   ├── base.py            # OptimizationResult データクラス
│   │   ├── mean_variance.py   # MVO（最大SR / 最小分散 / 目標リターン）
│   │   ├── efficient_frontier.py  # 効率的フロンティア計算
│   │   └── risk_parity.py     # リスクパリティ最適化
│   ├── backtest/
│   │   ├── engine.py          # バックテストエンジン
│   │   └── evaluator.py       # パフォーマンス評価指標
│   ├── services/
│   │   ├── recommend.py       # 推薦サービス
│   │   ├── optimize.py        # 最適化サービス
│   │   ├── analysis.py        # 分析サービス
│   │   └── backtest.py        # バックテストサービス
│   └── api/
│       ├── main.py            # FastAPI アプリ・ルーター登録
│       ├── deps.py            # 依存性注入（価格データ取得等）
│       ├── routers/
│       │   ├── recommend.py   # POST /recommend/portfolio
│       │   ├── optimize.py    # POST /optimize/{max-sharpe, min-volatility, ...}
│       │   ├── analysis.py    # GET /analysis/correlation, /analysis/universe
│       │   └── backtest.py    # POST /backtest/run
│       └── schemas/
│           ├── request.py     # リクエスト Pydantic モデル
│           └── response.py    # レスポンス Pydantic モデル
├── app/
│   └── dashboard.py           # Streamlit ダッシュボード（5 ページ）
├── scheduler/
│   └── jobs.py                # APScheduler 定期更新ジョブ
├── tests/
│   ├── conftest.py            # 共通フィクスチャ（合成株価データ）
│   ├── unit/
│   │   ├── test_analytics.py  # 分析モジュール単体テスト（17件）
│   │   ├── test_optimization.py  # 最適化モジュール単体テスト（13件）
│   │   └── test_backtest.py   # バックテスト単体テスト（9件）
│   └── integration/
│       └── test_api.py        # FastAPI 結合テスト
├── docs/
├── requirements.txt           # 本番依存
├── requirements-dev.txt       # 開発・テスト依存
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## 機能詳細

### ポートフォリオ推薦

ユーザーが投資金額・リスク許容度を指定すると、最適なウェイト・推奨株数・金額を計算する。

| リスク許容度 | 最適化手法 | 想定年率ボラティリティ |
|---|---|---|
| `low`（低） | 最小分散 | 8〜12% |
| `medium`（中） | 最大シャープレシオ | 12〜18% |
| `high`（高） | 目標リターン 15% | 18〜25% |

**制約条件**（設定可能）:
- 個別銘柄ウェイト上限（デフォルト 30%）
- 業種ウェイト上限（デフォルト 50%）
- 空売り禁止（デフォルト有効）

### 効率的フロンティア

リスク水準を変化させながら最適ポートフォリオを計算し、リスク・リターン平面に可視化する。最大シャープレシオ点・最小分散点をハイライト表示。

### 共分散推定

| 手法 | クラス | 特徴 |
|---|---|---|
| 標本共分散 | `SampleCovariance` | シンプル。小サンプルで不安定 |
| Ledoit-Wolf 収縮 | `LedoitWolfCovariance` | 正規化効果あり（**デフォルト**） |
| OAS | `OASCovariance` | Ledoit-Wolf の改良版 |
| EWMA | `EWMACovariance` | 直近データを指数加重で重視 |

### 期待リターン推定

| 手法 | メソッド | 説明 |
|---|---|---|
| 過去平均 | `mean_historical_return()` | 日次対数リターンの算術平均（年率換算） |
| 幾何平均 | `geometric_mean_return()` | 複利ベースのリターン |
| CAPM | `capm_return()` | β × 市場リスクプレミアム + 無リスク金利 |
| James-Stein | `james_stein_return()` | 収縮推定でゼロへのバイアス低減 |

### バックテスト

指定期間・リバランス頻度でポートフォリオ運用をシミュレーション。

- **リバランス頻度**: 月次 / 四半期 / 年次
- **取引コスト**: ターンオーバーに比例して控除
- **評価指標**: 総リターン・年率リターン・シャープレシオ・最大ドローダウン・カルマー比・情報比率・アルファ
- **ベンチマーク**: TOPIX との比較（任意）

### 使い方ガイド

アプリ内に操作マニュアルページを内包。各ページの操作手順・パラメータの意味・グラフの読み方・免責事項を確認できる。データ取得は行わないため即時表示される。

---

## API リファレンス

ベース URL: `http://localhost:8000/api/v1`

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `POST` | `/recommend/portfolio` | ポートフォリオ推薦 |
| `POST` | `/optimize/max-sharpe` | 最大シャープレシオ最適化 |
| `POST` | `/optimize/min-volatility` | 最小分散最適化 |
| `POST` | `/optimize/target-return` | 目標リターン最小リスク最適化 |
| `POST` | `/optimize/efficient-frontier` | 効率的フロンティア計算 |
| `POST` | `/backtest/run` | バックテスト実行 |
| `GET`  | `/analysis/correlation` | 相関行列取得 |
| `GET`  | `/analysis/universe` | 銘柄ユニバース取得 |
| `GET`  | `/health` | ヘルスチェック |

インタラクティブな API ドキュメント（Swagger UI）: `http://localhost:8000/docs`

### リクエスト例

```bash
# ポートフォリオ推薦
curl -X POST http://localhost:8000/api/v1/recommend/portfolio \
  -H "Content-Type: application/json" \
  -d '{
    "investment_amount": 1000000,
    "risk_tolerance": "medium",
    "filters": {"max_stocks": 20},
    "constraints": {"max_weight_per_stock": 0.3}
  }'

# 効率的フロンティア
curl -X POST http://localhost:8000/api/v1/optimize/efficient-frontier \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T"],
    "n_points": 50,
    "lookback_days": 252
  }'
```

---

## 設定

`.env` ファイルで以下の環境変数を設定する。

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `JQUANTS_MAIL` | `` | J-Quants API メールアドレス（任意） |
| `JQUANTS_PASSWORD` | `` | J-Quants API パスワード（任意） |
| `DATABASE_URL` | `postgresql://mvopt:mvopt@localhost:5432/mvopt` | PostgreSQL 接続文字列 |
| `DEFAULT_LOOKBACK_DAYS` | `252` | リターン計算に使う過去営業日数 |
| `DEFAULT_COVARIANCE_METHOD` | `ledoit_wolf` | 共分散推定手法 |
| `DEFAULT_RISK_FREE_RATE` | `0.001` | 無リスク金利（年率） |
| `DATA_UPDATE_CRON` | `30 17 * * 1-5` | 株価更新スケジュール（cron 形式） |
| `LOG_LEVEL` | `INFO` | ログレベル |

> **Note**: J-Quants API キーが未設定の場合は yfinance を自動的にフォールバックとして使用します。Streamlit ダッシュボードは DB 接続なしで単独起動できます。

---

## テスト

```bash
# 単体テストのみ（DB・ネットワーク不要）
pytest tests/unit/ -v

# カバレッジレポート付き
pytest tests/unit/ --cov=src --cov-report=term-missing

# 全テスト（ネットワーク接続が必要な場合はスキップされる）
pytest tests/ -v
```

### テスト構成

| ファイル | テスト数 | 内容 |
|---|---|---|
| `test_analytics.py` | 17 | リターン計算・共分散推定・リスク指標 |
| `test_optimization.py` | 13 | MVO・リスクパリティ・効率的フロンティア |
| `test_backtest.py` | 9 | バックテストエンジン・評価指標 |
| `test_api.py` | 5 | FastAPI エンドポイント結合テスト |

---

## Docker による起動

```bash
# 全サービス起動（API + Dashboard + PostgreSQL）
docker compose up

# バックグラウンド起動
docker compose up -d

# ログ確認
docker compose logs -f api

# 停止
docker compose down
```

| サービス | URL |
|---|---|
| Streamlit ダッシュボード | http://localhost:8501 |
| FastAPI（Swagger UI） | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

---

## 免責事項

本システムは数理モデルに基づく参考情報を提供するものであり、投資助言ではありません。
過去のデータに基づく最適化結果が将来の投資成果を保証するものではありません。
投資判断は自己責任で行ってください。
