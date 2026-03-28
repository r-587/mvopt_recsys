from __future__ import annotations

"""
Streamlit ダッシュボード
- ポートフォリオ推薦
- 効率的フロンティア
- バックテスト
- 相関行列
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.analytics.covariance import get_covariance_estimator
from src.analytics.returns import ReturnsCalculator
from src.data.fetchers.yfinance import YFinanceFetcher
from src.data.processors.price import PriceProcessor
from src.optimization.efficient_frontier import EfficientFrontier
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.services.recommend import RecommendService
from src.backtest.engine import BacktestEngine
from src.backtest.evaluator import BacktestEvaluator
from src.services.backtest import BacktestService

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="日本株ポートフォリオ最適化",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# キャッシュ：株価データ取得
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="銘柄一覧・銘柄名を取得中（初回のみ時間がかかります）...")
def load_tse_universe() -> pd.DataFrame:
    """J-Quants（設定済みなら）または YFinance デフォルトから銘柄一覧を取得。
    columns: ticker, name, market, sector_33
    """
    from src.data.fetchers.jquants import JQuantsFetcher
    from src.config import settings

    if settings.jquants_mail or settings.jquants_refresh_token:
        try:
            stocks = JQuantsFetcher().fetch_stock_list()
            if stocks:
                df = pd.DataFrame(stocks)[["ticker", "name", "market", "sector_33"]]
                return df.dropna(subset=["ticker"]).reset_index(drop=True)
        except Exception as e:
            st.sidebar.warning(f"J-Quants 接続失敗（デフォルト銘柄を使用）: {e}")

    stocks = YFinanceFetcher().fetch_stock_list()
    df = pd.DataFrame(stocks)[["ticker", "name", "market", "sector_33"]]
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)
    # YFinance の fetch_stock_list は name=ticker のため、yfinance .info で名前を補完
    names = {}
    for ticker in df["ticker"]:
        try:
            info = yf.Ticker(ticker).info
            names[ticker] = info.get("longName") or info.get("shortName") or ""
        except Exception:
            names[ticker] = ""
    df["name"] = df["ticker"].map(names)
    return df


def ticker_label(ticker: str, names: dict[str, str]) -> str:
    """'7203.T' → '7203 トヨタ自動車' のような表示ラベルを返す"""
    code = ticker.replace(".T", "")
    name = names.get(ticker, "")
    return f"{code} {name}".strip() if name else code


@st.cache_data(ttl=3600, show_spinner="株価データを取得中...")
def load_prices(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=lookback_days * 2)
    fetcher = YFinanceFetcher()
    records = fetcher.fetch_prices(list(tickers), start, end)
    if not records:
        return pd.DataFrame()
    data: dict[str, dict] = {}
    for r in records:
        t = r["ticker"]
        d = r["date"]
        if t not in data:
            data[t] = {}
        data[t][d] = r["adj_close"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.tail(lookback_days)


# ─────────────────────────────────────────────
# サイドバー：共通設定
# ─────────────────────────────────────────────
st.sidebar.title("設定")
page = st.sidebar.radio(
    "ページ選択",
    ["ポートフォリオ推薦", "効率的フロンティア", "バックテスト", "相関行列", "使い方"],
)

if page != "使い方":
    universe_df = load_tse_universe()
    _markets_available = sorted(universe_df["market"].dropna().unique().tolist())
    _sectors_available = sorted(universe_df["sector_33"].dropna().unique().tolist())

    st.sidebar.subheader("銘柄ユニバース")
    selected_markets = st.sidebar.multiselect(
        "市場区分",
        options=_markets_available,
        default=[m for m in ["Prime"] if m in _markets_available] or _markets_available[:1],
    )
    selected_sectors = st.sidebar.multiselect(
        "業種（空=全業種）",
        options=_sectors_available,
        default=[],
    )
    max_stocks = st.sidebar.slider("最大銘柄数", 20, 300, 100, step=10)

    _filtered = universe_df.copy()
    if selected_markets:
        _filtered = _filtered[_filtered["market"].isin(selected_markets)]
    if selected_sectors:
        _filtered = _filtered[_filtered["sector_33"].isin(selected_sectors)]

    tickers = tuple(_filtered["ticker"].dropna().unique()[:max_stocks])
    ticker_names: dict[str, str] = dict(zip(universe_df["ticker"], universe_df["name"].fillna("")))
    st.sidebar.caption(f"対象: {len(tickers):,} 銘柄")

    lookback_days = st.sidebar.slider("過去データ期間（日）", 60, 756, 252, step=21)
    risk_free_rate = st.sidebar.number_input("無リスク金利", value=0.001, step=0.001, format="%.3f")
    cov_method = st.sidebar.selectbox("共分散推定手法", ["ledoit_wolf", "sample", "oas", "ewma"])

    # ─────────────────────────────────────────────
    # データ取得
    # ─────────────────────────────────────────────
    price_df = load_prices(tickers, lookback_days)
    if price_df.empty:
        st.error("価格データを取得できませんでした。銘柄コードを確認してください。")
        st.stop()

    processor = PriceProcessor()
    calc = ReturnsCalculator()
    clean_df = processor.clean(price_df)
    returns_df = calc.daily_returns(clean_df)
    mu = calc.mean_historical_return(clean_df)
    cov_est = get_covariance_estimator(cov_method)
    cov = cov_est.fit(returns_df)


# ─────────────────────────────────────────────
# ページ 1: ポートフォリオ推薦
# ─────────────────────────────────────────────
if page == "ポートフォリオ推薦":
    st.title("ポートフォリオ推薦")
    st.caption("平均分散最適化による銘柄ウェイト推薦")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_tolerance = st.selectbox("リスク許容度", ["low（低）", "medium（中）", "high（高）"])
        rt = risk_tolerance.split("（")[0]
    with col2:
        investment_amount = st.number_input("投資金額（円）", value=1_000_000, step=100_000, min_value=100_000)
    with col3:
        max_w = st.slider("個別銘柄上限ウェイト", 0.05, 1.0, 0.3, step=0.05)

    if st.button("推薦を実行", type="primary"):
        with st.spinner("最適化計算中..."):
            opt = MeanVarianceOptimizer(weight_bounds=(0.0, max_w))
            if rt == "low":
                res = opt.min_volatility(mu, cov)
                method = "最小分散"
            elif rt == "high":
                res = opt.target_return(mu, cov, 0.15, risk_free_rate)
                method = "目標リターン(15%)"
            else:
                res = opt.max_sharpe(mu, cov, risk_free_rate)
                method = "最大シャープレシオ"

        # --- 結果表示 ---
        st.subheader(f"最適化結果 ({method})")
        m1, m2, m3 = st.columns(3)
        m1.metric("期待年率リターン", f"{res.expected_return:.2%}")
        m2.metric("年率ボラティリティ", f"{res.volatility:.2%}")
        m3.metric("シャープレシオ", f"{res.sharpe_ratio:.3f}")

        active = res.weights[res.weights > 0.001].sort_values(ascending=False)

        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            fig = px.pie(
                values=active.values,
                names=[ticker_label(t, ticker_names) for t in active.index],
                title="ポートフォリオ構成",
                hole=0.3,
            )
            st.plotly_chart(fig, width='stretch')

        with col_table:
            rows = []
            for ticker, w in active.items():
                latest_price = clean_df[ticker].iloc[-1] if ticker in clean_df.columns else None
                target_amount = investment_amount * w
                shares = int(target_amount / latest_price) if latest_price else 0
                amount = int(shares * latest_price) if latest_price and shares else 0
                rows.append({
                    "銘柄コード": ticker.replace(".T", ""),
                    "銘柄名": ticker_names.get(ticker, ""),
                    "ウェイト": f"{w:.2%}",
                    "目標金額": f"¥{int(target_amount):,}",
                    "推奨株数": f"{shares:,}株",
                    "投資金額": f"¥{amount:,}",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        if res.status != "optimal":
            st.warning(f"最適化ステータス: {res.status}")


# ─────────────────────────────────────────────
# ページ 2: 効率的フロンティア
# ─────────────────────────────────────────────
elif page == "効率的フロンティア":
    st.title("効率的フロンティア")

    n_points = st.slider("フロンティア点数", 20, 100, 50, step=10)
    max_w_ef = st.slider("個別銘柄上限ウェイト", 0.05, 1.0, 0.3, step=0.05)

    if st.button("フロンティアを計算", type="primary"):
        with st.spinner("効率的フロンティアを計算中..."):
            inner_opt = MeanVarianceOptimizer(weight_bounds=(0.0, max_w_ef))
            ef = EfficientFrontier()
            ef_result = ef.compute(mu, cov, n_points=n_points, risk_free_rate=risk_free_rate, optimizer=inner_opt)

        points = ef_result.points
        vols = [p["volatility"] for p in points]
        rets = [p["expected_return"] for p in points]
        srs = [p["sharpe_ratio"] for p in points]

        fig = go.Figure()

        # フロンティア曲線
        fig.add_trace(go.Scatter(
            x=vols, y=rets,
            mode="lines+markers",
            marker=dict(color=srs, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="シャープレシオ"), size=8),
            line=dict(color="lightgray", width=1),
            name="効率的フロンティア",
            hovertemplate="リスク: %{x:.2%}<br>リターン: %{y:.2%}<br>SR: %{marker.color:.3f}",
        ))

        # 最大シャープレシオ点
        ms = ef_result.max_sharpe_point
        fig.add_trace(go.Scatter(
            x=[ms["volatility"]], y=[ms["expected_return"]],
            mode="markers", marker=dict(color="red", size=14, symbol="star"),
            name=f"最大SR ({ms['sharpe_ratio']:.3f})",
        ))

        # 最小分散点
        mv = ef_result.min_vol_point
        fig.add_trace(go.Scatter(
            x=[mv["volatility"]], y=[mv["expected_return"]],
            mode="markers", marker=dict(color="blue", size=12, symbol="diamond"),
            name="最小分散",
        ))

        # 個別銘柄散布
        for ticker in mu.index:
            if ticker in cov.columns:
                w = np.zeros(len(mu))
                idx = mu.index.tolist().index(ticker)
                w[idx] = 1.0
                from src.analytics.risk_metrics import RiskMetrics
                v = RiskMetrics.annualized_volatility(w, cov)
                r = float(mu[ticker])
                fig.add_trace(go.Scatter(
                    x=[v], y=[r],
                    mode="markers+text",
                    marker=dict(color="gray", size=6, opacity=0.6),
                    text=[ticker_label(ticker, ticker_names)],
                    textposition="top center",
                    textfont=dict(size=9),
                    showlegend=False,
                ))

        fig.update_layout(
            title="効率的フロンティア",
            xaxis_title="年率ボラティリティ",
            yaxis_title="年率期待リターン",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            height=600,
        )
        st.plotly_chart(fig, width='stretch')

        # テーブル
        with st.expander("フロンティアデータ"):
            df_ef = pd.DataFrame([
                {"リターン": f"{p['expected_return']:.2%}", "ボラティリティ": f"{p['volatility']:.2%}", "SR": f"{p['sharpe_ratio']:.3f}"}
                for p in points
            ])
            st.dataframe(df_ef, hide_index=True, width='stretch')


# ─────────────────────────────────────────────
# ページ 3: バックテスト
# ─────────────────────────────────────────────
elif page == "バックテスト":
    st.title("バックテスト")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日", value=date.today() - timedelta(days=365 * 3))
        rebalance_freq = st.selectbox("リバランス頻度", ["monthly", "quarterly", "annually"])
    with col2:
        end_date = st.date_input("終了日", value=date.today() - timedelta(days=1))
        tx_cost = st.number_input("取引コスト", value=0.001, step=0.0005, format="%.4f")

    if st.button("バックテスト実行", type="primary"):
        if start_date >= end_date:
            st.error("開始日 < 終了日 にしてください")
        else:
            # 全期間の価格データを取得
            total_days = (end_date - start_date).days + lookback_days
            all_prices = load_prices(tickers, total_days)

            if all_prices.empty:
                st.error("価格データを取得できませんでした")
            else:
                with st.spinner("バックテスト実行中..."):
                    all_prices.index = pd.to_datetime(all_prices.index).date

                    # ベンチマーク（TOPIX連動ETF: 1306.T）
                    try:
                        bm_raw = yf.download("1306.T", start=str(start_date), end=str(end_date), progress=False)
                        bm_col = "Adj Close" if "Adj Close" in bm_raw.columns else "Close"
                        bm = bm_raw[bm_col].squeeze()
                        bm.index = bm.index.date
                    except Exception:
                        bm = None

                    engine = BacktestEngine(covariance_method=cov_method)
                    inner_opt = MeanVarianceOptimizer(weight_bounds=(0.0, 0.3))
                    try:
                        result = engine.run(
                            price_data=all_prices,
                            start_date=start_date,
                            end_date=end_date,
                            rebalance_freq=rebalance_freq,
                            optimizer=inner_opt,
                            lookback_days=lookback_days,
                            transaction_cost=tx_cost,
                            risk_free_rate=risk_free_rate,
                        )
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                    evaluator = BacktestEvaluator()
                    bm_values = None
                    if bm is not None:
                        bm_aligned = bm.reindex(result.portfolio_values.index).ffill()
                        if not bm_aligned.empty:
                            bm_values = bm_aligned / bm_aligned.iloc[0]

                    metrics = evaluator.evaluate(result, bm_values, risk_free_rate)

                # --- 結果表示 ---
                st.subheader("パフォーマンス推移")
                pv = result.portfolio_values

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(pv.index), y=list(pv.values),
                    mode="lines", name="MVO ポートフォリオ", line=dict(color="steelblue", width=2),
                ))
                if bm_values is not None:
                    fig.add_trace(go.Scatter(
                        x=list(bm_values.index), y=list(bm_values.values),
                        mode="lines", name="TOPIX ETF (1306)", line=dict(color="orange", width=1.5, dash="dash"),
                    ))
                for rd in result.rebalance_dates[:50]:
                    fig.add_vline(x=str(rd), line=dict(color="gray", width=0.5, dash="dot"))
                fig.update_layout(
                    yaxis_title="累積リターン（初期=1.0）",
                    height=400,
                    legend=dict(x=0, y=1),
                )
                st.plotly_chart(fig, width='stretch')

                # 指標
                st.subheader("評価指標")
                summary = evaluator.summary_table(result, bm_values, risk_free_rate)
                st.dataframe(summary, hide_index=True, width='stretch')


# ─────────────────────────────────────────────
# ページ 4: 相関行列
# ─────────────────────────────────────────────
elif page == "相関行列":
    st.title("銘柄間相関行列")

    corr = returns_df.corr()
    corr_labeled = corr.copy()
    label_map = {t: ticker_label(t, ticker_names) for t in corr.index}
    corr_labeled.index = [label_map[t] for t in corr.index]
    corr_labeled.columns = [label_map[t] for t in corr.columns]

    fig = px.imshow(
        corr_labeled,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="銘柄間相関係数ヒートマップ",
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width='stretch')

    st.subheader("期待リターン・リスク一覧")
    from src.analytics.risk_metrics import RiskMetrics
    rows = []
    for ticker in mu.index:
        if ticker not in cov.columns:
            continue
        w = np.zeros(len(mu))
        w[mu.index.tolist().index(ticker)] = 1.0
        vol = RiskMetrics.annualized_volatility(w, cov)
        rows.append({
            "銘柄コード": ticker.replace(".T", ""),
            "銘柄名": ticker_names.get(ticker, ""),
            "期待年率リターン": f"{mu[ticker]:.2%}",
            "年率ボラティリティ": f"{vol:.2%}",
            "シャープレシオ": f"{(mu[ticker] - risk_free_rate) / vol:.3f}" if vol > 0 else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


# ─────────────────────────────────────────────
# ページ 5: 使い方
# ─────────────────────────────────────────────
elif page == "使い方":
    st.title("使い方")
    st.caption("日本株ポートフォリオ最適化アプリの操作ガイド")

    st.header("このアプリについて")
    st.markdown("""
このアプリは **平均分散最適化（Mean-Variance Optimization, MVO）** 理論に基づき，
東京証券取引所（東証）に上場する日本個別株のポートフォリオを最適化し，
投資家のリスク許容度に応じた銘柄推薦を行います。

ハリー・マーコウィッツが提唱した MVO 理論では，「同じリスクなら期待リターンが高い」「同じリターンならリスクが低い」
ポートフォリオを **効率的** とみなし，そのような銘柄の組み合わせを数理的に求めます。
""")

    st.divider()

    st.header("共通設定（サイドバー）")
    st.markdown("""
左サイドバーの設定はすべてのページに共通して適用されます。

| 設定項目 | 説明 |
|---|---|
| **市場区分** | Prime / Standard / Growth から分析対象を絞り込みます。初期値は Prime のみです。 |
| **業種** | 空欄の場合は全業種を対象とします。特定業種のみを分析したい場合に使います。 |
| **最大銘柄数** | ユニバース（分析対象銘柄の母集団）の上限数です。多いほど計算に時間がかかります。 |
| **過去データ期間** | リターン・共分散の推定に使う過去データの長さ（営業日数）です。252 日 ≈ 1 年分に相当します。 |
| **無リスク金利** | シャープレシオの計算に使う基準金利です。日本の短期金利を参考に設定してください。 |
| **共分散推定手法** | 銘柄間の共分散行列の推定方法です。サンプル数が少ない場合は `ledoit_wolf` や `oas` が安定します。 |

> **共分散推定手法の目安**
> - `ledoit_wolf` … 収縮推定。銘柄数が多い場合に推奨（デフォルト）
> - `sample` … 標本共分散。シンプルだがデータ数が少ないと不安定
> - `oas` … Oracle Approximating Shrinkage。ledoit_wolf の改良版
> - `ewma` … 指数加重移動平均。直近の値動きを重視したい場合に有効
""")

    st.divider()

    st.header("ポートフォリオ推薦")
    st.markdown("""
MVO で求めた最適ポートフォリオの銘柄構成と投資金額を表示します。

**操作手順**
1. サイドバーで銘柄ユニバース・分析期間を設定する
2. **リスク許容度** を選択する（下表を参照）
3. **投資金額** を入力する（円単位）
4. **個別銘柄上限ウェイト** で 1 銘柄への集中投資を制限する
5. 「推薦を実行」ボタンを押す

**リスク許容度の選び方**

| 選択肢 | 最適化手法 | 特徴 |
|---|---|---|
| low（低） | 最小分散最適化 | 値動きの小さい安定ポートフォリオ |
| medium（中） | 最大シャープレシオ最適化 | リスク 1 単位あたりのリターンが最大 |
| high（高） | 目標リターン指定（年率 15%）で最小リスク | 高リターンを狙うが変動も大きくなる傾向 |

**結果の見方**

| 指標 | 説明 |
|---|---|
| 期待年率リターン | 過去データから推定した年率リターン（保証値ではありません） |
| 年率ボラティリティ | ポートフォリオの年率標準偏差。リスクの大きさを表します |
| シャープレシオ | （リターン − 無リスク金利）÷ ボラティリティ。値が大きいほど効率的 |
| 推奨株数・投資金額 | 入力した投資金額を各銘柄のウェイトで配分した参考値 |
""")

    st.divider()

    st.header("効率的フロンティア")
    st.markdown("""
リスク水準を変えながら最適ポートフォリオを計算し，リスク・リターン平面上に描画します。

**操作手順**
1. **フロンティア点数** を設定する（多いほど滑らかだが計算時間が増加）
2. **個別銘柄上限ウェイト** を設定する
3. 「フロンティアを計算」ボタンを押す

**グラフの見方**

| マーカー | 意味 |
|---|---|
| 曲線（色付き） | 効率的フロンティア。色はシャープレシオの高低を表します（黄色 → 高） |
| 赤★ | シャープレシオ最大点（効率最良のポートフォリオ） |
| 青◆ | 最小分散点（リスク最小のポートフォリオ） |
| 灰色の点 | 個別銘柄（1 銘柄だけ保有した場合のリスク・リターン） |

フロンティア上の点は，灰色の個別銘柄より左上に位置します。
これが分散投資の効果（リスクを下げながら同等以上のリターンを狙える）を示しています。
""")

    st.divider()

    st.header("バックテスト")
    st.markdown("""
過去の特定期間で MVO ポートフォリオを運用した場合のシミュレーションを行います。

**操作手順**
1. **開始日・終了日** を設定する（開始日 < 終了日）
2. **リバランス頻度** を選択する
3. **取引コスト** を設定する（片道コスト。0.001 = 0.1%）
4. 「バックテスト実行」ボタンを押す

**リバランス頻度の違い**

| 頻度 | 説明 |
|---|---|
| monthly | 毎月末にポートフォリオを再最適化 |
| quarterly | 四半期ごとに再最適化 |
| annually | 年 1 回再最適化。取引コストを抑えられる |

**評価指標の見方**

| 指標 | 説明 |
|---|---|
| 累積リターン | 期間全体の通算収益率 |
| 年率リターン | 複利換算した年平均リターン |
| 最大ドローダウン | ピークからの最大下落幅。損失の怖さの目安 |
| シャープレシオ | 超過リターン ÷ ボラティリティ |
| カルマー比 | 年率リターン ÷ 最大ドローダウン |

グラフの縦の点線はリバランス実施日を示しています。ベンチマーク（TOPIX ETF: 1306）と比較することで，MVO の有効性を確認できます。

> **注意**: 過去の成績は将来の運用成果を保証するものではありません。
""")

    st.divider()

    st.header("相関行列")
    st.markdown("""
銘柄間の収益率の相関係数をヒートマップで表示します。

**ヒートマップの読み方**

| 色 | 意味 |
|---|---|
| 濃い赤 | 相関係数 ≈ +1（強い正の相関。値動きがほぼ同方向） |
| 白 | 相関係数 ≈ 0（相関なし） |
| 濃い青 | 相関係数 ≈ −1（強い負の相関。値動きが逆方向） |

相関の低い（または負の相関を持つ）銘柄を組み合わせることで，ポートフォリオ全体のリスクを抑えられます。
MVO はこの相関関係を内部で考慮しながら最適ウェイトを計算しています。

**期待リターン・リスク一覧** には各銘柄を単独で保有した場合の年率リターン・ボラティリティ・シャープレシオが表示されます。
""")

    st.divider()

    st.warning("""
**免責事項**

本アプリは数理モデルに基づく参考情報を提供するものであり，投資助言を目的とするものではありません。
過去のデータに基づく最適化結果が将来の投資成果を保証するものではありません。
投資判断はご自身の責任において行ってください。
""")
