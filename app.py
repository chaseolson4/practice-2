# ============================================================
# STOCK ANALYTICS & PORTFOLIO DASHBOARD — Built with Streamlit
# ============================================================
# This app applies Python-based financial data analytics to:
#   Part 1 — Evaluate one individual stock (trend, RSI, volatility)
#   Part 2 — Analyze a multi-asset portfolio vs. a benchmark
#
# Data is pulled live from Yahoo Finance using yfinance.
# Every widget interaction re-runs the script top-to-bottom.
# ============================================================

# --- IMPORTS ---
import streamlit as st          # web app framework
import yfinance as yf           # Yahoo Finance data
import pandas as pd             # data manipulation
import numpy as np              # math / array operations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION — must be the very first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy  { color: #16a34a; font-weight: 700; font-size: 1.1rem; }
    .signal-sell { color: #dc2626; font-weight: 700; font-size: 1.1rem; }
    .signal-hold { color: #d97706; font-weight: 700; font-size: 1.1rem; }
    .tip-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        font-size: 15px;
        color: #1e3a5f;
        line-height: 1.7;
    }
    .section-intro {
        font-size: 16px;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.7;
    }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=600)  # cache data for 10 minutes to avoid repeated downloads
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and flatten any MultiIndex columns."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))  where RS = avg gain / avg loss over `window` days.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def trend_label(price: float, ma20: float, ma50: float) -> str:
    """Return a trend description based on price vs. moving averages."""
    if price > ma20 > ma50:
        return "🟢 Strong Uptrend"
    elif price < ma20 < ma50:
        return "🔴 Strong Downtrend"
    return "🟡 Mixed / Sideways"


def volatility_label(vol: float) -> str:
    """Classify annualized volatility into High / Medium / Low."""
    if vol > 0.40:
        return f"🔴 High ({vol*100:.1f}%)"
    elif vol > 0.25:
        return f"🟡 Medium ({vol*100:.1f}%)"
    return f"🟢 Low ({vol*100:.1f}%)"


def rsi_signal(rsi: float) -> tuple:
    """Return (label, css_class) based on RSI value."""
    if rsi > 70:
        return "Overbought — Possible Sell Signal", "signal-sell"
    elif rsi < 30:
        return "Oversold — Possible Buy Signal", "signal-buy"
    return "Neutral", "signal-hold"


def trading_recommendation(price, ma20, ma50, rsi, vol) -> tuple:
    """
    Generate a Buy / Sell / Hold recommendation by tallying bullish
    and bearish signals from trend, RSI, and volatility.
    Returns (recommendation_string, explanation_string).
    """
    bull, bear, reasons = 0, 0, []

    if price > ma20 > ma50:
        bull += 1; reasons.append("price above both MAs (uptrend)")
    elif price < ma20 < ma50:
        bear += 1; reasons.append("price below both MAs (downtrend)")

    if rsi < 30:
        bull += 1; reasons.append(f"RSI oversold at {rsi:.1f}")
    elif rsi > 70:
        bear += 1; reasons.append(f"RSI overbought at {rsi:.1f}")

    if vol > 0.40:
        reasons.append(f"high volatility ({vol*100:.1f}%) increases risk")

    note = "; ".join(reasons) + "." if reasons else "No strong directional bias detected."

    if bull > bear:
        return "BUY", f"Bullish signals: {note}"
    elif bear > bull:
        return "SELL", f"Bearish signals: {note}"
    return "HOLD", f"Mixed or neutral signals — {note}"


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.05) -> float:
    """
    Annualized Sharpe Ratio = (mean excess daily return / std daily return) × sqrt(252).
    Risk-free rate defaults to 5% annual → converted to daily.
    """
    daily_rf = risk_free_annual / 252
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


# ============================================================
# APP TITLE & TABS
# ============================================================
st.title("📈 Stock Analytics & Portfolio Dashboard")
st.markdown("*Real-time financial analysis powered by Yahoo Finance data.*")

tab1, tab2 = st.tabs([
    "🔍  Part 1 — Individual Stock Analysis",
    "💼  Part 2 — Portfolio Performance Dashboard"
])


# ============================================================
# TAB 1 — INDIVIDUAL STOCK ANALYSIS
# Steps: Data → Trend (MA) → Momentum (RSI) → Volatility → Recommendation
# ============================================================
with tab1:

    st.header("Individual Stock Analysis")
    st.markdown("""
    <p class="section-intro">
    Enter any stock ticker to retrieve 6 months of daily data from Yahoo Finance.
    The app calculates moving averages, RSI, and annualized volatility,
    then generates a trading recommendation.
    </p>
    """, unsafe_allow_html=True)

    # --- USER INPUT ---
    col_in1, col_in2 = st.columns([1, 3])
    with col_in1:
        ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()

    analyze_btn = st.button("Analyze Stock", type="primary")

    if analyze_btn or ticker_input:
        with st.spinner(f"Fetching 6 months of data for {ticker_input}..."):
            df = fetch_data(ticker_input, "6mo")

        if df.empty or "Close" not in df.columns:
            st.error(f"Could not retrieve data for '{ticker_input}'. Please check the ticker symbol.")
        else:
            close = df["Close"].squeeze()  # ensure 1-D Series

            # ── Step 2: Trend Analysis ────────────────────────────────────
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()

            current_price = float(close.iloc[-1])
            current_ma20  = float(ma20.iloc[-1])
            current_ma50  = float(ma50.iloc[-1])
            trend         = trend_label(current_price, current_ma20, current_ma50)

            # ── Step 3: RSI ───────────────────────────────────────────────
            rsi_series   = calc_rsi(close, 14)
            current_rsi  = float(rsi_series.iloc[-1])
            rsi_lbl, rsi_css = rsi_signal(current_rsi)

            # ── Step 4: Volatility ────────────────────────────────────────
            daily_returns = close.pct_change().dropna()
            # annualize: multiply daily std by sqrt(252 trading days)
            ann_vol = float(daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252))
            vol_lbl = volatility_label(ann_vol)

            # ── Step 5: Recommendation ────────────────────────────────────
            rec, rec_reason = trading_recommendation(
                current_price, current_ma20, current_ma50, current_rsi, ann_vol
            )
            rec_css = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[rec]

            # ── Display: Key Metrics ──────────────────────────────────────
            st.subheader(f"Results for {ticker_input}")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Current Price",    f"${current_price:.2f}")
            with m2:
                st.metric("20-Day MA",         f"${current_ma20:.2f}")
            with m3:
                st.metric("50-Day MA",         f"${current_ma50:.2f}")
            with m4:
                st.metric("14-Day RSI",        f"{current_rsi:.1f}")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"**Trend:** {trend}")
            with r2:
                st.markdown(f"**RSI Signal:** <span class='{rsi_css}'>{rsi_lbl}</span>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"**Volatility:** {vol_lbl}")

            # ── Recommendation Box ────────────────────────────────────────
            st.markdown(f"""
            <div class="tip-box">
                <strong>Trading Recommendation:
                <span class="{rec_css}">{rec}</span></strong><br><br>
                {rec_reason}
            </div>
            """, unsafe_allow_html=True)

            # ── Chart: Price + MAs + RSI ──────────────────────────────────
            st.subheader("Price Chart with Moving Averages & RSI")

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price & Moving Averages", "RSI (14-day)")
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"].squeeze(), high=df["High"].squeeze(),
                low=df["Low"].squeeze(),   close=close,
                name="Price", showlegend=False
            ), row=1, col=1)

            # Moving averages
            fig.add_trace(go.Scatter(x=df.index, y=ma20, name="20-Day MA",
                                     line=dict(color="#2563eb", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ma50, name="50-Day MA",
                                     line=dict(color="#f59e0b", width=1.5)), row=1, col=1)

            # RSI line
            fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI",
                                     line=dict(color="#7c3aed", width=1.5)), row=2, col=1)

            # Overbought / oversold reference lines
            for lvl, color in [(70, "red"), (30, "green")]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=color,
                              opacity=0.5, row=2, col=1)

            fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # ── Written Interpretation ────────────────────────────────────
            st.subheader("Written Interpretation")
            st.markdown(f"""
            <div class="tip-box">
                <strong>What the analysis suggests:</strong><br><br>
                <b>Trend:</b> {trend}. The current price is
                <b>${current_price:.2f}</b>, the 20-day MA is
                <b>${current_ma20:.2f}</b>, and the 50-day MA is
                <b>${current_ma50:.2f}</b>.<br><br>
                <b>RSI ({current_rsi:.1f}):</b> {rsi_lbl}.
                {"RSI above 70 suggests the stock may be overextended and due for a pullback."
                 if current_rsi > 70 else
                 "RSI below 30 suggests the stock may be undervalued or oversold."
                 if current_rsi < 30 else
                 "RSI in the neutral zone indicates no extreme momentum pressure."}<br><br>
                <b>Volatility ({ann_vol*100:.1f}% annualized):</b> {vol_lbl}.
                {"High volatility means larger price swings — proceed with caution."
                 if ann_vol > 0.40 else
                 "Moderate volatility is typical for most equities."
                 if ann_vol > 0.25 else
                 "Low volatility suggests a relatively stable price environment."}<br><br>
                <b>Overall:</b> {rec_reason}
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — PORTFOLIO PERFORMANCE DASHBOARD
# Steps: Setup → Benchmark → Data → Returns → Metrics → Interpretation
# ============================================================
with tab2:

    st.header("Portfolio Performance Dashboard")
    st.markdown("""
    <p class="section-intro">
    Build a 5-stock portfolio, assign weights, and compare 1-year performance
    against a benchmark ETF (default: SPY). Key metrics include total return,
    annualized volatility, and Sharpe ratio.
    </p>
    """, unsafe_allow_html=True)

    # ── Step 1: Portfolio Setup ───────────────────────────────────────────
    st.subheader("Step 1 — Define Your Portfolio")

    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    default_weights = [0.25, 0.25, 0.20, 0.15, 0.15]

    col_t, col_w = st.columns(2)
    with col_t:
        st.markdown("**Stock Tickers** (one per row)")
        raw_tickers = []
        for i, d in enumerate(default_tickers):
            raw_tickers.append(
                st.text_input(f"Stock {i+1}", value=d, key=f"ticker_{i}").upper().strip()
            )

    with col_w:
        st.markdown("**Portfolio Weights** (must sum to 1.00)")
        raw_weights = []
        for i, dw in enumerate(default_weights):
            raw_weights.append(
                st.number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0,
                                value=dw, step=0.05, key=f"weight_{i}")
            )

    # ── Step 2: Benchmark ─────────────────────────────────────────────────
    st.subheader("Step 2 — Benchmark")
    benchmark = st.text_input("Benchmark Ticker", value="SPY").upper().strip()

    weight_sum = round(sum(raw_weights), 4)
    if abs(weight_sum - 1.0) > 0.001:
        st.warning(f"⚠️ Weights sum to {weight_sum:.2f} — they must sum to exactly 1.00.")

    run_portfolio = st.button("Run Portfolio Analysis", type="primary")

    if run_portfolio:
        if abs(weight_sum - 1.0) > 0.001:
            st.error("Please fix portfolio weights before running analysis.")
        else:
            tickers = raw_tickers
            weights = np.array(raw_weights)

            with st.spinner("Downloading 1 year of price data..."):
                price_data = {}
                failed = []
                for t in tickers + [benchmark]:
                    d = fetch_data(t, "1y")
                    if d.empty or "Close" not in d.columns:
                        failed.append(t)
                    else:
                        price_data[t] = d["Close"].squeeze()

            if failed:
                st.error(f"Could not fetch data for: {', '.join(failed)}. Check ticker symbols.")
            else:
                # ── Step 4: Return Calculations ───────────────────────────
                prices_df  = pd.DataFrame({t: price_data[t] for t in tickers}).dropna()
                bench_s    = price_data[benchmark].dropna()

                # Align dates
                common_idx = prices_df.index.intersection(bench_s.index)
                prices_df  = prices_df.loc[common_idx]
                bench_s    = bench_s.loc[common_idx]

                # Daily returns for each stock
                stock_returns = prices_df.pct_change().dropna()

                # Portfolio daily return = weighted sum of individual returns
                port_returns  = stock_returns.dot(weights)

                # Benchmark daily returns
                bench_returns = bench_s.pct_change().dropna()

                # ── Step 5: Performance Metrics ───────────────────────────
                # Total return = (final price / initial price) - 1
                port_total  = float((1 + port_returns).prod() - 1)
                bench_total = float((1 + bench_returns).prod() - 1)
                outperf     = port_total - bench_total

                # Annualized volatility = daily std × sqrt(252)
                port_vol  = float(port_returns.std()  * np.sqrt(252))
                bench_vol = float(bench_returns.std() * np.sqrt(252))

                # Sharpe ratio (risk-free = 5%)
                port_sharpe  = sharpe_ratio(port_returns)
                bench_sharpe = sharpe_ratio(bench_returns)

                # ── Display: Performance Metrics ──────────────────────────
                st.subheader("Performance Metrics")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Portfolio Total Return", f"{port_total*100:.2f}%")
                with c2:
                    st.metric(f"{benchmark} Total Return", f"{bench_total*100:.2f}%")
                with c3:
                    delta_sign = "▲" if outperf >= 0 else "▼"
                    st.metric("Outperformance", f"{delta_sign} {abs(outperf)*100:.2f}%")
                with c4:
                    st.metric("Portfolio Sharpe Ratio", f"{port_sharpe:.2f}")

                v1, v2 = st.columns(2)
                with v1:
                    st.metric("Portfolio Ann. Volatility", f"{port_vol*100:.2f}%")
                with v2:
                    st.metric(f"{benchmark} Ann. Volatility", f"{bench_vol*100:.2f}%")

                # ── Chart: Cumulative Returns ─────────────────────────────
                st.subheader("Cumulative Return: Portfolio vs. Benchmark")

                cum_port  = (1 + port_returns).cumprod() - 1
                cum_bench = (1 + bench_returns).cumprod() - 1

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=cum_port.index, y=cum_port * 100,
                    name="Portfolio", line=dict(color="#2563eb", width=2)
                ))
                fig2.add_trace(go.Scatter(
                    x=cum_bench.index, y=cum_bench * 100,
                    name=benchmark, line=dict(color="#f59e0b", width=2, dash="dash")
                ))
                fig2.update_layout(
                    yaxis_title="Cumulative Return (%)",
                    height=380,
                    legend=dict(orientation="h", y=1.05)
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ── Chart: Individual Stock Returns ───────────────────────
                st.subheader("Individual Stock Returns")

                indiv_returns = {}
                for t in tickers:
                    s = price_data[t].loc[common_idx]
                    indiv_returns[t] = float((s.iloc[-1] / s.iloc[0]) - 1) * 100

                fig3 = go.Figure(go.Bar(
                    x=list(indiv_returns.keys()),
                    y=list(indiv_returns.values()),
                    marker_color=["#22c55e" if v >= 0 else "#ef4444"
                                  for v in indiv_returns.values()]
                ))
                fig3.update_layout(yaxis_title="Total Return (%)", height=320)
                st.plotly_chart(fig3, use_container_width=True)

                # ── Portfolio Weights Pie ─────────────────────────────────
                st.subheader("Portfolio Allocation")
                fig4 = go.Figure(go.Pie(
                    labels=tickers, values=weights,
                    hole=0.4, textinfo="label+percent"
                ))
                fig4.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

                # ── Step 6: Written Interpretation ────────────────────────
                st.subheader("Written Interpretation")
                outperf_text = (
                    f"outperformed {benchmark} by <b>{outperf*100:.2f}%</b>"
                    if outperf >= 0
                    else f"underperformed {benchmark} by <b>{abs(outperf)*100:.2f}%</b>"
                )
                risk_text = (
                    "more risky" if port_vol > bench_vol else "less risky"
                )
                sharpe_text = (
                    "efficient — generating good return per unit of risk"
                    if port_sharpe > 1.0
                    else "moderately efficient" if port_sharpe > 0.5
                    else "inefficient — low return relative to the risk taken"
                )

                st.markdown(f"""
                <div class="tip-box">
                    <strong>Portfolio Analysis Summary:</strong><br><br>
                    <b>Performance:</b> Your portfolio {outperf_text} over the past year,
                    returning <b>{port_total*100:.2f}%</b> vs.
                    <b>{bench_total*100:.2f}%</b> for {benchmark}.<br><br>
                    <b>Risk:</b> Your portfolio was <b>{risk_text}</b> than the benchmark.
                    Portfolio annualized volatility was <b>{port_vol*100:.2f}%</b> vs.
                    <b>{bench_vol*100:.2f}%</b> for {benchmark}.<br><br>
                    <b>Efficiency (Sharpe Ratio):</b> A Sharpe ratio of
                    <b>{port_sharpe:.2f}</b> (vs. {bench_sharpe:.2f} for {benchmark}) means your
                    portfolio was <b>{sharpe_text}</b>. A Sharpe ratio above 1.0 is
                    generally considered strong.<br><br>
                    <b>Top performer:</b> {max(indiv_returns, key=indiv_returns.get)}
                    at {max(indiv_returns.values()):.1f}% &nbsp;|&nbsp;
                    <b>Weakest performer:</b> {min(indiv_returns, key=indiv_returns.get)}
                    at {min(indiv_returns.values()):.1f}%
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Built with Python & Streamlit · Data sourced from Yahoo Finance via yfinance · For educational purposes only.")
