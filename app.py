import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Options Scanner", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #172554 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(59,130,246,.22), rgba(168,85,247,.18));
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    .stat-card {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,.15);
    }
    .stat-label {
        color: #93c5fd;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .stat-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .signal-card-buy {
        background: linear-gradient(135deg, rgba(34,197,94,.22), rgba(22,163,74,.14));
        border: 1px solid rgba(34,197,94,.35);
        border-radius: 18px;
        padding: 16px;
        min-height: 150px;
    }
    .signal-card-watch {
        background: linear-gradient(135deg, rgba(250,204,21,.22), rgba(245,158,11,.12));
        border: 1px solid rgba(250,204,21,.35);
        border-radius: 18px;
        padding: 16px;
        min-height: 150px;
    }
    .signal-card-avoid {
        background: linear-gradient(135deg, rgba(248,113,113,.22), rgba(239,68,68,.12));
        border: 1px solid rgba(248,113,113,.35);
        border-radius: 18px;
        padding: 16px;
        min-height: 150px;
    }
    .card-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .card-main {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.5;
    }
    .small-note {
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        padding: 12px 14px;
        border-radius: 18px;
        box-shadow: 0 10px 22px rgba(0,0,0,.14);
    }
    div[data-testid="stTabs"] button {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def normalize(series: pd.Series):
    if series.empty:
        return pd.Series(dtype=float)
    s = series.fillna(0).astype(float)
    mn = s.min()
    mx = s.max()
    if math.isclose(mx, mn):
        return pd.Series([50.0] * len(s), index=s.index)
    return ((s - mn) / (mx - mn) * 100).round(2)


def fetch_data(symbol):
    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options)
    return ticker, expirations


def fetch_chain(symbol, expiration):
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiration)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    calls["option_type"] = "Call"
    puts["option_type"] = "Put"

    return pd.concat([calls, puts], ignore_index=True)


def get_underlying_price(ticker):
    fast_info = getattr(ticker, "fast_info", {}) or {}
    for key in ["lastPrice", "regularMarketPrice", "previousClose", "open"]:
        value = safe_float(fast_info.get(key))
        if value > 0:
            return value

    history = ticker.history(period="1d", interval="1m")
    if not history.empty and "Close" in history.columns:
        return safe_float(history["Close"].dropna().iloc[-1])
    return 0.0


def clean_df(df, expiration, underlying_price):
    df = df.rename(
        columns={
            "contractSymbol": "symbol",
            "lastPrice": "last",
            "openInterest": "open_interest",
            "impliedVolatility": "iv",
        }
    )

    for col in ["bid", "ask", "last", "volume", "open_interest", "strike", "iv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    df["expiration_date"] = expiration

    df["mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2,
        np.where(df["last"] > 0, df["last"], np.where(df["ask"] > 0, df["ask"], df["bid"])),
    )

    df["spread"] = (df["ask"] - df["bid"]).clip(lower=0)
    df["spread_pct"] = np.where(df["ask"] > 0, df["spread"] / df["ask"] * 100, 100)
    df["distance_from_spot_pct"] = np.where(
        underlying_price > 0,
        ((df["strike"] - underlying_price).abs() / underlying_price) * 100,
        999.0,
    ).round(2)
    df["atm_flag"] = np.where(df["distance_from_spot_pct"] <= 1.0, "ATM", "")

    midpoint = underlying_price if underlying_price > 0 else (df["strike"].median() if not df.empty else 0)
    dist = (df["strike"] - midpoint).abs()
    scaled = 1 - normalize(dist) / 100
    scaled = scaled.clip(0.05, 0.95)

    df["delta"] = np.where(df["option_type"] == "Call", scaled, -scaled)
    df["notional"] = (df["mid"] * 100).round(2)

    return df


def score(df, option_type):
    side = df[df["option_type"] == option_type].copy()

    if side.empty:
        return side

    liquidity = normalize(np.log1p(side["volume"]) + np.log1p(side["open_interest"]))
    spread = 100 - normalize(side["spread_pct"])
    atm_fit = 100 - normalize(side["distance_from_spot_pct"])

    target = 0.35 if option_type == "Call" else -0.35
    delta_fit = 100 - (abs(side["delta"] - target) / 0.35 * 100).clip(0, 100)

    side["score"] = (
        liquidity * 0.35
        + spread * 0.25
        + delta_fit * 0.20
        + atm_fit * 0.20
    ).round(2)

    side["win_probability"] = (
        side["score"] * 0.55
        + liquidity * 0.15
        + spread * 0.10
        + atm_fit * 0.20
    ).clip(5, 95).round(1)

    side["signal"] = np.select(
        [
            (side["score"] >= 78) & (side["win_probability"] >= 70) & (side["spread_pct"] <= 12),
            (side["score"] >= 65) & (side["win_probability"] >= 58) & (side["spread_pct"] <= 18),
        ],
        ["BUY", "WATCH"],
        default="AVOID",
    )

    signal_order = {"BUY": 0, "WATCH": 1, "AVOID": 2}
    side["signal_rank"] = side["signal"].map(signal_order)
    side = side.sort_values(
        ["signal_rank", "score", "volume", "open_interest"],
        ascending=[True, False, False, False],
    )
    return side.drop(columns=["signal_rank"])


def signal_card_class(signal):
    if signal == "BUY":
        return "signal-card-buy"
    if signal == "WATCH":
        return "signal-card-watch"
    return "signal-card-avoid"


def signal_badge(signal):
    if signal == "BUY":
        return "🟢"
    if signal == "WATCH":
        return "🟡"
    return "🔴"


def render_idea_card(title, row):
    if row is None:
        st.info(f"No {title.lower()} matched your filters.")
        return

    signal = row["signal"]
    card_class = signal_card_class(signal)
    badge = signal_badge(signal)
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="card-title">{title}</div>
            <div class="card-main">
                <div style="font-size:1.1rem;font-weight:800;margin-bottom:6px;">{badge} {row['symbol']}</div>
                <div><strong>Signal:</strong> {signal}</div>
                <div><strong>Score:</strong> {row['score']:.2f}</div>
                <div><strong>Win %:</strong> {row['win_probability']:.1f}%</div>
                <div><strong>Ask:</strong> ${row['ask']:.2f}</div>
                <div><strong>Strike:</strong> ${row['strike']:.2f}</div>
                <div><strong>ATM:</strong> {row['atm_flag'] if row['atm_flag'] else 'No'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_signals(df):
    def color_signal(val):
        if val == "BUY":
            return "background-color: rgba(0, 200, 0, 0.18); font-weight: 700;"
        if val == "WATCH":
            return "background-color: rgba(255, 193, 7, 0.22); font-weight: 700;"
        if val == "AVOID":
            return "background-color: rgba(255, 0, 0, 0.16); font-weight: 700;"
        return ""

    def color_atm(val):
        if val == "ATM":
            return "background-color: rgba(0, 123, 255, 0.18); font-weight: 700;"
        return ""

    styled = (
        df.style
        .map(color_signal, subset=["Signal"])
        .map(color_atm, subset=["ATM"])
        .format({
            "Bid": "${:.2f}",
            "Ask": "${:.2f}",
            "Mid": "${:.2f}",
            "Spread %": "{:.2f}%",
            "Delta": "{:.2f}",
            "IV": "{:.2%}",
            "Score": "{:.2f}",
            "Win %": "{:.1f}%",
            "Est. Cost": "${:.2f}",
            "Distance %": "{:.2f}%",
        })
    )
    return styled


def show_table(df, title):
    st.subheader(title)
    if df.empty:
        st.info("No contracts matched your filters.")
        return

    pretty = df[[
        "signal",
        "atm_flag",
        "symbol",
        "strike",
        "expiration_date",
        "bid",
        "ask",
        "mid",
        "spread_pct",
        "volume",
        "open_interest",
        "delta",
        "iv",
        "distance_from_spot_pct",
        "score",
        "win_probability",
        "notional",
    ]].copy()

    pretty.columns = [
        "Signal",
        "ATM",
        "Contract",
        "Strike",
        "Expiry",
        "Bid",
        "Ask",
        "Mid",
        "Spread %",
        "Volume",
        "Open Interest",
        "Delta",
        "IV",
        "Distance %",
        "Score",
        "Win %",
        "Est. Cost",
    ]

    st.dataframe(style_signals(pretty), width="stretch", hide_index=True)


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📈 Options Trade Dashboard</div>
        <div class="hero-subtitle">A more colorful and user friendly scanner for calls, puts, signals, ATM setups, and win probability.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_top, right_top = st.columns([1.5, 1])
with left_top:
    symbol = st.text_input("Ticker", "SPY").upper().strip()
with right_top:
    st.markdown("<div class='small-note' style='padding-top: 10px;'>Tip: Start with liquid tickers like SPY, QQQ, NVDA, AAPL, or TSLA.</div>", unsafe_allow_html=True)

st.markdown("### Scanner Filters")
left_filter, right_filter = st.columns(2)
with left_filter:
    top_n = st.slider("Top contracts", 5, 25, 10)
    min_volume = st.number_input("Minimum volume", min_value=0, value=50, step=10)
with right_filter:
    min_oi = st.number_input("Minimum open interest", min_value=0, value=100, step=50)
    max_spread_pct = st.slider("Maximum spread %", 1, 50, 15)

try:
    ticker, expirations = fetch_data(symbol)
    underlying_price = get_underlying_price(ticker)
except Exception as e:
    st.error(f"Could not load market data for {symbol}: {e}")
    st.stop()

if not expirations:
    st.error("No options found for this ticker.")
    st.stop()

expiry = st.selectbox("Expiration", expirations)

try:
    chain = fetch_chain(symbol, expiry)
    df = clean_df(chain, expiry, underlying_price)
except Exception as e:
    st.error(f"Could not load option chain: {e}")
    st.stop()

filtered = df[
    (df["volume"] >= min_volume)
    & (df["open_interest"] >= min_oi)
    & (df["spread_pct"] <= max_spread_pct)
].copy()

calls = score(filtered, "Call").head(top_n)
puts = score(filtered, "Put").head(top_n)

m1, m2, m3, m4 = st.columns(4)
atm_count = int((filtered["atm_flag"] == "ATM").sum()) if not filtered.empty else 0
buy_count = int((pd.concat([calls, puts])["signal"] == "BUY").sum()) if (not calls.empty or not puts.empty) else 0
m1.metric("Underlying", symbol)
m2.metric("Spot Price", f"${underlying_price:,.2f}" if underlying_price else "N/A")
m3.metric("ATM Contracts", f"{atm_count}")
m4.metric("Buy Signals", f"{buy_count}")

stat1, stat2, stat3 = st.columns(3)
with stat1:
    st.markdown(f"<div class='stat-card'><div class='stat-label'>Top Expiration</div><div class='stat-value'>{expiry}</div></div>", unsafe_allow_html=True)
with stat2:
    top_call_score = f"{calls.iloc[0]['score']:.2f}" if not calls.empty else "N/A"
    st.markdown(f"<div class='stat-card'><div class='stat-label'>Best Call Score</div><div class='stat-value'>{top_call_score}</div></div>", unsafe_allow_html=True)
with stat3:
    top_put_score = f"{puts.iloc[0]['score']:.2f}" if not puts.empty else "N/A"
    st.markdown(f"<div class='stat-card'><div class='stat-label'>Best Put Score</div><div class='stat-value'>{top_put_score}</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    render_idea_card("Best Call", calls.iloc[0] if not calls.empty else None)

with col2:
    render_idea_card("Best Put", puts.iloc[0] if not puts.empty else None)

st.info(
    "Signal logic: BUY favors stronger liquidity, tighter spreads, near-ATM contracts, and delta close to the target. Win % is a model score, not a guaranteed outcome."
)

tab1, tab2, tab3 = st.tabs(["Top Calls", "Top Puts", "ATM Snapshot"])

with tab1:
    show_table(calls, f"Top Call Contracts for {symbol} {expiry}")

with tab2:
    show_table(puts, f"Top Put Contracts for {symbol} {expiry}")

with tab3:
    scored_all = pd.concat([score(filtered, "Call"), score(filtered, "Put")], ignore_index=True)
    atm_df = scored_all[scored_all["atm_flag"] == "ATM"].copy()

    if atm_df.empty:
        st.info("No ATM contracts matched the current filters.")
    else:
        show_table(atm_df.head(20), f"ATM Contracts for {symbol} {expiry}")
