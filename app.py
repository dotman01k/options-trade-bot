import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Options Trade Dashboard", page_icon="📈", layout="wide")

APP_VERSION = "v4 smart dashboard"

st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
    }
    .metric-note {
        color: #9ca3af;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 10px;
    }
    .info-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px;
    }
    .info-label {
        color: #93c5fd;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    .info-value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 800;
    }
    .signal-card-buy {
        background: #062e1f;
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 14px;
        min-height: 170px;
    }
    .signal-card-watch {
        background: #3b2f00;
        border: 1px solid #facc15;
        border-radius: 12px;
        padding: 14px;
        min-height: 170px;
    }
    .signal-card-avoid {
        background: #3b0a0a;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 14px;
        min-height: 170px;
    }
    .card-title {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .card-main {
        color: #d1d5db;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .small-note {
        color: #9ca3af;
        font-size: 0.85rem;
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


def fetch_data(symbol: str):
    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options)
    return ticker, expirations


def fetch_chain(symbol: str, expiration: str):
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


def get_trend_strength(ticker):
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty or "Close" not in hist.columns:
        return 50.0, "Neutral"

    close = hist["Close"].dropna()
    if len(close) < 21:
        return 50.0, "Neutral"

    ema9 = close.ewm(span=9).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]

    if ema9 > ema21:
        return 80.0, "Bullish"
    if ema9 < ema21:
        return 20.0, "Bearish"
    return 50.0, "Neutral"


def clean_df(df: pd.DataFrame, expiration: str, underlying_price: float):
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
    df["spread_pct"] = np.where(df["ask"] > 0, df["spread"] / df["ask"] * 100, 100).round(2)

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

    df["delta"] = np.where(df["option_type"] == "Call", scaled, -scaled).round(2)
    df["notional"] = (df["mid"] * 100).round(2)

    return df


def score(df: pd.DataFrame, option_type: str, trend_strength: float):
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

    trend_boost = trend_strength if option_type == "Call" else 100 - trend_strength
    side["confidence"] = (
        side["score"] * 0.40
        + side["win_probability"] * 0.30
        + trend_boost * 0.30
    ).clip(5, 100).round(1)

    side["signal"] = np.select(
        [
            side["confidence"] >= 75,
            side["confidence"] >= 60,
        ],
        ["STRONG BUY", "BUY"],
        default="AVOID",
    )

    signal_order = {"STRONG BUY": 0, "BUY": 1, "AVOID": 2}
    side["signal_rank"] = side["signal"].map(signal_order)
    side = side.sort_values(
        ["signal_rank", "confidence", "score", "volume", "open_interest"],
        ascending=[True, False, False, False, False],
    )
    return side.drop(columns=["signal_rank"])


def signal_card_class(signal: str):
    if signal == "STRONG BUY":
        return "signal-card-buy"
    if signal == "BUY":
        return "signal-card-watch"
    return "signal-card-avoid"


def signal_badge(signal: str):
    if signal == "STRONG BUY":
        return "🟢"
    if signal == "BUY":
        return "🟡"
    return "🔴"


def render_idea_card(title: str, row):
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
                <div style="font-size:1.1rem;font-weight:800;margin-bottom:8px;">{badge} {row['symbol']}</div>
                <div><strong>Signal:</strong> {signal}</div>
                <div><strong>Confidence:</strong> {row['confidence']:.1f}</div>
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


def style_signals(df: pd.DataFrame):
    def color_signal(val):
        if val == "STRONG BUY":
            return "background-color: rgba(0, 200, 0, 0.18); font-weight: 700;"
        if val == "BUY":
            return "background-color: rgba(255, 193, 7, 0.22); font-weight: 700;"
        if val == "AVOID":
            return "background-color: rgba(255, 0, 0, 0.16); font-weight: 700;"
        return ""

    def color_atm(val):
        if val == "ATM":
            return "background-color: rgba(0, 123, 255, 0.18); font-weight: 700;"
        return ""

    return (
        df.style
        .map(color_signal, subset=["Signal"])
        .map(color_atm, subset=["ATM"])
        .format(
            {
                "Bid": "${:.2f}",
                "Ask": "${:.2f}",
                "Mid": "${:.2f}",
                "Spread %": "{:.2f}%",
                "Delta": "{:.2f}",
                "IV": "{:.2%}",
                "Distance %": "{:.2f}%",
                "Score": "{:.2f}",
                "Win %": "{:.1f}%",
                "Confidence": "{:.1f}",
                "Est. Cost": "${:.2f}",
            }
        )
    )


def show_table(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty:
        st.info("No contracts matched your filters.")
        return

    pretty = df[
        [
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
            "confidence",
            "notional",
        ]
    ].copy()

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
        "Confidence",
        "Est. Cost",
    ]

    st.dataframe(style_signals(pretty), width="stretch", hide_index=True)


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📈 Options Trade Dashboard</div>
        <div class="hero-subtitle">Smarter scanner with confidence score, ATM highlighting, and signal ranking.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(APP_VERSION)

left_top, right_top = st.columns([1.6, 1])
with left_top:
    symbol = st.text_input("Ticker", "SPY").upper().strip()
with right_top:
    st.markdown(
        "<div class='small-note' style='padding-top: 12px;'>Tip: liquid tickers usually give cleaner options data and tighter spreads.</div>",
        unsafe_allow_html=True,
    )

st.markdown("### Scanner Filters")
f1, f2, f3 = st.columns(3)
with f1:
    top_n = st.slider("Top contracts", 5, 25, 10)
with f2:
    min_volume = st.number_input("Minimum volume", min_value=0, value=50, step=10)
with f3:
    min_oi = st.number_input("Minimum open interest", min_value=0, value=100, step=50)

max_spread_pct = st.slider("Maximum spread %", 1, 50, 15)

try:
    ticker, expirations = fetch_data(symbol)
    underlying_price = get_underlying_price(ticker)
    trend_strength, trend_label = get_trend_strength(ticker)
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

calls_scored = score(filtered, "Call", trend_strength)
puts_scored = score(filtered, "Put", trend_strength)

calls = calls_scored.head(top_n)
puts = puts_scored.head(top_n)

scored_all = pd.concat([calls_scored, puts_scored], ignore_index=True)

m1, m2, m3, m4 = st.columns(4)
atm_count = int((scored_all["atm_flag"] == "ATM").sum()) if not scored_all.empty else 0
buy_count = int(scored_all["signal"].isin(["STRONG BUY", "BUY"]).sum()) if not scored_all.empty else 0

m1.metric("Underlying", symbol)
m2.metric("Spot Price", f"${underlying_price:,.2f}" if underlying_price else "N/A")
m3.metric("Trend Bias", trend_label)
m4.metric("Actionable Signals", f"{buy_count}")

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f"<div class='info-card'><div class='info-label'>Selected Expiration</div><div class='info-value'>{expiry}</div></div>",
        unsafe_allow_html=True,
    )
with i2:
    best_call_conf = f"{calls.iloc[0]['confidence']:.1f}" if not calls.empty else "N/A"
    st.markdown(
        f"<div class='info-card'><div class='info-label'>Best Call Confidence</div><div class='info-value'>{best_call_conf}</div></div>",
        unsafe_allow_html=True,
    )
with i3:
    best_put_conf = f"{puts.iloc[0]['confidence']:.1f}" if not puts.empty else "N/A"
    st.markdown(
        f"<div class='info-card'><div class='info-label'>Best Put Confidence</div><div class='info-value'>{best_put_conf}</div></div>",
        unsafe_allow_html=True,
    )

c1, c2 = st.columns(2)
with c1:
    render_idea_card("Best Call", calls.iloc[0] if not calls.empty else None)
with c2:
    render_idea_card("Best Put", puts.iloc[0] if not puts.empty else None)

st.info(
    "Confidence combines contract quality, win model, and trend alignment. It is a model score for decision support, not a guarantee."
)

tab1, tab2, tab3 = st.tabs(["Top Calls", "Top Puts", "ATM Snapshot"])

with tab1:
    show_table(calls, f"Top Call Contracts for {symbol} {expiry}")

with tab2:
    show_table(puts, f"Top Put Contracts for {symbol} {expiry}")

with tab3:
    atm_df = scored_all[scored_all["atm_flag"] == "ATM"].copy()
    if atm_df.empty:
        st.info("No ATM contracts matched the current filters.")
    else:
        show_table(atm_df.head(20), f"ATM Contracts for {symbol} {expiry}")