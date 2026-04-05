import math
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Beginner Options Dashboard", page_icon="📈", layout="wide")

APP_VERSION = "v10 beginner dashboard + market scanner"

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
    .simple-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px;
    }
    .simple-label {
        color: #93c5fd;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .simple-value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 800;
    }
    .signal-card-strong {
        background: #062e1f;
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 16px;
        min-height: 280px;
    }
    .signal-card-buy {
        background: #3b2f00;
        border: 1px solid #facc15;
        border-radius: 12px;
        padding: 16px;
        min-height: 280px;
    }
    .signal-card-watch {
        background: #1f2937;
        border: 1px solid #64748b;
        border-radius: 12px;
        padding: 16px;
        min-height: 280px;
    }
    .signal-card-no {
        background: #3b0a0a;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        min-height: 280px;
    }
    .card-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .card-main {
        color: #d1d5db;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .help-note {
        color: #9ca3af;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 10px;
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


def calculate_contracts(capital: float, option_price: float):
    if option_price <= 0:
        return 0
    return int(capital // (option_price * 100))


def clean_symbol(user_input: str):
    name_map = {
        "MICROSOFT": "MSFT",
        "MSFT": "MSFT",
        "TESLA": "TSLA",
        "TSLA": "TSLA",
        "APPLE": "AAPL",
        "AAPL": "AAPL",
        "NVIDIA": "NVDA",
        "NVDA": "NVDA",
        "AMAZON": "AMZN",
        "AMZN": "AMZN",
        "GOOGLE": "GOOGL",
        "GOOGL": "GOOGL",
        "ALPHABET": "GOOGL",
        "META": "META",
        "FACEBOOK": "META",
        "NETFLIX": "NFLX",
        "NFLX": "NFLX",
        "SPY": "SPY",
        "S&P500": "SPY",
        "S&P 500": "SPY",
        "SP500": "SPY",
        "S&P": "SPY",
        "QQQ": "QQQ",
        "AMD": "AMD",
        "COIN": "COIN",
        "PALANTIR": "PLTR",
        "PLTR": "PLTR",
    }
    raw = (user_input or "").upper().strip()
    cleaned = raw.replace("$", "").strip()
    symbol = name_map.get(cleaned, cleaned)
    return raw, symbol


def validate_ticker(symbol: str):
    ticker = yf.Ticker(symbol)
    test = ticker.history(period="5d")
    if test.empty:
        return None, False
    return ticker, True


def get_spot_price(ticker):
    fast_info = getattr(ticker, "fast_info", {}) or {}
    for key in ["lastPrice", "regularMarketPrice", "previousClose", "open"]:
        value = safe_float(fast_info.get(key))
        if value > 0:
            return value

    hist = ticker.history(period="1d", interval="1m")
    if not hist.empty and "Close" in hist.columns:
        return safe_float(hist["Close"].dropna().iloc[-1])
    return 0.0


def get_stock_context(ticker):
    hourly = ticker.history(period="20d", interval="1h")
    intraday = ticker.history(period="5d", interval="15m")

    if hourly.empty or intraday.empty:
        return {
            "trend_label": "Neutral",
            "trend_strength": 50.0,
            "price": 0.0,
            "ema9": None,
            "ema21": None,
            "rsi": None,
            "vwap": None,
            "relative_volume": None,
            "support": None,
            "resistance": None,
        }

    hourly_close = hourly["Close"].dropna()
    ema9 = hourly_close.ewm(span=9).mean().iloc[-1]
    ema21 = hourly_close.ewm(span=21).mean().iloc[-1]

    if ema9 > ema21:
        trend_label = "Bullish"
        trend_strength = 80.0
    elif ema9 < ema21:
        trend_label = "Bearish"
        trend_strength = 20.0
    else:
        trend_label = "Neutral"
        trend_strength = 50.0

    price = float(intraday["Close"].dropna().iloc[-1])

    delta = intraday["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = safe_float(rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else 50)

    typical_price = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
    cum_vol = intraday["Volume"].cumsum()
    vwap_series = (typical_price * intraday["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
    vwap = safe_float(vwap_series.dropna().iloc[-1] if not vwap_series.dropna().empty else price)

    last_volume = safe_float(intraday["Volume"].iloc[-1])
    avg_volume = safe_float(intraday["Volume"].tail(20).mean(), 1)
    relative_volume = round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0

    support = safe_float(hourly["Low"].tail(20).min())
    resistance = safe_float(hourly["High"].tail(20).max())

    return {
        "trend_label": trend_label,
        "trend_strength": trend_strength,
        "price": price,
        "ema9": float(ema9),
        "ema21": float(ema21),
        "rsi": float(rsi),
        "vwap": float(vwap),
        "relative_volume": float(relative_volume),
        "support": float(support),
        "resistance": float(resistance),
    }


def fetch_expirations(symbol: str):
    ticker = yf.Ticker(symbol)
    return list(ticker.options)


def fetch_chain(symbol: str, expiration: str):
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiration)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    calls["option_type"] = "Call"
    puts["option_type"] = "Put"

    return pd.concat([calls, puts], ignore_index=True)


def clean_chain_df(df: pd.DataFrame, expiration: str, spot_price: float):
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
    df["spread_pct"] = np.where(df["ask"] > 0, (df["spread"] / df["ask"]) * 100, 100).round(2)

    df["distance_from_spot_pct"] = np.where(
        spot_price > 0,
        ((df["strike"] - spot_price).abs() / spot_price) * 100,
        999.0,
    ).round(2)

    df["atm_flag"] = np.where(df["distance_from_spot_pct"] <= 1.0, "ATM", "")

    midpoint = spot_price if spot_price > 0 else df["strike"].median()
    strike_dist = (df["strike"] - midpoint).abs()
    scaled = 1 - normalize(strike_dist) / 100
    scaled = scaled.clip(0.05, 0.95)
    df["delta"] = np.where(df["option_type"] == "Call", scaled, -scaled).round(2)

    return df


def add_trade_plan(df: pd.DataFrame):
    df["entry_price"] = df["ask"].round(2)
    df["stop_price"] = (df["ask"] * 0.75).round(2)
    df["target_price"] = (df["ask"] * 1.40).round(2)
    df["risk_per_contract"] = ((df["entry_price"] - df["stop_price"]) * 100).round(2)
    df["reward_per_contract"] = ((df["target_price"] - df["entry_price"]) * 100).round(2)
    df["rr_ratio"] = np.where(
        df["risk_per_contract"] > 0,
        df["reward_per_contract"] / df["risk_per_contract"],
        0,
    ).round(2)
    return df


def score_options(df: pd.DataFrame, option_type: str, stock_ctx: dict, capital: float):
    side = df[df["option_type"] == option_type].copy()
    if side.empty:
        return side

    liquidity_score = normalize(np.log1p(side["volume"]) + np.log1p(side["open_interest"]))
    spread_score = 100 - normalize(side["spread_pct"])
    atm_score = 100 - normalize(side["distance_from_spot_pct"])

    target_delta = 0.35 if option_type == "Call" else -0.35
    delta_score = 100 - (abs(side["delta"] - target_delta) / 0.35 * 100).clip(0, 100)

    side["base_score"] = (
        liquidity_score * 0.35
        + spread_score * 0.25
        + delta_score * 0.20
        + atm_score * 0.20
    ).round(2)

    trend_label = stock_ctx["trend_label"]
    price = stock_ctx["price"]
    vwap = stock_ctx["vwap"]
    rsi = stock_ctx["rsi"]
    rel_vol = stock_ctx["relative_volume"]

    call_trend_ok = trend_label == "Bullish"
    put_trend_ok = trend_label == "Bearish"
    price_above_vwap = price > vwap if vwap else False
    price_below_vwap = price < vwap if vwap else False
    call_rsi_ok = 55 <= rsi <= 70 if rsi is not None else False
    put_rsi_ok = 30 <= rsi <= 45 if rsi is not None else False
    rel_vol_ok = rel_vol >= 1.2 if rel_vol is not None else False

    if option_type == "Call":
        side["trend_confirmation"] = call_trend_ok
        side["vwap_confirmation"] = price_above_vwap
        side["rsi_confirmation"] = call_rsi_ok
    else:
        side["trend_confirmation"] = put_trend_ok
        side["vwap_confirmation"] = price_below_vwap
        side["rsi_confirmation"] = put_rsi_ok

    side["volume_confirmation"] = rel_vol_ok
    side["spread_confirmation"] = side["spread_pct"] <= 12

    side["confirmation_count"] = (
        side["trend_confirmation"].astype(int)
        + side["vwap_confirmation"].astype(int)
        + side["rsi_confirmation"].astype(int)
        + side["volume_confirmation"].astype(int)
        + side["spread_confirmation"].astype(int)
    )

    side["win_probability"] = (
        side["base_score"] * 0.55
        + (side["confirmation_count"] / 5 * 100) * 0.45
    ).clip(5, 95).round(1)

    side["confidence"] = (
        side["base_score"] * 0.45
        + side["win_probability"] * 0.35
        + (side["confirmation_count"] / 5 * 100) * 0.20
    ).clip(5, 100).round(1)

    side["signal"] = np.select(
        [
            (side["confirmation_count"] >= 5) & (side["confidence"] >= 78),
            (side["confirmation_count"] >= 4) & (side["confidence"] >= 65),
            (side["confirmation_count"] >= 3) & (side["confidence"] >= 55),
        ],
        ["STRONG BUY", "BUY", "WATCH"],
        default="NO TRADE",
    )

    side["contracts"] = side["ask"].apply(lambda x: calculate_contracts(capital, x))
    side["position_cost"] = (side["contracts"] * side["ask"] * 100).round(2)
    side["capital_used_pct"] = np.where(
        capital > 0,
        (side["position_cost"] / capital) * 100,
        0,
    ).round(1)

    side = add_trade_plan(side)

    reasons = []
    for _, row in side.iterrows():
        pieces = []
        if row["trend_confirmation"]:
            pieces.append("trend agrees")
        if row["vwap_confirmation"]:
            pieces.append("price confirms direction")
        if row["rsi_confirmation"]:
            pieces.append("RSI looks healthy")
        if row["volume_confirmation"]:
            pieces.append("relative volume is strong")
        if row["spread_confirmation"]:
            pieces.append("spread is tight")

        if not pieces:
            reasons.append("Not enough confirmation")
        else:
            reasons.append(", ".join(pieces))

    side["reason"] = reasons

    order_map = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "NO TRADE": 3}
    side["signal_rank"] = side["signal"].map(order_map)
    side = side.sort_values(
        ["signal_rank", "confidence", "base_score", "volume", "open_interest"],
        ascending=[True, False, False, False, False],
    )

    return side.drop(columns=["signal_rank"])


def signal_card_class(signal: str):
    if signal == "STRONG BUY":
        return "signal-card-strong"
    if signal == "BUY":
        return "signal-card-buy"
    if signal == "WATCH":
        return "signal-card-watch"
    return "signal-card-no"


def signal_emoji(signal: str):
    if signal == "STRONG BUY":
        return "🟢"
    if signal == "BUY":
        return "🟡"
    if signal == "WATCH":
        return "👀"
    return "🔴"


def render_trade_card(title: str, row):
    if row is None:
        st.info(f"No {title.lower()} found.")
        return

    signal = row["signal"]
    st.markdown(
        f"""
        <div class="{signal_card_class(signal)}">
            <div class="card-title">{title}</div>
            <div class="card-main">
                <div style="font-size:1.08rem;font-weight:800;margin-bottom:8px;">{signal_emoji(signal)} {row['symbol']}</div>
                <div><strong>Signal:</strong> {row['signal']}</div>
                <div><strong>Confidence:</strong> {row['confidence']:.1f}</div>
                <div><strong>Win %:</strong> {row['win_probability']:.1f}%</div>
                <div><strong>Entry:</strong> ${row['entry_price']:.2f}</div>
                <div><strong>Stop:</strong> ${row['stop_price']:.2f}</div>
                <div><strong>Target:</strong> ${row['target_price']:.2f}</div>
                <div><strong>Risk / Reward:</strong> {row['rr_ratio']:.2f}</div>
                <div><strong>Contracts you can afford:</strong> {int(row['contracts'])}</div>
                <div><strong>Total estimated cost:</strong> ${row['position_cost']:.2f}</div>
                <div><strong>Why:</strong> {row['reason']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_table(df: pd.DataFrame):
    def color_signal(val):
        if val == "STRONG BUY":
            return "background-color: rgba(0, 200, 0, 0.18); font-weight: 700;"
        if val == "BUY":
            return "background-color: rgba(255, 193, 7, 0.22); font-weight: 700;"
        if val == "WATCH":
            return "background-color: rgba(100, 116, 139, 0.28); font-weight: 700;"
        if val == "NO TRADE":
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
                "Ask": "${:.2f}",
                "Entry": "${:.2f}",
                "Stop": "${:.2f}",
                "Target": "${:.2f}",
                "Spread %": "{:.2f}%",
                "Delta": "{:.2f}",
                "IV": "{:.2%}",
                "Score": "{:.2f}",
                "Win %": "{:.1f}%",
                "Confidence": "{:.1f}",
                "Total Cost": "${:.2f}",
                "Capital Used %": "{:.1f}%",
                "R/R": "{:.2f}",
            }
        )
    )


def show_table(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty:
        st.info("Nothing matched your filters.")
        return

    pretty = df[
        [
            "signal",
            "atm_flag",
            "symbol",
            "strike",
            "expiration_date",
            "ask",
            "entry_price",
            "stop_price",
            "target_price",
            "spread_pct",
            "volume",
            "open_interest",
            "delta",
            "iv",
            "base_score",
            "win_probability",
            "confidence",
            "contracts",
            "position_cost",
            "capital_used_pct",
            "rr_ratio",
            "reason",
        ]
    ].copy()

    pretty.columns = [
        "Signal",
        "ATM",
        "Contract",
        "Strike",
        "Expiry",
        "Ask",
        "Entry",
        "Stop",
        "Target",
        "Spread %",
        "Volume",
        "Open Interest",
        "Delta",
        "IV",
        "Score",
        "Win %",
        "Confidence",
        "Contracts",
        "Total Cost",
        "Capital Used %",
        "R/R",
        "Why",
    ]

    st.dataframe(style_table(pretty), width="stretch", hide_index=True)


def get_best_setup_for_symbol(symbol: str, capital: float, max_spread_pct: int):
    validated_ticker, is_valid = validate_ticker(symbol)
    if not is_valid:
        return None

    try:
        ticker = validated_ticker
        expirations = fetch_expirations(symbol)
        if not expirations:
            return None

        stock_ctx = get_stock_context(ticker)
        spot_price = stock_ctx["price"] if stock_ctx["price"] > 0 else get_spot_price(ticker)
        nearest_expiry = expirations[0]

        raw_chain = fetch_chain(symbol, nearest_expiry)
        chain_df = clean_chain_df(raw_chain, nearest_expiry, spot_price)

        filtered = chain_df[
            (chain_df["volume"] >= 50)
            & (chain_df["open_interest"] >= 100)
            & (chain_df["spread_pct"] <= max_spread_pct)
        ].copy()

        calls_scored = score_options(filtered, "Call", stock_ctx, capital)
        puts_scored = score_options(filtered, "Put", stock_ctx, capital)

        combined = pd.concat([calls_scored, puts_scored], ignore_index=True)
        if combined.empty:
            return None

        order_map = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "NO TRADE": 3}
        combined["signal_rank"] = combined["signal"].map(order_map)
        combined = combined.sort_values(
            ["signal_rank", "confidence", "win_probability", "base_score"],
            ascending=[True, False, False, False],
        )

        best = combined.iloc[0].copy()
        best["stock_symbol"] = symbol
        best["trend"] = stock_ctx["trend_label"]
        best["spot_price"] = spot_price
        return best

    except Exception:
        return None


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📈 Beginner Options Dashboard</div>
        <div class="hero-subtitle">A simple options scanner for one stock and a market scanner for top opportunities.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(APP_VERSION)

page1, page2 = st.tabs(["Single Stock Scanner", "Market Scanner"])

with page1:
    top_left, top_right = st.columns([1.6, 1])
    with top_left:
        raw_symbol = st.text_input("Enter stock ticker or name", "SPY")
    with top_right:
        st.markdown(
            "<div class='help-note' style='padding-top:12px;'>Examples: SPY, QQQ, MSFT, NVDA, Tesla, Microsoft, S&P500</div>",
            unsafe_allow_html=True,
        )

    raw_symbol, symbol = clean_symbol(raw_symbol)

    if raw_symbol != symbol:
        st.info(f"Using ticker symbol: {symbol}")

    st.markdown("### Basic Settings")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        top_n = st.slider("How many ideas to show", 5, 25, 10)
    with s2:
        min_volume = st.number_input("Minimum option volume", min_value=0, value=50, step=10)
    with s3:
        min_oi = st.number_input("Minimum open interest", min_value=0, value=100, step=50)
    with s4:
        max_spread_pct = st.slider("Maximum spread %", 1, 50, 15)

    st.markdown("### Money Settings")
    capital = st.number_input(
        "How much money are you investing ($)",
        min_value=100,
        value=1000,
        step=100,
    )

    validated_ticker, is_valid = validate_ticker(symbol)
    if not is_valid:
        st.error(f"Invalid ticker: {symbol}. Try SPY, QQQ, MSFT, TSLA, NVDA, or AAPL.")
        st.stop()

    try:
        ticker = validated_ticker
        expirations = fetch_expirations(symbol)
        stock_ctx = get_stock_context(ticker)
        spot_price = stock_ctx["price"] if stock_ctx["price"] > 0 else get_spot_price(ticker)
    except Exception as e:
        st.error(f"Could not load stock data for {symbol}: {e}")
        st.stop()

    if not expirations:
        st.error(f"No options found for {symbol}.")
        st.stop()

    expiry = st.selectbox("Choose expiration", expirations)

    try:
        raw_chain = fetch_chain(symbol, expiry)
        chain_df = clean_chain_df(raw_chain, expiry, spot_price)
    except Exception as e:
        st.error(f"Could not load options chain: {e}")
        st.stop()

    filtered = chain_df[
        (chain_df["volume"] >= min_volume)
        & (chain_df["open_interest"] >= min_oi)
        & (chain_df["spread_pct"] <= max_spread_pct)
    ].copy()

    calls_scored = score_options(filtered, "Call", stock_ctx, capital)
    puts_scored = score_options(filtered, "Put", stock_ctx, capital)

    calls = calls_scored.head(top_n)
    puts = puts_scored.head(top_n)
    all_scored = pd.concat([calls_scored, puts_scored], ignore_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock", symbol)
    m2.metric("Stock Price", f"${spot_price:,.2f}" if spot_price else "N/A")
    m3.metric("Trend", stock_ctx["trend_label"])
    actionable_count = int(all_scored["signal"].isin(["STRONG BUY", "BUY"]).sum()) if not all_scored.empty else 0
    m4.metric("Good Setups", actionable_count)

    rsi_value = f"{stock_ctx['rsi']:.1f}" if stock_ctx["rsi"] is not None else "N/A"
    vwap_value = f"${stock_ctx['vwap']:.2f}" if stock_ctx["vwap"] is not None else "N/A"
    rel_vol_value = f"{stock_ctx['relative_volume']:.2f}x" if stock_ctx["relative_volume"] is not None else "N/A"
    support_value = f"${stock_ctx['support']:.2f}" if stock_ctx["support"] is not None else "N/A"
    resistance_value = f"${stock_ctx['resistance']:.2f}" if stock_ctx["resistance"] is not None else "N/A"

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>RSI</div><div class='simple-value'>{rsi_value}</div></div>",
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>VWAP</div><div class='simple-value'>{vwap_value}</div></div>",
            unsafe_allow_html=True,
        )
    with i3:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>Relative Volume</div><div class='simple-value'>{rel_vol_value}</div></div>",
            unsafe_allow_html=True,
        )
    with i4:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>Expiration</div><div class='simple-value'>{expiry}</div></div>",
            unsafe_allow_html=True,
        )

    j1, j2 = st.columns(2)
    with j1:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>Support</div><div class='simple-value'>{support_value}</div></div>",
            unsafe_allow_html=True,
        )
    with j2:
        st.markdown(
            f"<div class='simple-card'><div class='simple-label'>Resistance</div><div class='simple-value'>{resistance_value}</div></div>",
            unsafe_allow_html=True,
        )

    if all_scored.empty:
        st.warning("No trades matched your filters. Try lowering your filters or switching tickers.")
    elif actionable_count == 0:
        st.warning("No strong trade right now. The dashboard thinks it may be better to wait.")
    else:
        st.success("Good — the scanner found setups with enough confirmation to review.")

    c1, c2 = st.columns(2)
    with c1:
        render_trade_card("Best Call Idea", calls.iloc[0] if not calls.empty else None)
    with c2:
        render_trade_card("Best Put Idea", puts.iloc[0] if not puts.empty else None)

    st.info(
        "Beginner guide: STRONG BUY means most important conditions agree. BUY means decent setup. WATCH means not ready yet. NO TRADE means the chart and contract do not line up well enough."
    )

    tab1, tab2, tab3 = st.tabs(["Top Calls", "Top Puts", "ATM Contracts"])

    with tab1:
        show_table(calls, f"Top Call Contracts for {symbol} {expiry}")

    with tab2:
        show_table(puts, f"Top Put Contracts for {symbol} {expiry}")

    with tab3:
        atm_df = all_scored[all_scored["atm_flag"] == "ATM"].copy()
        if atm_df.empty:
            st.info("No ATM contracts matched your filters.")
        else:
            show_table(atm_df.head(20), f"ATM Contracts for {symbol} {expiry}")

with page2:
    st.markdown("### Market Scanner")
    st.markdown(
        "<div class='help-note'>This scans a watchlist of liquid option stocks and ranks the top 5 opportunities. With free Yahoo data, this is the best practical version of a whole market scan.</div>",
        unsafe_allow_html=True,
    )

    default_watchlist = "SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, COIN, PLTR"
    watchlist_input = st.text_area("Stocks to scan", value=default_watchlist, height=90)
    scan_capital = st.number_input("Capital to size each idea ($)", min_value=100, value=1000, step=100, key="scan_capital")
    scan_spread = st.slider("Scanner max spread %", 1, 50, 15, key="scan_spread")
    run_scan = st.button("Run Market Scan")

    if run_scan:
        raw_symbols = [x.strip() for x in watchlist_input.split(",") if x.strip()]
        symbols = [clean_symbol(x)[1] for x in raw_symbols]

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, sym in enumerate(symbols):
            status.write(f"Scanning {sym}...")
            result = get_best_setup_for_symbol(sym, scan_capital, scan_spread)
            if result is not None:
                results.append(result)
            progress.progress((i + 1) / len(symbols))
            time.sleep(0.1)

        status.empty()

        if not results:
            st.warning("No strong setups were found in this watchlist.")
        else:
            market_df = pd.DataFrame(results)

            order_map = {"STRONG BUY": 0, "BUY": 1, "WATCH": 2, "NO TRADE": 3}
            market_df["signal_rank"] = market_df["signal"].map(order_map)
            market_df = market_df.sort_values(
                ["signal_rank", "confidence", "win_probability", "base_score"],
                ascending=[True, False, False, False],
            )

            top5 = market_df.head(5).copy()

            st.success("Scan complete. Here are the top 5 options stocks to review.")

            st.subheader("Top 5 Options Stocks to Buy")
            display_df = top5[
                [
                    "stock_symbol",
                    "option_type",
                    "signal",
                    "confidence",
                    "win_probability",
                    "trend",
                    "spot_price",
                    "symbol",
                    "strike",
                    "expiration_date",
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "contracts",
                    "position_cost",
                    "rr_ratio",
                    "reason",
                ]
            ].copy()

            display_df.columns = [
                "Stock",
                "Side",
                "Signal",
                "Confidence",
                "Win %",
                "Trend",
                "Stock Price",
                "Option Contract",
                "Strike",
                "Expiry",
                "Entry",
                "Stop",
                "Target",
                "Contracts",
                "Total Cost",
                "R/R",
                "Why",
            ]

            st.dataframe(display_df, width="stretch", hide_index=True)

            st.markdown("### Best Overall Setup")
            best = top5.iloc[0]
            render_trade_card("Top Market Setup", best)