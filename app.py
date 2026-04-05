import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Options + Crypto Trade Dashboard", page_icon="📈", layout="wide")

APP_VERSION = "v8 options + crypto scanner"

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
        min-height: 280px;
    }
    .signal-card-watch {
        background: #3b2f00;
        border: 1px solid #facc15;
        border-radius: 12px;
        padding: 14px;
        min-height: 280px;
    }
    .signal-card-avoid {
        background: #3b0a0a;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 14px;
        min-height: 280px;
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


def calculate_position_size(capital: float, unit_price: float, multiplier: int = 100):
    if unit_price <= 0:
        return 0
    return int(capital // (unit_price * multiplier))


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


def clean_crypto_symbol(user_input: str):
    crypto_map = {
        "BITCOIN": "BTC-USD",
        "BTC": "BTC-USD",
        "BTC-USD": "BTC-USD",
        "ETHEREUM": "ETH-USD",
        "ETH": "ETH-USD",
        "ETH-USD": "ETH-USD",
        "SOLANA": "SOL-USD",
        "SOL": "SOL-USD",
        "SOL-USD": "SOL-USD",
        "DOGE": "DOGE-USD",
        "DOGECOIN": "DOGE-USD",
        "DOGE-USD": "DOGE-USD",
        "XRP": "XRP-USD",
        "XRP-USD": "XRP-USD",
        "CARDANO": "ADA-USD",
        "ADA": "ADA-USD",
        "ADA-USD": "ADA-USD",
    }
    raw = (user_input or "").upper().strip()
    cleaned = raw.replace("$", "").strip()
    symbol = crypto_map.get(cleaned, cleaned)
    return raw, symbol


def validate_ticker(symbol: str):
    ticker = yf.Ticker(symbol)
    test = ticker.history(period="5d")
    if test.empty:
        return None, False
    return ticker, True


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


def get_trend_context(ticker):
    hist = ticker.history(period="20d", interval="1h")
    if hist.empty or "Close" not in hist.columns:
        return {
            "trend_strength": 50.0,
            "trend_label": "Neutral",
            "ema9": None,
            "ema21": None,
            "support": None,
            "resistance": None,
        }

    close = hist["Close"].dropna()
    low = hist["Low"].dropna() if "Low" in hist.columns else pd.Series(dtype=float)
    high = hist["High"].dropna() if "High" in hist.columns else pd.Series(dtype=float)

    if len(close) < 21:
        return {
            "trend_strength": 50.0,
            "trend_label": "Neutral",
            "ema9": None,
            "ema21": None,
            "support": None,
            "resistance": None,
        }

    ema9 = close.ewm(span=9).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]

    if ema9 > ema21:
        trend_strength = 80.0
        trend_label = "Bullish"
    elif ema9 < ema21:
        trend_strength = 20.0
        trend_label = "Bearish"
    else:
        trend_strength = 50.0
        trend_label = "Neutral"

    support = low.tail(20).min() if not low.empty else None
    resistance = high.tail(20).max() if not high.empty else None

    return {
        "trend_strength": trend_strength,
        "trend_label": trend_label,
        "ema9": float(ema9),
        "ema21": float(ema21),
        "support": float(support) if support is not None else None,
        "resistance": float(resistance) if resistance is not None else None,
    }


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


def add_trade_plan(side: pd.DataFrame):
    side["entry_price"] = side["ask"].round(2)
    side["stop_price"] = (side["ask"] * 0.75).round(2)
    side["target_price"] = (side["ask"] * 1.40).round(2)

    risk_per_contract = (side["entry_price"] - side["stop_price"]) * 100
    reward_per_contract = (side["target_price"] - side["entry_price"]) * 100

    side["risk_per_contract"] = risk_per_contract.round(2)
    side["reward_per_contract"] = reward_per_contract.round(2)
    side["rr_ratio"] = np.where(
        side["risk_per_contract"] > 0,
        side["reward_per_contract"] / side["risk_per_contract"],
        0,
    ).round(2)

    return side


def score_options(df: pd.DataFrame, option_type: str, trend_strength: float, trend_label: str, capital: float):
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

    trend_ok = ((option_type == "Call") and (trend_label == "Bullish")) or ((option_type == "Put") and (trend_label == "Bearish"))
    side["trend_ok"] = trend_ok

    side["signal"] = np.select(
        [
            (side["confidence"] >= 78) & (side["trend_ok"]) & (side["spread_pct"] <= 12),
            (side["confidence"] >= 62) & (side["spread_pct"] <= 18),
        ],
        ["STRONG BUY", "BUY"],
        default="AVOID",
    )

    side["contracts"] = side["ask"].apply(lambda x: calculate_position_size(capital, x, 100))
    side["position_cost"] = (side["contracts"] * side["ask"] * 100).round(2)
    side["capital_used_pct"] = np.where(capital > 0, (side["position_cost"] / capital) * 100, 0).round(1)

    side = add_trade_plan(side)

    side["reason"] = np.select(
        [
            side["signal"] == "STRONG BUY",
            side["signal"] == "BUY",
        ],
        [
            "Trend aligned, strong confidence, tight spread, good liquidity",
            "Decent setup, but weaker than top tier",
        ],
        default="Weak setup or trend not aligned",
    )

    signal_order = {"STRONG BUY": 0, "BUY": 1, "AVOID": 2}
    side["signal_rank"] = side["signal"].map(signal_order)
    side = side.sort_values(
        ["signal_rank", "confidence", "score", "volume", "open_interest"],
        ascending=[True, False, False, False, False],
    )
    return side.drop(columns=["signal_rank"])


def fetch_crypto_history(symbol: str):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="30d", interval="1h")
    return ticker, hist


def score_crypto(hist: pd.DataFrame, symbol: str, capital: float):
    if hist.empty or "Close" not in hist.columns:
        return None, []

    df = hist.copy()
    df["ema9"] = df["Close"].ewm(span=9).mean()
    df["ema21"] = df["Close"].ewm(span=21).mean()
    df["ret_1"] = df["Close"].pct_change().fillna(0)
    df["volatility"] = df["ret_1"].rolling(20).std().fillna(0)

    last = df.iloc[-1]
    price = float(last["Close"])
    ema9 = float(last["ema9"])
    ema21 = float(last["ema21"])
    volume = float(last["Volume"]) if "Volume" in df.columns else 0.0

    trend_label = "Bullish" if ema9 > ema21 else "Bearish" if ema9 < ema21 else "Neutral"
    trend_strength = 80 if ema9 > ema21 else 20 if ema9 < ema21 else 50

    momentum_score = 80 if price > ema9 > ema21 else 20 if price < ema9 < ema21 else 50
    volatility_score = 100 - min(float(last["volatility"]) * 4000, 100)
    volume_score = min((volume / max(df["Volume"].tail(20).mean(), 1)) * 50, 100) if "Volume" in df.columns else 50

    confidence = round((trend_strength * 0.4) + (momentum_score * 0.35) + (volatility_score * 0.10) + (volume_score * 0.15), 1)

    if confidence >= 78 and trend_label == "Bullish":
        signal = "STRONG BUY"
    elif confidence >= 62 and trend_label in ["Bullish", "Neutral"]:
        signal = "BUY"
    else:
        signal = "AVOID"

    entry_price = round(price, 2)
    stop_price = round(price * 0.96, 2)
    target_price = round(price * 1.08, 2)

    risk_per_unit = round(entry_price - stop_price, 2)
    reward_per_unit = round(target_price - entry_price, 2)
    rr_ratio = round(reward_per_unit / risk_per_unit, 2) if risk_per_unit > 0 else 0

    units = round(capital / entry_price, 6) if entry_price > 0 else 0
    position_cost = round(units * entry_price, 2)

    support = round(df["Low"].tail(20).min(), 2) if "Low" in df.columns else None
    resistance = round(df["High"].tail(20).max(), 2) if "High" in df.columns else None

    reason = (
        "Bullish crypto trend with improving momentum"
        if signal == "STRONG BUY"
        else "Moderate crypto setup, watch risk"
        if signal == "BUY"
        else "Weak trend or low confidence"
    )

    result = {
        "symbol": symbol,
        "price": entry_price,
        "trend_label": trend_label,
        "confidence": confidence,
        "signal": signal,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_unit": risk_per_unit,
        "reward_per_unit": reward_per_unit,
        "rr_ratio": rr_ratio,
        "units": units,
        "position_cost": position_cost,
        "support": support,
        "resistance": resistance,
        "reason": reason,
        "volume": volume,
    }

    return result, df.tail(50)


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


def render_option_card(title: str, row):
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
                <div><strong>Entry:</strong> ${row['entry_price']:.2f}</div>
                <div><strong>Stop:</strong> ${row['stop_price']:.2f}</div>
                <div><strong>Target:</strong> ${row['target_price']:.2f}</div>
                <div><strong>R/R:</strong> {row['rr_ratio']:.2f}</div>
                <div><strong>Contracts:</strong> {int(row['contracts'])}</div>
                <div><strong>Total Cost:</strong> ${row['position_cost']:.2f}</div>
                <div><strong>Reason:</strong> {row['reason']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_crypto_card(title: str, row):
    if row is None:
        st.info(f"No {title.lower()} available.")
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
                <div><strong>Trend:</strong> {row['trend_label']}</div>
                <div><strong>Confidence:</strong> {row['confidence']:.1f}</div>
                <div><strong>Entry:</strong> ${row['entry_price']:.2f}</div>
                <div><strong>Stop:</strong> ${row['stop_price']:.2f}</div>
                <div><strong>Target:</strong> ${row['target_price']:.2f}</div>
                <div><strong>R/R:</strong> {row['rr_ratio']:.2f}</div>
                <div><strong>Units:</strong> {row['units']}</div>
                <div><strong>Total Cost:</strong> ${row['position_cost']:.2f}</div>
                <div><strong>Reason:</strong> {row['reason']}</div>
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
                "Entry": "${:.2f}",
                "Stop": "${:.2f}",
                "Target": "${:.2f}",
                "Spread %": "{:.2f}%",
                "Delta": "{:.2f}",
                "IV": "{:.2%}",
                "Distance %": "{:.2f}%",
                "Score": "{:.2f}",
                "Win %": "{:.1f}%",
                "Confidence": "{:.1f}",
                "Total Cost": "${:.2f}",
                "Capital Used %": "{:.1f}%",
                "Risk/Contract": "${:.2f}",
                "Reward/Contract": "${:.2f}",
                "R/R": "{:.2f}",
                "Est. Cost": "${:.2f}",
            }
        )
    )


def show_option_table(df: pd.DataFrame, title: str):
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
            "entry_price",
            "stop_price",
            "target_price",
            "spread_pct",
            "volume",
            "open_interest",
            "delta",
            "iv",
            "distance_from_spot_pct",
            "score",
            "win_probability",
            "confidence",
            "contracts",
            "position_cost",
            "capital_used_pct",
            "risk_per_contract",
            "reward_per_contract",
            "rr_ratio",
            "reason",
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
        "Entry",
        "Stop",
        "Target",
        "Spread %",
        "Volume",
        "Open Interest",
        "Delta",
        "IV",
        "Distance %",
        "Score",
        "Win %",
        "Confidence",
        "Contracts",
        "Total Cost",
        "Capital Used %",
        "Risk/Contract",
        "Reward/Contract",
        "R/R",
        "Reason",
        "Est. Cost",
    ]

    st.dataframe(style_signals(pretty), width="stretch", hide_index=True)


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📈 Options + Crypto Trade Dashboard</div>
        <div class="hero-subtitle">One scanner for options setups and crypto spot setups.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(APP_VERSION)

main_tab1, main_tab2 = st.tabs(["Options Scanner", "Crypto Scanner"])

with main_tab1:
    left_top, right_top = st.columns([1.6, 1])
    with left_top:
        raw_symbol = st.text_input("Enter ticker (e.g. MSFT, TSLA, SPY)", "SPY", key="opt_symbol")
    with right_top:
        st.markdown(
            "<div class='small-note' style='padding-top: 12px;'>Tip: type ticker symbols or common names like Microsoft, Tesla, or S&P500.</div>",
            unsafe_allow_html=True,
        )

    raw_symbol, symbol = clean_symbol(raw_symbol)

    if raw_symbol != symbol:
        st.info(f"Using ticker symbol: {symbol}")

    st.markdown("### Scanner Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        top_n = st.slider("Top contracts", 5, 25, 10, key="opt_top_n")
    with f2:
        min_volume = st.number_input("Minimum volume", min_value=0, value=50, step=10, key="opt_min_volume")
    with f3:
        min_oi = st.number_input("Minimum open interest", min_value=0, value=100, step=50, key="opt_min_oi")
    with f4:
        min_confidence = st.slider("Minimum confidence", 0, 100, 60, key="opt_min_conf")

    max_spread_pct = st.slider("Maximum spread %", 1, 50, 15, key="opt_max_spread")

    st.markdown("### 💰 Position Settings")
    capital = st.number_input(
        "How much are you investing ($)",
        min_value=100,
        value=1000,
        step=100,
        key="opt_capital",
    )

    validated_ticker, is_valid = validate_ticker(symbol)
    if not is_valid:
        st.error(f"Invalid ticker: {symbol}. Try symbols like MSFT, TSLA, NVDA, SPY, or QQQ.")
    else:
        try:
            ticker, expirations = fetch_data(symbol)
            underlying_price = get_underlying_price(validated_ticker)
            trend = get_trend_context(validated_ticker)
            trend_strength = trend["trend_strength"]
            trend_label = trend["trend_label"]
        except Exception as e:
            st.error(f"Could not load market data for {symbol}: {e}")
            st.stop()

        if not expirations:
            st.error(f"No options found for ticker: {symbol}")
        else:
            expiry = st.selectbox("Expiration", expirations, key="opt_expiry")

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

            calls_scored = score_options(filtered, "Call", trend_strength, trend_label, capital)
            puts_scored = score_options(filtered, "Put", trend_strength, trend_label, capital)

            calls_scored = calls_scored[calls_scored["confidence"] >= min_confidence]
            puts_scored = puts_scored[puts_scored["confidence"] >= min_confidence]

            calls = calls_scored.head(top_n)
            puts = puts_scored.head(top_n)

            scored_all = pd.concat([calls_scored, puts_scored], ignore_index=True)

            m1, m2, m3, m4 = st.columns(4)
            actionable_count = int(scored_all["signal"].isin(["STRONG BUY", "BUY"]).sum()) if not scored_all.empty else 0
            m1.metric("Underlying", symbol)
            m2.metric("Spot Price", f"${underlying_price:,.2f}" if underlying_price else "N/A")
            m3.metric("Trend Bias", trend_label)
            m4.metric("Actionable Signals", f"{actionable_count}")

            i1, i2, i3 = st.columns(3)
            with i1:
                support_txt = f"${trend['support']:.2f}" if trend["support"] is not None else "N/A"
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Support</div><div class='info-value'>{support_txt}</div></div>",
                    unsafe_allow_html=True,
                )
            with i2:
                resistance_txt = f"${trend['resistance']:.2f}" if trend["resistance"] is not None else "N/A"
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Resistance</div><div class='info-value'>{resistance_txt}</div></div>",
                    unsafe_allow_html=True,
                )
            with i3:
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Selected Expiration</div><div class='info-value'>{expiry}</div></div>",
                    unsafe_allow_html=True,
                )

            if scored_all.empty or actionable_count == 0:
                st.warning("⚠️ No strong options setup right now. Best move may be no trade.")
            else:
                st.success("✅ Tradeable options setups found based on your filters and current trend.")

            c1, c2 = st.columns(2)
            with c1:
                render_option_card("Best Call", calls.iloc[0] if not calls.empty else None)
            with c2:
                render_option_card("Best Put", puts.iloc[0] if not puts.empty else None)

            st.info(
                "Trend confirmation checks whether calls align with a bullish trend and puts align with a bearish trend. Entry, stop, and target are model-based planning levels, not guarantees."
            )

            tab1, tab2, tab3 = st.tabs(["Top Calls", "Top Puts", "ATM Snapshot"])

            with tab1:
                show_option_table(calls, f"Top Call Contracts for {symbol} {expiry}")

            with tab2:
                show_option_table(puts, f"Top Put Contracts for {symbol} {expiry}")

            with tab3:
                atm_df = scored_all[scored_all["atm_flag"] == "ATM"].copy()
                if atm_df.empty:
                    st.info("No ATM contracts matched the current filters.")
                else:
                    show_option_table(atm_df.head(20), f"ATM Contracts for {symbol} {expiry}")

with main_tab2:
    left_top, right_top = st.columns([1.6, 1])
    with left_top:
        raw_crypto = st.text_input("Enter crypto (e.g. BTC, ETH, SOL, BTC-USD)", "BTC-USD", key="crypto_symbol")
    with right_top:
        st.markdown(
            "<div class='small-note' style='padding-top: 12px;'>Tip: you can type Bitcoin, BTC, Ethereum, ETH, Solana, or SOL.</div>",
            unsafe_allow_html=True,
        )

    raw_crypto, crypto_symbol = clean_crypto_symbol(raw_crypto)

    if raw_crypto != crypto_symbol:
        st.info(f"Using crypto symbol: {crypto_symbol}")

    st.markdown("### Crypto Settings")
    cset1, cset2 = st.columns(2)
    with cset1:
        crypto_capital = st.number_input(
            "How much are you investing in crypto ($)",
            min_value=50,
            value=1000,
            step=50,
            key="crypto_capital",
        )
    with cset2:
        crypto_min_conf = st.slider("Minimum crypto confidence", 0, 100, 60, key="crypto_min_conf")

    crypto_ticker, crypto_valid = validate_ticker(crypto_symbol)
    if not crypto_valid:
        st.error(f"Invalid crypto symbol: {crypto_symbol}. Try BTC-USD, ETH-USD, SOL-USD, or ADA-USD.")
    else:
        try:
            crypto_ticker, crypto_hist = fetch_crypto_history(crypto_symbol)
            crypto_result, crypto_recent = score_crypto(crypto_hist, crypto_symbol, crypto_capital)
        except Exception as e:
            st.error(f"Could not load crypto data: {e}")
            st.stop()

        if crypto_result is None:
            st.warning("No crypto data available right now.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Crypto", crypto_symbol)
            m2.metric("Spot Price", f"${crypto_result['price']:,.2f}")
            m3.metric("Trend Bias", crypto_result["trend_label"])
            m4.metric("Signal", crypto_result["signal"])

            i1, i2, i3 = st.columns(3)
            with i1:
                support_txt = f"${crypto_result['support']:,.2f}" if crypto_result["support"] is not None else "N/A"
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Support</div><div class='info-value'>{support_txt}</div></div>",
                    unsafe_allow_html=True,
                )
            with i2:
                resistance_txt = f"${crypto_result['resistance']:,.2f}" if crypto_result["resistance"] is not None else "N/A"
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Resistance</div><div class='info-value'>{resistance_txt}</div></div>",
                    unsafe_allow_html=True,
                )
            with i3:
                st.markdown(
                    f"<div class='info-card'><div class='info-label'>Confidence</div><div class='info-value'>{crypto_result['confidence']:.1f}</div></div>",
                    unsafe_allow_html=True,
                )

            if crypto_result["confidence"] < crypto_min_conf or crypto_result["signal"] == "AVOID":
                st.warning("⚠️ No strong crypto setup right now. Best move may be no trade.")
            else:
                st.success("✅ Tradeable crypto setup found.")

            render_crypto_card("Best Crypto Setup", crypto_result)

            st.subheader("Recent Crypto Prices")
            if crypto_recent is not None and not crypto_recent.empty:
                chart_df = crypto_recent[["Close"]].copy()
                st.line_chart(chart_df, height=300)