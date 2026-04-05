import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live Options Scanner", page_icon="📈", layout="wide")


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


def clean_df(df, expiration):
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

    midpoint = df["strike"].median() if not df.empty else 0
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

    target = 0.35 if option_type == "Call" else -0.35
    delta_fit = 100 - (abs(side["delta"] - target) / 0.35 * 100).clip(0, 100)

    side["score"] = (
        liquidity * 0.4
        + spread * 0.3
        + delta_fit * 0.3
    ).round(2)

    return side.sort_values(["score", "volume", "open_interest"], ascending=[False, False, False])


st.title("📈 Options Scanner")
st.caption("Yahoo Finance version. No API key required.")

symbol = st.text_input("Ticker", "SPY").upper().strip()
top_n = st.slider("Top contracts", 5, 25, 10)
min_volume = st.number_input("Minimum volume", min_value=0, value=50, step=10)
min_oi = st.number_input("Minimum open interest", min_value=0, value=100, step=50)
max_spread_pct = st.slider("Maximum spread %", 1, 50, 15)

try:
    ticker, expirations = fetch_data(symbol)
except Exception as e:
    st.error(f"Could not load market data for {symbol}: {e}")
    st.stop()

if not expirations:
    st.error("No options found for this ticker.")
    st.stop()

expiry = st.selectbox("Expiration", expirations)

try:
    chain = fetch_chain(symbol, expiry)
    df = clean_df(chain, expiry)
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Best Call")
    if not calls.empty:
        top = calls.iloc[0]
        st.success(f"{top['symbol']} | Score {top['score']:.2f} | Ask ${top['ask']:.2f}")
    else:
        st.info("No call matched your filters.")

with col2:
    st.subheader("Best Put")
    if not puts.empty:
        top = puts.iloc[0]
        st.success(f"{top['symbol']} | Score {top['score']:.2f} | Ask ${top['ask']:.2f}")
    else:
        st.info("No put matched your filters.")

st.subheader("Top Calls")
if calls.empty:
    st.info("No call contracts matched your filters.")
else:
    st.dataframe(calls, width="stretch", hide_index=True)

st.subheader("Top Puts")
if puts.empty:
    st.info("No put contracts matched your filters.")
else:
    st.dataframe(puts, width="stretch", hide_index=True)
