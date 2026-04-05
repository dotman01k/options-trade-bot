from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class MarketSnapshot:
    symbol: str
    spot: float
    history: pd.DataFrame
    expirations: List[str]


def get_snapshot(symbol: str, period: str = "6mo", interval: str = "1d") -> MarketSnapshot:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price history returned for {symbol}.")
    expirations = list(ticker.options)
    if not expirations:
        raise ValueError(f"No options expirations found for {symbol}.")
    spot = float(hist["Close"].dropna().iloc[-1])
    return MarketSnapshot(symbol=symbol.upper(), spot=spot, history=hist, expirations=expirations)


def get_option_chain(symbol: str, expiration: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiration)

    calls = chain.calls.copy()
    calls["option_type"] = "call"

    puts = chain.puts.copy()
    puts["option_type"] = "put"

    df = pd.concat([calls, puts], ignore_index=True)
    if df.empty:
        raise ValueError(f"No option contracts found for {symbol} at {expiration}.")

    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mid"] = np.where(
        (df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0),
        (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2,
        df["lastPrice"],
    )
    df["spread"] = df["ask"].fillna(df["mid"]) - df["bid"].fillna(df["mid"])
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.nan)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_technicals(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    close = df["Close"]
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi_14"] = rsi(close, 14)
    df["atr_proxy"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["returns"] = close.pct_change()
    df["hv_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    return df
