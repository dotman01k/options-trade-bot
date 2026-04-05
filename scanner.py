from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import compute_technicals, get_option_chain, get_snapshot


@dataclass
class ScanConfig:
    symbol: str
    option_side: Literal["call", "put", "both"] = "both"
    min_open_interest: int = 100
    min_volume: int = 10
    max_spread_pct: float = 0.15
    min_dte: int = 7
    max_dte: int = 45
    max_contracts: int = 20
    risk_free_rate: float = 0.04
    budget: float = 1000.0


def _days_to_exp(expiration: str) -> int:
    exp = pd.Timestamp(expiration).normalize()
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return max((exp - now).days, 0)


def _bs_delta(spot: float, strike: float, t: float, r: float, sigma: float, option_type: str) -> float:
    if min(spot, strike, t, sigma) <= 0:
        return np.nan
    d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    scaled = 100 * (s - s.min()) / (s.max() - s.min())
    return 100 - scaled if invert else scaled


def scan_best_options(config: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    snapshot = get_snapshot(config.symbol)
    tech = compute_technicals(snapshot.history)
    latest = tech.iloc[-1]

    bullish_bias = int(latest["sma_20"] > latest["sma_50"]) + int(latest["macd"] > latest["macd_signal"]) + int(latest["rsi_14"] > 50)
    bearish_bias = int(latest["sma_20"] < latest["sma_50"]) + int(latest["macd"] < latest["macd_signal"]) + int(latest["rsi_14"] < 50)
    market_bias = "bullish" if bullish_bias > bearish_bias else "bearish" if bearish_bias > bullish_bias else "neutral"

    frames: list[pd.DataFrame] = []
    for expiration in snapshot.expirations:
        dte = _days_to_exp(expiration)
        if dte < config.min_dte or dte > config.max_dte:
            continue
        try:
            chain = get_option_chain(config.symbol, expiration)
        except Exception:
            continue
        chain["expiration"] = expiration
        chain["dte"] = dte
        frames.append(chain)

    if not frames:
        raise ValueError("No option chains matched the chosen DTE window.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["openInterest"].fillna(0) >= config.min_open_interest]
    df = df[df["volume"].fillna(0) >= config.min_volume]
    df = df[df["mid"].fillna(0) > 0]
    df = df[df["spread_pct"].fillna(1) <= config.max_spread_pct]

    if config.option_side != "both":
        df = df[df["option_type"] == config.option_side]

    if df.empty:
        raise ValueError("No contracts passed your liquidity and spread filters.")

    spot = snapshot.spot
    df["moneyness_pct"] = (df["strike"] - spot) / spot
    df["distance_abs"] = df["moneyness_pct"].abs()
    df["premium"] = df["mid"] * 100
    df["contracts_affordable"] = (config.budget // df["premium"]).clip(lower=0)
    df["time_years"] = df["dte"] / 365
    df["delta"] = df.apply(
        lambda row: _bs_delta(
            spot=spot,
            strike=float(row["strike"]),
            t=max(float(row["time_years"]), 1e-6),
            r=config.risk_free_rate,
            sigma=max(float(row["impliedVolatility"] or 0), 1e-6),
            option_type=row["option_type"],
        ),
        axis=1,
    )
    df["notional_move_needed_pct"] = (df["mid"] / spot) * 100

    preferred_type = "call" if market_bias == "bullish" else "put" if market_bias == "bearish" else None
    df["bias_fit"] = np.where(df["option_type"] == preferred_type, 100, 55 if preferred_type else 70)
    df["liquidity_score"] = 0.6 * _normalize(np.log1p(df["openInterest"])) + 0.4 * _normalize(np.log1p(df["volume"]))
    df["spread_score"] = _normalize(df["spread_pct"].fillna(df["spread_pct"].max()), invert=True)
    df["proximity_score"] = _normalize(df["distance_abs"], invert=True)
    df["affordability_score"] = _normalize(df["premium"], invert=True)
    df["iv_score"] = _normalize(df["impliedVolatility"].fillna(df["impliedVolatility"].median()), invert=True)
    df["delta_fit_score"] = _normalize((df["delta"].abs() - 0.35).abs(), invert=True)

    df["trade_score"] = (
        0.24 * df["liquidity_score"]
        + 0.20 * df["spread_score"]
        + 0.14 * df["proximity_score"]
        + 0.12 * df["affordability_score"]
        + 0.10 * df["iv_score"]
        + 0.10 * df["delta_fit_score"]
        + 0.10 * df["bias_fit"]
    ).round(2)

    df["thesis"] = np.select(
        [
            (df["option_type"] == "call") & (market_bias == "bullish"),
            (df["option_type"] == "put") & (market_bias == "bearish"),
            df["option_type"] == "call",
        ],
        [
            "Momentum supports upside continuation",
            "Trend supports downside continuation",
            "Contrarian upside candidate",
        ],
        default="Contrarian downside candidate",
    )

    cols = [
        "contractSymbol",
        "option_type",
        "expiration",
        "dte",
        "strike",
        "mid",
        "premium",
        "bid",
        "ask",
        "spread_pct",
        "volume",
        "openInterest",
        "impliedVolatility",
        "delta",
        "contracts_affordable",
        "trade_score",
        "thesis",
    ]

    ranked = df.sort_values(["trade_score", "openInterest", "volume"], ascending=[False, False, False])[cols].head(config.max_contracts)

    summary = {
        "symbol": config.symbol.upper(),
        "spot": round(spot, 2),
        "market_bias": market_bias,
        "rsi_14": round(float(latest["rsi_14"]), 2) if pd.notna(latest["rsi_14"]) else None,
        "sma_20": round(float(latest["sma_20"]), 2) if pd.notna(latest["sma_20"]) else None,
        "sma_50": round(float(latest["sma_50"]), 2) if pd.notna(latest["sma_50"]) else None,
        "macd": round(float(latest["macd"]), 4) if pd.notna(latest["macd"]) else None,
        "macd_signal": round(float(latest["macd_signal"]), 4) if pd.notna(latest["macd_signal"]) else None,
        "hv_20": round(float(latest["hv_20"]), 4) if pd.notna(latest["hv_20"]) else None,
    }
    return ranked, tech.reset_index(), summary
