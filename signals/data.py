"""OHLCV download with a local parquet cache."""

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "idx_")
    return CACHE_DIR / f"{safe}__{start}__{end}.parquet"


def load_ohlcv(symbol: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    path = _cache_path(symbol, start, end)
    if path.exists() and not force:
        return pd.read_parquet(path)

    raw = yf.download(
        symbol, start=start, end=end, progress=False, auto_adjust=True, threads=False
    )
    if raw.empty:
        raise RuntimeError(f"[DATA] no rows returned for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])

    df.to_parquet(path)
    print(f"[DATA] {symbol:10s} {len(df):5d} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_universe(universe: list[dict], start: str, end: str) -> dict[str, pd.DataFrame]:
    out = {}
    for item in universe:
        out[item["alias"]] = load_ohlcv(item["symbol"], start, end)
    return out
