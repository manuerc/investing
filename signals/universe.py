"""Universes.

RESEARCH is deliberately much wider than the personal watchlist: features are
screened here so that the selected indicators are not fitted to 18 names.
The watchlist is only used to APPLY the result.

Caveat: RESEARCH uses today's large caps, so it carries survivorship bias.
That inflates absolute returns; benchmark-relative metrics are far less
affected, which is one more reason the primary target is excess return.
"""

RESEARCH_STOCKS = [
    # mega/large cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
    "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM", "NOW", "INTU", "AMAT", "MU",
    # health
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "LLY", "AMGN", "GILD", "BMY",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "USB",
    # consumer / industrial / energy
    "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "PG", "HD", "LOW",
    "CAT", "DE", "BA", "HON", "UPS", "GE", "LMT", "MMM",
    "XOM", "CVX", "COP", "SLB", "OXY",
    "DIS", "NFLX", "T", "VZ", "TSLA", "UBER", "PYPL", "MELI",
    # sector / country ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "EEM", "EWZ", "QQQ", "IWM",
]

RESEARCH_CRYPTO = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
                   "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD"]

BENCH = {"stock": "SPY", "crypto": "BTC-USD"}


def research_universe() -> list[dict]:
    out = [{"alias": s, "symbol": s, "class": "stock"} for s in RESEARCH_STOCKS]
    out += [{"alias": s.replace("-USD", ""), "symbol": s, "class": "crypto"} for s in RESEARCH_CRYPTO]
    return out
