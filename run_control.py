"""Survivorship-bias control.

The mean-reversion result ("buy deep drawdowns") is exactly the finding that a
survivor-selected universe would fabricate: today's large caps are the ones
whose -40% drawdowns recovered. The ones that did not are delisted and absent.

Sector/index ETFs do not disappear, so they are a clean control group.
If mean reversion survives on ETFs, it is real. If it vanishes, it was bias.
"""

import warnings

import numpy as np
import pandas as pd

from run_screen import build_panel, screen

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "EEM", "EWZ", "QQQ", "IWM"]
FOCUS = ["drawdown_252", "dist_52w_high", "rsi14", "dist_ema200_atr", "px_ema200",
         "atr_pct", "bb_bandwidth", "rs_252", "ret_252", "mom_12_1", "ema50_ema200",
         "bench_px_ema200", "dist_52w_low", "zscore_20", "rsi2"]

COLS = ["feature", "n", "ic_pooled", "ic_ts", "ic_xs", "ic_is", "ic_oos", "ts_consist", "yr_consist"]


def main():
    panel = build_panel()
    panel["date"] = pd.to_datetime(panel["date"])

    groups = {
        "ETFs (sin survivorship bias)": panel[panel["asset"].isin(ETFS)],
        "Acciones individuales (con bias)": panel[(panel["class"] == "stock") & (~panel["asset"].isin(ETFS))],
        "Cripto": panel[panel["class"] == "crypto"],
    }

    for target in ("fwd_ret_63", "fwd_excess_63"):
        print("\n" + "=" * 140)
        print(f"  CONTROL — target {target}")
        print("=" * 140)
        tables = {}
        for name, sub in groups.items():
            res = screen(sub, target, min_rows=2000)
            tables[name] = res.set_index("feature")

        rows = []
        for f in FOCUS:
            row = {"feature": f}
            for name, tb in tables.items():
                if f in tb.index:
                    row[f"{name.split()[0]}_ic"] = tb.loc[f, "ic_pooled"]
                    row[f"{name.split()[0]}_ts"] = tb.loc[f, "ic_ts"]
                    row[f"{name.split()[0]}_yr"] = tb.loc[f, "yr_consist"]
            rows.append(row)
        out = pd.DataFrame(rows).set_index("feature")
        print(out.to_string(float_format=lambda z: f"{z:,.4f}"))


if __name__ == "__main__":
    main()
