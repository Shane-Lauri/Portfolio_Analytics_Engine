# stress_test_pure.py
# Pure hypothetical stress test for a stock portfolio given user-defined shocks.
# Outputs: console summary + CSV report "stress_test_report.csv" in the same folder.

import pandas as pd
from pathlib import Path

def pure_hypothetical_stress_test(prices: dict, holdings: dict, shocks: dict, default_shock: float = 0.0):
    """
    prices:   {ticker: current_price}
    holdings: {ticker: shares}
    shocks:   {ticker: shock_return} (e.g., -0.25 for -25%)
    default_shock: applied to tickers not in shocks
    """
    rows = []
    for tkr, shares in holdings.items():
        if tkr not in prices:
            rows.append([tkr, shares, None, shocks.get(tkr, default_shock), None, None, None, "MISSING_PRICE"])
            continue

        p0 = float(prices[tkr])
        v0 = p0 * float(shares)
        r  = float(shocks.get(tkr, default_shock))
        v1 = v0 * (1.0 + r)
        pnl = v1 - v0
        rows.append([tkr, shares, p0, r, v0, v1, pnl, "OK"])

    detail = pd.DataFrame(
        rows,
        columns=["Ticker", "Shares", "Price", "ShockReturn", "BaseValue", "StressedValue", "PnL", "Status"]
    )

    ok = detail[detail["Status"] == "OK"].copy()
    summary = pd.DataFrame([{
        "BaseValue": ok["BaseValue"].sum(),
        "StressedValue": ok["StressedValue"].sum(),
        "PnL": ok["PnL"].sum(),
        "PnL_pct": (ok["PnL"].sum() / ok["BaseValue"].sum()) if ok["BaseValue"].sum() else None
    }])

    return detail.sort_values("PnL"), summary

if __name__ == "__main__":
    # ---- EDIT THESE ----
    # Current prices for each ticker (you can also load these from a file)
    PRICES = {
        "NVDA": 185.41,
    }

    # Holdings in shares
    HOLDINGS = {
        "NVDA": 10,
    }

    # Shock returns (pure hypothetical). Any ticker not listed uses DEFAULT_SHOCK.
    SHOCKS = {
        "NVDA": -0.40,   # -40%
    }

    DEFAULT_SHOCK = -0.15  # e.g., market-wide -15% for everything else
    # ---------------------

    detail, summary = pure_hypothetical_stress_test(PRICES, HOLDINGS, SHOCKS, DEFAULT_SHOCK)

    print("\n--- Stress Test Summary ---")
    print(summary.to_string(index=False))

    print("\n--- Position Detail (worst first) ---")
    print(detail.to_string(index=False))

    out_path = Path(__file__).with_name("stress_test_report.csv")
    detail.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
