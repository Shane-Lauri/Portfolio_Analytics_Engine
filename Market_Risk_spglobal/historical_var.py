"""
Historical VaR (1-day) at 95% and 99% using daily_closes.py

- Uses the same portfolio definition (tickers + weights)
- Fetches adjusted daily closes
- Computes daily portfolio returns
- Historical VaR:
    VaR_95 uses the 5th percentile of historical daily returns
    VaR_99 uses the 1st percentile of historical daily returns
- Plots histogram and highlights bottom 5% and 1% cutoffs

No scipy required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from daily_closes import download_daily_closes


# ============================================================
# SAME PORTFOLIO AS BEFORE (edit if needed)
# ============================================================
TICKERS = ["JPM", "NVDA", "AAPL", "MSFT", "GOOGL"]
WEIGHTS = {"NVDA": 0.25, "AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.15, "JPM": 0.50}  # normalized inside code

PORTFOLIO_VALUE = 1_000_000.0
YEARS_BACK = 5  # <-- change this to any number of years you want
# ============================================================


def _normalize_weights(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("Weights contain non-finite values.")
    s = float(w.sum())
    if s == 0:
        raise ValueError("Weights sum to 0; provide non-zero weights.")
    return w / s


def get_portfolio_returns_from_daily_closes(
    tickers: list[str],
    weights: dict[str, float],
    years_back: int,
) -> pd.Series:
    days_back = int(years_back * 365)

    closes = download_daily_closes(tickers, days_back)
    if closes.empty:
        raise RuntimeError("No price data returned (closes is empty).")
    if "Date" not in closes.columns:
        raise RuntimeError("Expected a 'Date' column in closes output.")

    prices = closes.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for t in tickers:
        if t not in prices.columns:
            raise ValueError(f"Ticker '{t}' missing from downloaded closes columns.")
        prices[t] = pd.to_numeric(prices[t], errors="coerce")

    prices = prices[tickers].dropna(how="any")
    if len(prices) < 2:
        raise RuntimeError("Not enough price rows to compute returns (need at least 2).")

    asset_rets = prices.pct_change().dropna(how="any")

    w = _normalize_weights(weights, tickers)
    port_rets = asset_rets.to_numpy(dtype=float) @ w
    return pd.Series(port_rets, index=asset_rets.index, name="portfolio_return")


def historical_var_1day(portfolio_returns: pd.Series, portfolio_value: float) -> dict:
    r = portfolio_returns.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError("Need at least ~30 observations to compute historical VaR.")

    q05 = float(np.quantile(r, 0.05))  # 5th percentile return (VaR 95 cutoff)
    q01 = float(np.quantile(r, 0.01))  # 1st percentile return (VaR 99 cutoff)

    var_95 = max(0.0, -q05 * portfolio_value)
    var_99 = max(0.0, -q01 * portfolio_value)

    return {
        "n_obs": int(r.size),
        "q05_return": q05,
        "q01_return": q01,
        "VaR_95_$": float(var_95),
        "VaR_99_$": float(var_99),
        "VaR_95_pct": float(var_95 / portfolio_value),
        "VaR_99_pct": float(var_99 / portfolio_value),
    }


if __name__ == "__main__":
    pr = get_portfolio_returns_from_daily_closes(TICKERS, WEIGHTS, YEARS_BACK)
    res = historical_var_1day(pr, PORTFOLIO_VALUE)

    print("--------------------------------------------------")
    print(f"Historical VaR (1-day) using last {YEARS_BACK} year(s) of returns")
    print(f"Observations used : {res['n_obs']}")
    print(f"Portfolio value   : {PORTFOLIO_VALUE:,.2f}")
    print("")
    print(f"5th pct return (VaR 95 cutoff): {res['q05_return']:.6%}")
    print(f"1st pct return (VaR 99 cutoff): {res['q01_return']:.6%}")
    print("")
    print(f"VaR 95% ($): {res['VaR_95_$']:,.2f}   ({res['VaR_95_pct']*100:.2f}%)")
    print(f"VaR 99% ($): {res['VaR_99_$']:,.2f}   ({res['VaR_99_pct']*100:.2f}%)")
    print("--------------------------------------------------")

    # --- Plot histogram with bottom 5% and 1% cutoffs ---
    r = pr.to_numpy(dtype=float)
    r = r[np.isfinite(r)]

    q05 = res["q05_return"]
    q01 = res["q01_return"]

    plt.figure(figsize=(10, 6))
    plt.hist(r, bins=60, density=True, alpha=0.6, label="Historical daily portfolio returns")

    plt.axvline(q05, linestyle="--", linewidth=2, label="5th percentile (95% VaR cutoff)")
    plt.axvline(q01, linestyle="--", linewidth=2, label="1st percentile (99% VaR cutoff)")

    plt.title(f"Historical Daily Portfolio Returns (last {YEARS_BACK} year(s)) with VaR cutoffs")
    plt.xlabel("Daily portfolio return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
