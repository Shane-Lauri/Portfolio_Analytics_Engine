# factor_historical_stress_test.py
#
# Factor-level historical stress test:
# - Estimates CURRENT betas of each asset to factors (SPY/SMH/TLT/VXX proxies)
# - Replays a REAL historical factor move over a chosen window [STRESS_START, STRESS_END]
# - Computes implied asset returns and portfolio PnL for today's holdings
#
# Data source:
# - Uses your existing daily_adjusted_closes.csv (created by daily_closes.py)
# - If required tickers are missing from the CSV, it will download them via download_daily_closes()
#
# Outputs:
# - factor_historical_betas.csv
# - factor_historical_stress_results.csv

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

# IMPORTANT: uses your working downloader (with SSL fix) to top-up missing tickers in the CSV
from daily_closes import download_daily_closes

# ---------------------------- SSL / ENV ----------------------------
# Keep this so any imports that rely on requests/SSL use the certifi bundle.
CERT_PATH = certifi.where()
os.environ["SSL_CERT_FILE"] = CERT_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CERT_PATH
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("CURL_SSL_BACKEND", None)

# ---------------------------- CONFIG ----------------------------

BASE_DIR = Path(r"C:\Users\franc\Downloads\VSC\AdSynthAI\Market_Risk_spglobal")
CLOSES_CSV = BASE_DIR / "daily_adjusted_closes.csv"

# Assets and holdings
ASSETS = ["NVDA", "AAPL"]
HOLDINGS_SHARES = {"NVDA": 10, "AAPL": 20}

# Factors: committed set
FACTORS = {
    "MKT": "SPY",
    "SEMI": "SMH",
    "RATES": "TLT",
    "VOL": "VXX",
}

# Beta estimation
MIN_OBS = 120
USE_ROLLING = True
ROLLING_WINDOW = 252  # 1Y
USE_ORTHOG = True
USE_SHRINKAGE = True
SHRINK_LAMBDA = 0.70
BETA_PRIOR = {"MKT": 1.0, "SEMI": 0.0, "RATES": 0.0, "VOL": 0.0}

# Historical stress window (inclusive dates)
# Choose any past episode you want that exists in the data.
# Examples:
# - COVID crash: 2020-02-19 to 2020-03-23
# - 2022 tech/rates shock (illustrative): 2022-01-03 to 2022-10-12
STRESS_START = "2020-02-19"
STRESS_END = "2020-03-23"

# Replay mode:
# - "cumulative": apply total move over the window as a single shock (most common for reporting)
# - "path": compute a daily PnL path over the window (useful for drawdowns)
REPLAY_MODE = "cumulative"  # or "path"

DAYS_BACK_NEEDED = 365 * 15  # ~15 years of calendar days

# Outputs
BETAS_OUT = BASE_DIR / "factor_historical_betas.csv"
RESULTS_OUT = BASE_DIR / "factor_historical_stress_results.csv"

# -------------------------- END CONFIG --------------------------


def ensure_closes_csv_has_columns(csv_path: Path, required_tickers: list[str], days_back: int, required_start: str) -> None:
    """
    Ensures csv_path exists, contains required_tickers, AND extends history back to required_start.
    If columns are missing OR history is too short, downloads data and merges by Date.
    """
    required_start_dt = pd.to_datetime(required_start)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            raise ValueError(f"{csv_path.name} exists but has no 'Date' column.")
    else:
        df = pd.DataFrame(columns=["Date"])

    # Normalize Date
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        current_min = df["Date"].min()
    else:
        current_min = None

    missing_cols = [t for t in required_tickers if t not in df.columns]

    # Determine if we need to extend history
    need_extend = (current_min is None) or (current_min > required_start_dt)

    # If any columns missing OR history doesn't reach required_start, download
    if not missing_cols and not need_extend:
        return

    if need_extend:
        print(f"CSV starts at {current_min.date() if current_min is not None else 'N/A'}; need <= {required_start_dt.date()}. Extending history...")
        tickers_to_download = required_tickers  # re-download all required tickers to extend history
    else:
        print(f"CSV missing columns {missing_cols}. Downloading and merging...")
        tickers_to_download = missing_cols

    new_df = download_daily_closes(tickers_to_download, days_back)
    if new_df.empty:
        raise RuntimeError(f"Failed to download tickers: {tickers_to_download}")

    # Normalize new_df Date
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
    new_df = new_df.dropna(subset=["Date"])

    # Merge
    if df.empty or df.shape[1] == 1:
        merged = new_df
    else:
        merged = pd.merge(df, new_df, on="Date", how="outer")

    merged = merged.sort_values("Date")
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")

    try:
        merged.to_csv(csv_path, index=False)
        print(f"Updated file saved: {csv_path}")
    except PermissionError:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = csv_path.with_name(f"{csv_path.stem}_{ts}{csv_path.suffix}")
        merged.to_csv(fallback, index=False)
        print(f"Could not overwrite (file likely open). Saved instead: {fallback}")



def load_prices(csv_path: Path, tickers: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a Date column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df[tickers].astype(float)


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def ols_betas(y: pd.Series, X: pd.DataFrame, min_obs: int) -> tuple[pd.Series, float]:
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < min_obs:
        raise ValueError(f"Not enough overlapping observations: {df.shape[0]} < {min_obs}")

    yv = df.iloc[:, 0].to_numpy(dtype=float)
    Xv = df.iloc[:, 1:].to_numpy(dtype=float)

    # add intercept
    Xv = np.column_stack([np.ones(len(yv)), Xv])

    coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
    betas = pd.Series(coef[1:], index=X.columns, dtype=float)

    yhat = Xv @ coef
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return betas, r2


def shrink_betas(betas: pd.Series, prior: dict, lam: float) -> pd.Series:
    prior_vec = pd.Series({k: float(prior.get(k, 0.0)) for k in betas.index}, dtype=float)
    return lam * betas + (1.0 - lam) * prior_vec


def orthogonalize_factors(factor_rets: pd.DataFrame) -> pd.DataFrame:
    """
    Make SEMI orthogonal to MKT: SEMI <- SEMI - gamma*MKT
    This reduces overlap/double counting between SPY and SMH.
    """
    fr = factor_rets.copy()
    if "MKT" in fr.columns and "SEMI" in fr.columns:
        m = fr["MKT"].dropna()
        s = fr["SEMI"].dropna()
        idx = m.index.intersection(s.index)
        if len(idx) >= 30:
            m = m.loc[idx].to_numpy(dtype=float)
            s = s.loc[idx].to_numpy(dtype=float)
            var_m = float(np.var(m))
            if var_m > 0:
                gamma = float(np.cov(s, m, ddof=1)[0, 1] / var_m)
                fr["SEMI"] = fr["SEMI"] - gamma * fr["MKT"]
    return fr


def factor_shock_cumulative(factor_prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """
    Returns cumulative factor move over [start, end]:
      shock_k = (P_end / P_start) - 1
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    fp = factor_prices.loc[(factor_prices.index >= s) & (factor_prices.index <= e)].dropna(how="any")
    if fp.empty or len(fp) < 2:
        raise ValueError("Not enough factor price data in the specified stress window.")

    p0 = fp.iloc[0]
    p1 = fp.iloc[-1]
    return (p1 / p0) - 1.0


def main():
    required = ASSETS + list(FACTORS.values())
    ensure_closes_csv_has_columns(CLOSES_CSV, required, DAYS_BACK_NEEDED, required_start=STRESS_START)


    # Load asset + factor prices from same CSV
    all_tickers = ASSETS + list(FACTORS.values())
    prices_all = load_prices(CLOSES_CSV, all_tickers)

    asset_prices = prices_all[ASSETS]
    factor_prices_raw = prices_all[list(FACTORS.values())].copy()
    factor_prices_raw = factor_prices_raw.rename(columns={v: k for k, v in FACTORS.items()})

    # Compute returns
    asset_rets = pct_returns(asset_prices)
    factor_rets = pct_returns(factor_prices_raw)

    # Align on dates
    common = asset_rets.index.intersection(factor_rets.index)
    asset_rets = asset_rets.loc[common]
    factor_rets = factor_rets.loc[common]

    # Orthogonalize factor returns if enabled
    if USE_ORTHOG:
        factor_rets = orthogonalize_factors(factor_rets)

    # Estimate CURRENT betas using latest rolling window (or full)
    beta_rows = []
    for a in ASSETS:
        y = asset_rets[a].dropna()
        X = factor_rets.loc[y.index, list(FACTORS.keys())]

        if USE_ROLLING:
            y = y.tail(ROLLING_WINDOW)
            X = X.tail(ROLLING_WINDOW)

        betas, r2 = ols_betas(y, X, MIN_OBS)

        if USE_SHRINKAGE:
            betas = shrink_betas(betas, BETA_PRIOR, SHRINK_LAMBDA)

        beta_rows.append({"Asset": a, "R2": r2, **betas.to_dict()})

    betas_df = pd.DataFrame(beta_rows).set_index("Asset")
    betas_df.to_csv(BETAS_OUT)

    # Compute historical factor shocks over the chosen window
    if REPLAY_MODE == "cumulative":
        shocks = factor_shock_cumulative(factor_prices_raw, STRESS_START, STRESS_END)
        # Align to factor order
        shocks = shocks.reindex(list(FACTORS.keys())).astype(float)

        # Stress returns and PnL
        stressed_asset_returns = betas_df[list(FACTORS.keys())].mul(shocks, axis=1).sum(axis=1)

        latest_prices = asset_prices.iloc[-1]
        base_values = pd.Series(
            {a: float(latest_prices[a]) * float(HOLDINGS_SHARES.get(a, 0.0)) for a in ASSETS},
            dtype=float,
        )

        pnl = base_values * stressed_asset_returns
        stressed_values = base_values + pnl

        results = pd.DataFrame({
            "LatestPrice": latest_prices.astype(float),
            "Shares": pd.Series({a: float(HOLDINGS_SHARES.get(a, 0.0)) for a in ASSETS}),
            "BaseValue": base_values,
            "StressedReturn": stressed_asset_returns,
            "PnL": pnl,
            "StressedValue": stressed_values,
        })

        portfolio_base = float(results["BaseValue"].sum())
        portfolio_pnl = float(results["PnL"].sum())
        portfolio_ret = (portfolio_pnl / portfolio_base) if portfolio_base else np.nan

        summary_row = pd.DataFrame([{
            "LatestPrice": np.nan,
            "Shares": np.nan,
            "BaseValue": portfolio_base,
            "StressedReturn": portfolio_ret,
            "PnL": portfolio_pnl,
            "StressedValue": portfolio_base + portfolio_pnl,
        }], index=["PORTFOLIO"])

        out = pd.concat([results, summary_row], axis=0)
        out.to_csv(RESULTS_OUT)

        print("Historical factor stress complete (cumulative).")
        print(f"Episode: {STRESS_START} to {STRESS_END}")
        print(f"Betas saved to: {BETAS_OUT}")
        print(f"Results saved to: {RESULTS_OUT}")
        print(f"Portfolio BaseValue: {portfolio_base:,.2f}")
        print(f"Portfolio PnL: {portfolio_pnl:,.2f}")
        print(f"Portfolio Return: {portfolio_ret:.4%}")
        print("Factor shocks used:")
        for k in shocks.index:
            print(f"  {k}: {shocks[k]:+.4%}")

    elif REPLAY_MODE == "path":
        # Path replay: produce daily stressed PnL path over the historical window
        s = pd.to_datetime(STRESS_START)
        e = pd.to_datetime(STRESS_END)
        fr = factor_rets.loc[(factor_rets.index >= s) & (factor_rets.index <= e)].dropna(how="any")
        if fr.empty or len(fr) < 2:
            raise ValueError("Not enough factor return data in the specified stress window for path replay.")

        # Daily asset returns implied by factors: r_i,t = sum_k beta_i,k * f_k,t
        betas_only = betas_df[list(FACTORS.keys())]
        implied_asset_daily = fr @ betas_only.T  # index=date, columns=assets

        # Apply to today's base values to get daily PnL path approximation
        latest_prices = asset_prices.iloc[-1]
        base_values = pd.Series({a: float(latest_prices[a]) * float(HOLDINGS_SHARES.get(a, 0.0)) for a in ASSETS})

        pnl_path = implied_asset_daily.mul(base_values, axis=1)
        portfolio_pnl_path = pnl_path.sum(axis=1)

        out = pnl_path.copy()
        out["PORTFOLIO_PnL"] = portfolio_pnl_path
        out.to_csv(RESULTS_OUT)

        print("Historical factor stress complete (path).")
        print(f"Episode: {STRESS_START} to {STRESS_END}")
        print(f"Betas saved to: {BETAS_OUT}")
        print(f"PnL path saved to: {RESULTS_OUT}")

    else:
        raise ValueError("REPLAY_MODE must be 'cumulative' or 'path'.")


if __name__ == "__main__":
    main()
