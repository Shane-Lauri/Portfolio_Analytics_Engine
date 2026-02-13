# factor_stress_test.py
#
# Factor-based hypothetical stress test that:
# 1) Uses your existing daily closes CSV produced by:
#    C:\Users\franc\Downloads\VSC\AdSynthAI\Market_Risk_spglobal\daily_closes.py
#    -> daily_adjusted_closes.csv
# 2) Downloads factor proxies (ETFs/index tickers) via yfinance for the same window
# 3) Estimates betas (sensitivities) for each asset to the factors using OLS
# 4) Applies a hypothetical factor shock scenario to compute implied stressed returns
# 5) Produces portfolio P&L given holdings and writes CSV outputs
#
# Outputs (in the same folder):
# - factor_betas.csv
# - factor_stress_results.csv

from __future__ import annotations

import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import certifi

# Force cert bundle for requests
cert_path = certifi.where()
os.environ["SSL_CERT_FILE"] = cert_path
os.environ["REQUESTS_CA_BUNDLE"] = cert_path

# Critical: prevent curl/curl_cffi from being used
os.environ["YFINANCE_USE_CURL_CFFI"] = "0"

# Remove any broken curl bundle pointers
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("CURL_SSL_BACKEND", None)
import yfinance as yf

from daily_closes import download_daily_closes



# ---------------------------- CONFIG ----------------------------

BASE_DIR = Path(r"C:\Users\franc\Downloads\VSC\AdSynthAI\Market_Risk_spglobal")

# This is the CSV produced by your daily_closes.py script.
ASSET_CLOSES_CSV = BASE_DIR / "daily_adjusted_closes.csv"

# Assets must be columns in daily_adjusted_closes.csv (besides Date).
ASSETS = ["NVDA", "AAPL", "MSFT", "JPM", "XOM"]  # any number

# Holdings (shares). Used for portfolio $ P&L. Add/remove as needed.
HOLDINGS_SHARES = {
    "NVDA": 8, 
    "AAPL":15, 
    "MSFT":10, 
    "JPM":20, 
    "XOM":25,
}

# Factor proxies (factor_name -> yahoo_ticker). Choose what you want.
FACTORS = {
    "MKT": "SPY",     # broad market
    "SEMI": "SMH",    # semiconductors
    #"TECH": "QQQ",    # tech/growth tilt (pick QQQ OR XLK typically)
    "VOL": "VXX",     # volatility
    "RATES": "TLT",   # optional rates proxy (price return proxy)
    #"USD": "UUP",     # USD proxy
    #"ENERGY": "XLE",  # Energy / inflation
    #"FIN": "XLF",     # Banking / credit cycle
}

# Hypothetical factor shocks (factor_name -> shock_return)
# Example: market -25%, semis -35%, rates -15%, volatility +60%
FACTOR_SHOCKS = {
    "MKT": -0.20,    # equity selloff
    "SEMI": -0.25,   # growth derating
    "VOL": -0.30,    # credit stress
    "RATES": -0.15,  # yields up → bond prices down
}

# Regression / estimation settings
LOOKBACK_TRADING_DAYS = 252 * 3  # ~3 years of daily returns (adjust as you want)
MIN_OBS = 120                    # minimum overlapping return observations required

# Output files
BETAS_OUT = BASE_DIR / "factor_betas.csv"
RESULTS_OUT = BASE_DIR / "factor_stress_results.csv"

# --- beta robustness settings ---
USE_ROLLING = True
ROLLING_WINDOW = 252       # 1Y daily betas
USE_ORTHOG = True          # reduce overlap between SPY and SMH
USE_SHRINKAGE = True
SHRINK_LAMBDA = 0.70       # 70% data, 30% prior

# Priors (very standard starting point)
BETA_PRIOR = {
    "MKT": 1.0,
    "SEMI": 0.0,   # after orthogonalization this is "semis beyond market"
    "RATES": 0.0,
    "VOL": 0.0,
}


# -------------------------- END CONFIG --------------------------


def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Input CSV must contain a 'Date' column.")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Could not parse some dates in the 'Date' column.")
    return df.sort_values("Date")


def _load_asset_prices(csv_path: Path, assets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _ensure_date_col(df)

    missing = [a for a in assets if a not in df.columns]
    if missing:
        raise ValueError(f"Missing asset columns in {csv_path.name}: {missing}")

    out = df[["Date"] + assets].dropna(subset=assets, how="all")
    out = out.set_index("Date")
    return out


def _download_factor_prices(factor_tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # yfinance end is exclusive-ish; add a day buffer
    data = yf.download(
        tickers=factor_tickers,
        start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            "No factor data downloaded from yfinance. "
            "If you see curl (77), SSL cert env vars are not configured in this script."
        )


    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def _pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def _ols_betas(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    OLS regression: y = a + X b + e
    Returns:
      betas: Series indexed by factor names
      r2: float
    """
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < MIN_OBS:
        raise ValueError(f"Not enough overlapping observations: {df.shape[0]} < {MIN_OBS}")

    yv = df.iloc[:, 0].to_numpy(dtype=float)
    Xv = df.iloc[:, 1:].to_numpy(dtype=float)

    # add intercept
    Xv = np.column_stack([np.ones(len(yv)), Xv])

    # solve least squares
    coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
    alpha = float(coef[0])
    betas = pd.Series(coef[1:], index=X.columns, dtype=float)

    # R^2
    yhat = Xv @ coef
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return betas, r2


def ensure_closes_csv_has_columns(csv_path: Path, required_tickers: list[str], days_back: int) -> None:
    """
    Ensures csv_path exists and contains required_tickers as columns (plus Date).
    If missing tickers are found, downloads only those and merges by Date into the CSV.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            raise ValueError(f"{csv_path.name} exists but has no 'Date' column.")
    else:
        df = pd.DataFrame(columns=["Date"])

    missing = [t for t in required_tickers if t not in df.columns]
    if not missing:
        return

    print(f"CSV missing columns {missing}. Downloading and merging into {csv_path.name}...")

    new_df = download_daily_closes(missing, days_back)
    if new_df.empty:
        raise RuntimeError(f"Failed to download missing tickers: {missing}")

    # --- normalize Date types before merge ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")

    # Drop any rows where Date couldn't be parsed
    df = df.dropna(subset=["Date"])
    new_df = new_df.dropna(subset=["Date"])

    # Merge on Date
    if df.empty or df.shape[1] == 1:  # only Date
        merged = new_df
    else:
        merged = pd.merge(df, new_df, on="Date", how="outer")

    # Sort and format Date back to string for CSV
    merged = merged.sort_values("Date")
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")


    try:
        merged.to_csv(csv_path, index=False)
        print(f"Updated file saved: {csv_path}")
    except PermissionError:
        # If Excel has it open, write a timestamped fallback
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = csv_path.with_name(f"{csv_path.stem}_{ts}{csv_path.suffix}")
        merged.to_csv(fallback, index=False)
        print(f"Could not overwrite (file likely open). Saved instead: {fallback}")

def shrink_betas(betas: pd.Series, prior: dict, lam: float) -> pd.Series:
    prior_vec = pd.Series({k: float(prior.get(k, 0.0)) for k in betas.index}, dtype=float)
    return lam * betas + (1.0 - lam) * prior_vec


def orthogonalize_factors(factor_rets: pd.DataFrame) -> pd.DataFrame:
    """
    Make correlated factors more interpretable/stable.
    - SEMI becomes 'semis beyond market' by removing its market component.
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


def main():
    if not ASSET_CLOSES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {ASSET_CLOSES_CSV}. Run daily_closes.py first to create daily_adjusted_closes.csv."
        )

    # Ensure the CSV contains all required asset + factor tickers.
    required_tickers = ASSETS + list(FACTORS.values())

    # Use a days_back large enough to cover your regression window.
    days_back_needed = LOOKBACK_TRADING_DAYS + 60  # buffer
    ensure_closes_csv_has_columns(ASSET_CLOSES_CSV, required_tickers, days_back_needed)


    # Load asset prices and restrict to lookback window
    asset_prices = _load_asset_prices(ASSET_CLOSES_CSV, ASSETS)

    if len(asset_prices) < 5:
        raise ValueError("Not enough rows in asset price history.")

    asset_prices = asset_prices.tail(LOOKBACK_TRADING_DAYS)

    start_dt = asset_prices.index.min()
    end_dt = asset_prices.index.max()

    # Read factor prices from the same CSV (columns are the Yahoo tickers like SPY/SMH/VXX/TLT)
    df_all = pd.read_csv(ASSET_CLOSES_CSV)
    df_all = _ensure_date_col(df_all).set_index("Date")

    # Restrict to same window used for assets
    df_all = df_all.loc[start_dt:end_dt]

    # Build factor_prices with columns named by factor keys (MKT/SEMI/VOL/RATES)
    factor_prices = pd.DataFrame(index=df_all.index)
    for factor_name, ticker in FACTORS.items():
        if ticker not in df_all.columns:
            raise ValueError(f"Factor ticker '{ticker}' missing from CSV even after ensure step.")
        factor_prices[factor_name] = df_all[ticker].astype(float)


    # Compute returns
    asset_rets = _pct_returns(asset_prices)
    factor_rets = _pct_returns(factor_prices)

    # Align on dates
    common_index = asset_rets.index.intersection(factor_rets.index)
    asset_rets = asset_rets.loc[common_index]
    factor_rets = factor_rets.loc[common_index]

    if USE_ORTHOG:
        factor_rets = orthogonalize_factors(factor_rets)

    # Estimate betas for each asset
    beta_rows = []
    for a in ASSETS:
        y = asset_rets[a].dropna()
        X = factor_rets.loc[y.index, list(FACTORS.keys())]

        # Rolling window (use most recent observations)
        if USE_ROLLING:
            y = y.tail(ROLLING_WINDOW)
            X = X.tail(ROLLING_WINDOW)

        betas, r2 = _ols_betas(y, X)

        # Shrinkage toward priors (stabilize noisy estimates)
        if USE_SHRINKAGE:
            betas = shrink_betas(betas, BETA_PRIOR, SHRINK_LAMBDA)

        row = {"Asset": a, "R2": r2, **betas.to_dict()}
        beta_rows.append(row)


    betas_df = pd.DataFrame(beta_rows).set_index("Asset")
    betas_df.to_csv(BETAS_OUT)

    # Apply hypothetical factor shocks to get implied stressed return for each asset
    shocks_vec = pd.Series(FACTOR_SHOCKS, dtype=float)

    missing_shocks = [k for k in FACTORS.keys() if k not in shocks_vec.index]
    if missing_shocks:
        raise ValueError(f"Missing shocks for factors: {missing_shocks}")

    stressed_asset_returns = betas_df[list(FACTORS.keys())].mul(shocks_vec, axis=1).sum(axis=1)

    # Convert stressed return to $ impact using latest prices and shares
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

    # Add portfolio summary row (optional but handy)
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

    print("Factor stress complete.")
    print(f"Betas saved to: {BETAS_OUT}")
    print(f"Results saved to: {RESULTS_OUT}")
    print(f"Window used: {start_dt.date()} to {end_dt.date()} ({len(common_index)} return obs)")
    print(f"Portfolio BaseValue: {portfolio_base:,.2f}")
    print(f"Portfolio PnL: {portfolio_pnl:,.2f}")
    print(f"Portfolio Return: {portfolio_ret:.4%}")


if __name__ == "__main__":
    main()
