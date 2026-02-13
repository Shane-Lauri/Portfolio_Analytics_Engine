"""
Monte Carlo VaR (1Y) using daily_closes.py for data fetching

FASTER copula-based simulation:
- Empirical marginals via FAST inverse ECDF (rank lookup, no interpolation)
- Uses float32 for large simulation arrays (cuts memory + speeds up)
- Keeps terminal value / quantile computations in float64 for numerical stability
- Optional switch: Gaussian copula is much faster than t-copula

Requires: scipy
    pip install scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, t
from scipy.linalg import cholesky

from daily_closes import download_daily_closes


# ============================================================
# EDIT PORTFOLIO HERE
# ============================================================
PORTFOLIO_WEIGHTS = {
    "NVDA": 0.25,
    "AAPL": 0.25,
    "MSFT": 0.20,
    "GOOGL": 0.15,
    "TSLA": 0.15,
}

DAYS_BACK = 365 * 3          # historical window for estimation
ALPHA = 0.99                 # VaR confidence

# Biggest speed lever:
N_PATHS = 50_000             # was 200_000 (4x faster). Raise to 100k+ for final runs.

N_DAYS = 252                 # 1-year horizon
INITIAL_VALUE = 1_000_000    # portfolio notional
RANDOM_SEED = 42

# Copula settings
# NOTE: Gaussian copula is faster than t-copula. Use t for final runs if needed.
COPULA_TYPE = "t"            # "gaussian" or "t"
COPULA_DF = 6                # only used if COPULA_TYPE == "t"

# Plot settings
N_PLOT = 50                  # was 150 (faster plotting)

# Simulation dtype (speed/memory). Keep VaR quantile step in float64.
SIM_DTYPE = np.float32
# ============================================================


def build_asset_returns(closes: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    df = closes.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    prices = df[tickers].astype(float).dropna(how="any")
    return prices.pct_change().dropna()


def _normalize_weights(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    w = pd.Series(weights, dtype=float).reindex(tickers)
    if w.isna().any():
        missing = w[w.isna()].index.tolist()
        raise ValueError(f"Missing weights for tickers: {missing}")
    s = float(w.sum())
    if s == 0:
        raise ValueError("Sum of weights is zero.")
    return (w / s).to_numpy(dtype=np.float64)


def _rank_to_uniform(x: np.ndarray) -> np.ndarray:
    """
    Rank-based pseudo uniforms u_i = rank_i/(n+1). O(n log n) per asset, done once.
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks / (n + 1.0)


def _empirical_ppf_fast(samples_sorted: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    FAST inverse-ECDF using rank lookup (no interpolation).
    Maps u in (0,1) -> integer index 0..n-1 and returns samples_sorted[idx].
    Much faster than interpolation and typically adequate for VaR/ES with enough history/paths.
    """
    n = samples_sorted.size
    idx = (u * n).astype(np.int64)
    np.clip(idx, 0, n - 1, out=idx)
    return samples_sorted[idx]


def fit_copula_and_simulate_paths(
    asset_returns: pd.DataFrame,
    n_paths: int,
    n_days: int,
    copula_type: str = "t",
    copula_df: int = 6,
    seed: int | None = None,
    sim_dtype=np.float32,
) -> np.ndarray:
    """
    Returns simulated daily returns with shape (n_paths, n_days, n_assets)
    using empirical marginals + Gaussian or t-copula dependence.
    """
    if copula_type not in {"gaussian", "t"}:
        raise ValueError("copula_type must be 'gaussian' or 't'.")

    rng = np.random.default_rng(seed)

    X = asset_returns.to_numpy(dtype=np.float64)  # (T, d), small enough to keep float64
    T_obs, d = X.shape
    if T_obs < 100:
        raise ValueError("Not enough historical observations to fit a copula robustly.")

    # 1) Empirical marginals (sorted) — store in sim_dtype for faster lookup
    sorted_marginals = [np.sort(X[:, j]).astype(sim_dtype, copy=False) for j in range(d)]

    # 2) Historical uniforms via ranks
    U = np.column_stack([_rank_to_uniform(X[:, j]) for j in range(d)])  # float64

    # 3) Latent for dependence estimation
    if copula_type == "gaussian":
        Z = norm.ppf(U)
    else:
        if copula_df <= 2:
            raise ValueError("COPULA_DF must be > 2 for stable t-copula dependence estimation.")
        Z = t.ppf(U, df=copula_df)

    # 4) Correlation in latent space
    R = np.corrcoef(Z, rowvar=False)
    R = (R + R.T) / 2.0
    R = R + 1e-10 * np.eye(d)

    # 5) Cholesky
    L = cholesky(R, lower=True).astype(sim_dtype, copy=False)

    # 6) Simulate latent draws and convert to uniforms
    # Allocate simulation arrays in float32
    E = rng.standard_normal(size=(n_paths, n_days, d)).astype(sim_dtype, copy=False)
    G = E @ L.T  # correlate last dimension

    if copula_type == "gaussian":
        # norm.cdf will upcast to float64 internally; cast back
        U_sim = norm.cdf(G).astype(sim_dtype, copy=False)
    else:
        # t-copula: G / sqrt(S/df), S~chi2(df)
        S = rng.chisquare(df=copula_df, size=(n_paths, n_days, 1)).astype(sim_dtype, copy=False)
        Z_sim = G / np.sqrt(S / np.float32(copula_df))
        U_sim = t.cdf(Z_sim, df=copula_df).astype(sim_dtype, copy=False)

    # 7) Inverse empirical marginals (FAST)
    X_sim = np.empty_like(U_sim, dtype=sim_dtype)
    for j in range(d):
        X_sim[:, :, j] = _empirical_ppf_fast(sorted_marginals[j], U_sim[:, :, j])

    return X_sim


def monte_carlo_var_1y_copula(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    alpha: float,
    n_paths: int,
    n_days: int,
    initial_value: float,
    copula_type: str,
    copula_df: int,
    seed: int | None = None,
    sim_dtype=np.float32,
):
    """
    Returns: (var, sim_portfolio_returns, terminal_values)
      - sim_portfolio_returns shape: (n_paths, n_days)  (sim_dtype)
      - terminal_values shape: (n_paths,)               (float64)
    """
    tickers = list(asset_returns.columns)
    w = _normalize_weights(weights, tickers)  # float64

    sim_asset_returns = fit_copula_and_simulate_paths(
        asset_returns=asset_returns,
        n_paths=n_paths,
        n_days=n_days,
        copula_type=copula_type,
        copula_df=copula_df,
        seed=seed,
        sim_dtype=sim_dtype,
    )  # (n_paths, n_days, d) in float32

    # Portfolio daily returns
    # tensordot returns float64 if w is float64; keep float32 by casting w to float32 for dot,
    # then upcast later only for terminal/quantile.
    w32 = w.astype(sim_dtype, copy=False)
    sim_port_rets = np.tensordot(sim_asset_returns, w32, axes=([2], [0])).astype(sim_dtype, copy=False)  # (n_paths, n_days)

    # Terminal values in float64 for stability
    terminal_values = (np.float64(initial_value) * np.prod(1.0 + sim_port_rets.astype(np.float64), axis=1))
    losses = np.float64(initial_value) - terminal_values
    var = float(np.quantile(losses, alpha))

    return var, sim_port_rets, terminal_values


if __name__ == "__main__":
    tickers = list(PORTFOLIO_WEIGHTS.keys())
    closes_df = download_daily_closes(tickers, DAYS_BACK)
    if closes_df.empty:
        raise RuntimeError("Failed to fetch daily closes.")

    asset_returns = build_asset_returns(closes_df, tickers)

    var_1y, sim_returns, terminal_values = monte_carlo_var_1y_copula(
        asset_returns=asset_returns,
        weights=PORTFOLIO_WEIGHTS,
        alpha=ALPHA,
        n_paths=N_PATHS,
        n_days=N_DAYS,
        initial_value=INITIAL_VALUE,
        copula_type=COPULA_TYPE,
        copula_df=COPULA_DF,
        seed=RANDOM_SEED,
        sim_dtype=SIM_DTYPE,
    )

    print("--------------------------------------------------")
    if COPULA_TYPE == "t":
        print(f"Monte Carlo Copula VaR (1Y): t-copula df={COPULA_DF}, empirical marginals (FAST)")
    else:
        print("Monte Carlo Copula VaR (1Y): Gaussian copula, empirical marginals (FAST)")
    print(f"Confidence level : {int(ALPHA*100)}%")
    print(f"Paths simulated  : {N_PATHS:,}")
    print(f"VaR (absolute)   : {var_1y:.4f}")
    print(f"VaR (% of NAV)   : {var_1y / INITIAL_VALUE * 100:.2f}%")
    print("--------------------------------------------------")

    # ---------------- Plot paths ----------------
    path_values = np.empty((sim_returns.shape[0], sim_returns.shape[1] + 1), dtype=np.float64)
    path_values[:, 0] = INITIAL_VALUE
    path_values[:, 1:] = np.float64(INITIAL_VALUE) * np.cumprod(1.0 + sim_returns.astype(np.float64), axis=1)

    bottom_q = np.quantile(terminal_values, 0.05)
    worst_mask = terminal_values <= bottom_q

    worst_idx = np.where(worst_mask)[0]
    rest_idx = np.where(~worst_mask)[0]

    rng = np.random.default_rng(RANDOM_SEED)

    n_worst_plot = min(max(1, N_PLOT // 5), worst_idx.size)
    n_rest_plot = min(N_PLOT - n_worst_plot, rest_idx.size)

    sel_worst = rng.choice(worst_idx, size=n_worst_plot, replace=False) if n_worst_plot > 0 else np.array([], dtype=int)
    sel_rest = rng.choice(rest_idx, size=n_rest_plot, replace=False) if n_rest_plot > 0 else np.array([], dtype=int)

    x = np.arange(path_values.shape[1])

    COLOR_NORMAL = "steelblue"
    COLOR_WORST = "crimson"

    plt.figure(figsize=(11, 6))

    for j, i in enumerate(sel_rest):
        plt.plot(
            x,
            path_values[i],
            color=COLOR_NORMAL,
            linewidth=0.8,
            alpha=0.35,
            label="Other paths" if j == 0 else None,
        )

    for j, i in enumerate(sel_worst):
        plt.plot(
            x,
            path_values[i],
            color=COLOR_WORST,
            linewidth=1.2,
            alpha=0.9,
            label="Bottom 5%" if j == 0 else None,
        )

    if COPULA_TYPE == "t":
        title = f"Monte Carlo 1Y Portfolio Paths (FAST t-copula df={COPULA_DF}, bottom 5% highlighted)"
    else:
        title = "Monte Carlo 1Y Portfolio Paths (FAST Gaussian copula, bottom 5% highlighted)"

    plt.title(title)
    plt.xlabel("Trading days")
    plt.ylabel("Portfolio value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()



ALL_ALPHAS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]

def run_mc_var_copula(
    portfolio_df: pd.DataFrame,
    confidence_level: float,
    horizon_days: int,
    start_year: int,
    end_year: int,
    portfolio_value: float = 1_000_000.0,
    n_simulations: int = 50_000,
    copula_type: str = "t",
    copula_df: int = 6,
):
    import datetime as dt

    tickers = portfolio_df["ticker"].astype(str).str.strip().tolist()
    weights = portfolio_df["weight"].astype(float).tolist()
    weights_dict = {t: w for t, w in zip(tickers, weights)}

    days_back = (dt.datetime(end_year, 12, 31) - dt.datetime(start_year, 1, 1)).days

    closes_df = download_daily_closes(tickers, days_back)
    if closes_df.empty:
        raise RuntimeError("Failed to fetch daily closes.")

    asset_returns = build_asset_returns(closes_df, tickers)

    # Run MC once — reuse same simulation for all confidence levels
    _, sim_returns, terminal_values = monte_carlo_var_1y_copula(
        asset_returns=asset_returns,
        weights=weights_dict,
        alpha=confidence_level,
        n_paths=n_simulations,
        n_days=horizon_days,
        initial_value=portfolio_value,
        copula_type=copula_type,
        copula_df=copula_df,
        seed=42,
        sim_dtype=SIM_DTYPE,
    )

    losses = portfolio_value - terminal_values

    # Compute VaR at every confidence level
    rows = []
    for alpha in ALL_ALPHAS:
        var_val = float(np.quantile(losses, alpha))
        rows.append({
            "Confidence": f"{alpha*100:.1f}%",
            "VaR ($)": f"${var_val:,.0f}",
            "VaR (% NAV)": f"{var_val / portfolio_value * 100:.2f}%",
            "Horizon (days)": horizon_days,
            "Simulations": f"{n_simulations:,}",
            "Copula": f"{copula_type} (df={copula_df})" if copula_type == "t" else "Gaussian",
        })

    results_df = pd.DataFrame(rows)
    headline_var = float(np.quantile(losses, confidence_level))

    # Plot — highlight worst (1 - confidence_level) paths
    tail_pct = 1.0 - confidence_level
    N_PLOT = 150

    path_values = np.empty((sim_returns.shape[0], sim_returns.shape[1] + 1), dtype=np.float64)
    path_values[:, 0] = portfolio_value
    path_values[:, 1:] = np.float64(portfolio_value) * np.cumprod(
        1.0 + sim_returns.astype(np.float64), axis=1
    )

    cutoff_q = np.quantile(terminal_values, tail_pct)
    worst_mask = terminal_values <= cutoff_q
    worst_idx = np.where(worst_mask)[0]
    rest_idx  = np.where(~worst_mask)[0]

    rng = np.random.default_rng(42)
    n_worst_plot = min(max(1, N_PLOT // 5), worst_idx.size)
    n_rest_plot  = min(N_PLOT - n_worst_plot, rest_idx.size)
    sel_worst = rng.choice(worst_idx, size=n_worst_plot, replace=False)
    sel_rest  = rng.choice(rest_idx,  size=n_rest_plot,  replace=False)

    x = np.arange(path_values.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))

    copula_label = f"t-copula (df={copula_df})" if copula_type == "t" else "Gaussian copula"

    for j, i in enumerate(sel_rest):
        ax.plot(x, path_values[i], color="steelblue", lw=0.8, alpha=0.35,
                label="Other paths" if j == 0 else None)
    for j, i in enumerate(sel_worst):
        ax.plot(x, path_values[i], color="crimson", lw=1.2, alpha=0.9,
                label=f"Worst {tail_pct*100:.1f}%" if j == 0 else None)

    ax.set_title(
        f"MC Copula ({copula_label}) – {horizon_days}-Day Paths  "
        f"(Worst {tail_pct*100:.1f}% highlighted  |  VaR {confidence_level*100:.1f}% = ${headline_var:,.0f})"
    )
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()

    return results_df, fig, headline_var