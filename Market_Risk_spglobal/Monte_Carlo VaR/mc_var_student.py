"""
Monte Carlo VaR (1Y) using daily_closes.py for data fetching

Assumptions:
- Portfolio daily returns follow a Student-t distribution (i.i.d.)
- Simple returns
- 252 trading days per year
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT YOUR EXISTING SCRIPT
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
N_PATHS = 200_000            # MC simulations
N_DAYS = 252                 # 1-year horizon
INITIAL_VALUE = 1_000_000    # portfolio notional
RANDOM_SEED = 42

# Student-t degrees of freedom (lower = heavier tails). Common range: 4..10
T_DF = 6
# ============================================================


def build_portfolio_returns(
    closes: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute daily portfolio simple returns from adjusted close prices."""
    df = closes.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    tickers = list(weights.keys())
    prices = df[tickers].astype(float).dropna(how="any")

    # Normalize weights to sum to 1
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()

    asset_rets = prices.pct_change().dropna()
    port_rets = asset_rets.mul(w, axis=1).sum(axis=1)
    return port_rets


def _t_scale_for_target_sigma(df: float, target_sigma: float) -> float:
    """
    If Z ~ t_df (standard, loc=0, scale=1), then Var(Z) = df/(df-2) for df>2.
    For X = scale * Z, Var(X) = scale^2 * df/(df-2).
    Choose scale so that Std(X) = target_sigma:
        scale = target_sigma * sqrt((df-2)/df)
    """
    if df <= 2:
        raise ValueError("T_DF must be > 2 to have finite variance.")
    return float(target_sigma * np.sqrt((df - 2.0) / df))


def monte_carlo_var_1y_t(
    daily_returns: pd.Series,
    alpha: float,
    n_paths: int,
    n_days: int,
    initial_value: float,
    df: float,
    seed: int | None = None,
):
    """
    Student-t Monte Carlo VaR over 1 year.
    Uses:
      sim_returns = mu + scale * t_df_samples
    with scale chosen so simulated daily std matches historical sigma.
    """
    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std(ddof=1))

    rng = np.random.default_rng(seed)

    scale = _t_scale_for_target_sigma(df=df, target_sigma=sigma)

    # Generate t innovations (mean 0 for df>1). We keep mu separately.
    t_innov = rng.standard_t(df=df, size=(n_paths, n_days))
    sim_returns = mu + scale * t_innov

    terminal_values = initial_value * np.prod(1.0 + sim_returns, axis=1)
    losses = initial_value - terminal_values
    var = float(np.quantile(losses, alpha))

    return var, sim_returns, terminal_values


if __name__ == "__main__":
    # --------------------------------------------------------
    # 1. Fetch data USING daily_closes.py
    # --------------------------------------------------------
    tickers = list(PORTFOLIO_WEIGHTS.keys())
    closes_df = download_daily_closes(tickers, DAYS_BACK)

    if closes_df.empty:
        raise RuntimeError("Failed to fetch daily closes.")

    # --------------------------------------------------------
    # 2. Build portfolio daily returns
    # --------------------------------------------------------
    portfolio_returns = build_portfolio_returns(
        closes=closes_df,
        weights=PORTFOLIO_WEIGHTS,
    )

    # --------------------------------------------------------
    # 3. Monte Carlo VaR (Student-t)
    # --------------------------------------------------------
    var_1y, sim_returns, terminal_values = monte_carlo_var_1y_t(
        daily_returns=portfolio_returns,
        alpha=ALPHA,
        n_paths=N_PATHS,
        n_days=N_DAYS,
        initial_value=INITIAL_VALUE,
        df=T_DF,
        seed=RANDOM_SEED,
    )

    print("--------------------------------------------------")
    print(f"Monte Carlo Student-t VaR (1Y), df={T_DF}")
    print(f"Confidence level : {int(ALPHA*100)}%")
    print(f"Paths simulated  : {N_PATHS:,}")
    print(f"VaR (absolute)   : {var_1y:.4f}")
    print(f"VaR (% of NAV)   : {var_1y / INITIAL_VALUE * 100:.2f}%")
    print("--------------------------------------------------")

    # --------------------------------------------------------
    # 4. Plot paths and highlight bottom 5%
    # --------------------------------------------------------
    N_PLOT = 150  # change as you like

    path_values = np.empty((sim_returns.shape[0], sim_returns.shape[1] + 1), dtype=float)
    path_values[:, 0] = INITIAL_VALUE
    path_values[:, 1:] = INITIAL_VALUE * np.cumprod(1.0 + sim_returns, axis=1)

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

    plt.title(f"Monte Carlo 1Y Portfolio Paths (Student-t, df={T_DF})")
    plt.xlabel("Trading days")
    plt.ylabel("Portfolio value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


ALL_ALPHAS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]

def run_mc_var_student(
    portfolio_df: pd.DataFrame,
    confidence_level: float,
    horizon_days: int,
    start_year: int,
    end_year: int,
    portfolio_value: float = 1_000_000.0,
    n_simulations: int = 200_000,
    df: float = 6.0,
):
    import datetime as dt

    tickers = portfolio_df["ticker"].astype(str).str.strip().tolist()
    weights = portfolio_df["weight"].astype(float).tolist()
    weights_dict = {t: w for t, w in zip(tickers, weights)}

    days_back = (dt.datetime(end_year, 12, 31) - dt.datetime(start_year, 1, 1)).days

    closes_df = download_daily_closes(tickers, days_back)
    if closes_df.empty:
        raise RuntimeError("Failed to fetch daily closes.")

    portfolio_returns = build_portfolio_returns(closes_df, weights_dict)

    # Run MC once — reuse same simulation for all confidence levels
    _, sim_returns, terminal_values = monte_carlo_var_1y_t(
        daily_returns=portfolio_returns,
        alpha=confidence_level,
        n_paths=n_simulations,
        n_days=horizon_days,
        initial_value=portfolio_value,
        df=df,
        seed=42,
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
        })

    results_df = pd.DataFrame(rows)
    headline_var = float(np.quantile(losses, confidence_level))

    # Plot — highlight worst (1 - confidence_level) paths
    tail_pct = 1.0 - confidence_level
    N_PLOT = 150

    path_values = np.empty((sim_returns.shape[0], sim_returns.shape[1] + 1))
    path_values[:, 0] = portfolio_value
    path_values[:, 1:] = portfolio_value * np.cumprod(1.0 + sim_returns, axis=1)

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

    for j, i in enumerate(sel_rest):
        ax.plot(x, path_values[i], color="steelblue", lw=0.8, alpha=0.35,
                label="Other paths" if j == 0 else None)
    for j, i in enumerate(sel_worst):
        ax.plot(x, path_values[i], color="crimson", lw=1.2, alpha=0.9,
                label=f"Worst {tail_pct*100:.1f}%" if j == 0 else None)

    ax.set_title(
        f"MC Student-t (df={df}) – {horizon_days}-Day Paths  "
        f"(Worst {tail_pct*100:.1f}% highlighted  |  VaR {confidence_level*100:.1f}% = ${headline_var:,.0f})"
    )
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()

    return results_df, fig, headline_var