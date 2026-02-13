"""
One-day Parametric VaR (95% and 99%) for Normal and Student-t
using daily_closes.py to fetch adjusted closes.

Outputs:
- Normal VaR 95/99 (1-day)
- Student-t VaR 95/99 (1-day) with df inferred from excess kurtosis (heuristic)
- Plot: histogram of historical daily portfolio returns + fitted Normal and Student-t PDFs
        + 4 VaR vertical lines

No scipy required.
Requires: mpmath (pure python). If missing: pip install mpmath
"""

from __future__ import annotations



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from daily_closes import download_daily_closes


def run_bin_t_student_var(
    portfolio_df,
    confidence_level: float,
    horizon_days: int,
    start_year: int,
    end_year: int,
):
    """
    Called from Excel via xlwings.
    Returns:
        var_value (float)
        fig (matplotlib.figure.Figure)
    """



# -----------------------------
# Config
# -----------------------------
TICKERS = ["JPM", "NVDA", "AAPL", "MSFT", "GOOGL"]
WEIGHTS = {"NVDA": 0.25, "AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.15, "JPM": 0.50}  # normalized inside code
DAYS_BACK = 365 * 5
PORTFOLIO_VALUE = 1_000_000.0

ALPHAS = (0.95, 0.99)  # confidence levels


# -----------------------------
# Helpers
# -----------------------------
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
    days_back: int,
) -> pd.Series:
    closes = download_daily_closes(tickers, days_back)
    if closes.empty:
        raise RuntimeError("No price data returned (closes is empty).")
    if "Date" not in closes.columns:
        raise RuntimeError("Expected a 'Date' column in closes output.")

    prices = closes.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for tkr in tickers:
        if tkr not in prices.columns:
            raise ValueError(f"Ticker '{tkr}' missing from downloaded closes columns.")
        prices[tkr] = pd.to_numeric(prices[tkr], errors="coerce")

    prices = prices[tickers].dropna(how="any")
    if len(prices) < 2:
        raise RuntimeError("Not enough price rows to compute returns (need at least 2).")

    rets = prices.pct_change().dropna(how="any")
    if rets.empty:
        raise RuntimeError("Returns computed as empty; check price data.")

    w = _normalize_weights(weights, tickers)
    rp = rets.to_numpy(dtype=float) @ w
    return pd.Series(rp, index=rets.index, name="portfolio_return")


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    if m2 <= 0:
        return float("nan")
    return float(m4 / (m2**2) - 3.0)


def estimate_student_t_params_from_history(portfolio_returns: pd.Series) -> dict:
    """
    Heuristic:
      - loc = sample mean
      - df from excess kurtosis: excess = 6/(df-4)  => df = 4 + 6/excess   (valid for df>4)
      - scale chosen so that daily std matches sample sigma:
            Var(t_df) = df/(df-2)
            if R = loc + scale*T, then Std(R)=sigma => scale = sigma*sqrt((df-2)/df)
    """
    r = portfolio_returns.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError("Need at least ~30 observations to fit.")

    loc = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0:
        raise ValueError("Zero/negative volatility; cannot fit.")

    ex_k = _excess_kurtosis(r)
    if not np.isfinite(ex_k) or ex_k <= 0:
        df = 1000.0  # ~Normal
    else:
        df = 4.0 + 6.0 / ex_k
        df = float(np.clip(df, 4.01, 1000.0))

    scale = sigma * math.sqrt((df - 2.0) / df)
    return {"df": df, "loc": loc, "scale": scale, "sigma": sigma, "excess_kurtosis": ex_k}


# -----------------------------
# Quantiles (no scipy)
# -----------------------------
def norm_ppf(p: float) -> float:
    """
    Accurate inverse Normal CDF (Acklam approximation).
    Max error ~ 1e-9.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")

    # Coefficients
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]

    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]

    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]

    d = [7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)



def t_cdf(x: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be > 0")
    x = float(x)
    df = float(df)
    if x == 0.0:
        return 0.5

    z = df / (df + x * x)
    a = df / 2.0
    b = 0.5

    # regularized incomplete beta I_z(a,b)
    #I = mp.betainc(a, b, 0, z, regularized=True)

    # if x > 0:
    #     return float(1.0 - 0.5 * I)
    # else:
    #     return float(0.5 * I)



def t_ppf_approx(p: float, df: float) -> float:
    """
    Approximate Student-t inverse CDF using Cornish–Fisher expansion.
    Accurate for df >= 4 and p in [0.95, 0.99].
    """
    if df <= 2:
        raise ValueError("df must be > 2")

    z = norm_ppf(p)
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2   # <-- THIS WAS MISSING

    return z + (z3 + z) / (4 * df) + (5 * z5 + 16 * z3 + 3 * z) / (96 * df * df)




# -----------------------------
# VaR computation (1-day)
# -----------------------------
def var_1day_parametric_normal(mu: float, sigma: float, alpha: float, portfolio_value: float) -> dict:
    """
    1-day parametric Normal VaR at confidence alpha.
    Computes left-tail return quantile q_{1-alpha}, then VaR = max(0, -q * V0).
    """
    p_left = 1.0 - alpha
    q = mu + sigma * norm_ppf(p_left)  # left-tail return quantile
    var_dollars = max(0.0, -float(q) * float(portfolio_value))
    return {"alpha": float(alpha), "q_return": float(q), "VaR_$": float(var_dollars)}


def var_1day_parametric_t(loc: float, scale: float, df: float, alpha: float, portfolio_value: float) -> dict:
    p_left = 1.0 - alpha
    q = loc + scale * t_ppf_approx(p_left, df=df)  # left-tail return quantile
    var_dollars = max(0.0, -float(q) * float(portfolio_value))
    return {"alpha": float(alpha), "q_return": float(q), "VaR_$": float(var_dollars)}


# -----------------------------
# Plotting PDFs
# -----------------------------
def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def student_t_pdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    # PDF of standardized t: Γ((ν+1)/2)/(sqrt(νπ)Γ(ν/2)) * (1 + t^2/ν)^(-(ν+1)/2)
    tt = (x - loc) / scale
    coef = (math.gamma((df + 1.0) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0)))
    t_pdf_std = coef * (1.0 + (tt**2) / df) ** (-(df + 1.0) / 2.0)
    return (1.0 / scale) * t_pdf_std


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    pr = get_portfolio_returns_from_daily_closes(TICKERS, WEIGHTS, DAYS_BACK)
    r = pr.to_numpy(dtype=float)
    r = r[np.isfinite(r)]

    if r.size < 30:
        raise RuntimeError("Need at least ~30 daily returns.")

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))

    t_params = estimate_student_t_params_from_history(pr)
    df = float(t_params["df"])
    loc = float(t_params["loc"])
    scale = float(t_params["scale"])

    # --- Compute 1-day VaRs at 95 and 99 for both distributions ---
    normal_vars = [var_1day_parametric_normal(mu, sigma, a, PORTFOLIO_VALUE) for a in ALPHAS]
    t_vars = [var_1day_parametric_t(loc, scale, df, a, PORTFOLIO_VALUE) for a in ALPHAS]

    print(f"Observations used: {len(r)}")
    print(f"Portfolio value: {PORTFOLIO_VALUE:,.2f}")

    print("\n--- Normal (parametric, 1-day) ---")
    print(f"mu={mu:.6%}, sigma={sigma:.6%}")
    for dct in normal_vars:
        print(f"VaR {int(dct['alpha']*100)}%: cutoff={dct['q_return']:.6%}, VaR($)={dct['VaR_$']:,.2f}")

    print("\n--- Student-t (parametric, 1-day; df via kurtosis heuristic) ---")
    print(f"df={df:.4f}, loc={loc:.6%}, scale={scale:.6%}, excess_kurtosis={t_params['excess_kurtosis']:.4f}")
    for dct in t_vars:
        print(f"VaR {int(dct['alpha']*100)}%: cutoff={dct['q_return']:.6%}, VaR($)={dct['VaR_$']:,.2f}")

    # --- Plot: histogram + fitted PDFs + four VaR lines ---
    # X-grid
    q_candidates = [d["q_return"] for d in normal_vars + t_vars]
    x_min = min(r.min(), min(q_candidates)) - 0.01
    x_max = max(r.max(), mu + 4 * sigma) + 0.01
    x = np.linspace(x_min, x_max, 2000)

    pdf_n = normal_pdf(x, mu, sigma)
    pdf_t = student_t_pdf(x, df=df, loc=loc, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.hist(r, bins=60, density=True, alpha=0.5, label="Historical daily returns")

    plt.plot(x, pdf_n, linewidth=2, label="Normal fit")
    plt.plot(x, pdf_t, linewidth=2, label=f"Student-t fit (df={df:.2f})")

    # VaR lines (returns cutoffs)
    # Use distinct linestyles so the four lines are readable without relying on many colors.
    styles = {
        ("normal", 0.95): ("--", "Normal 95% VaR cutoff"),
        ("normal", 0.99): (":", "Normal 99% VaR cutoff"),
        ("t", 0.95): ("-.", "Student-t 95% VaR cutoff"),
        ("t", 0.99): ("-", "Student-t 99% VaR cutoff"),
    }

    # Normal
    for dct in normal_vars:
        ls, lab = styles[("normal", dct["alpha"])]
        plt.axvline(dct["q_return"], linestyle=ls, linewidth=2, label=lab)

    # Student-t
    for dct in t_vars:
        ls, lab = styles[("t", dct["alpha"])]
        plt.axvline(dct["q_return"], linestyle=ls, linewidth=2, label=lab)

    plt.title("Daily Portfolio Returns: Historical vs Normal vs Student-t (with 1-day VaR cutoffs)")
    plt.xlabel("Daily portfolio return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()











def run_bin_t_student_var(
    portfolio_df: pd.DataFrame,
    confidence_level: float,
    horizon_days: int,
    start_year: int,
    end_year: int,
    portfolio_value: float = 1_000_000.0,
):
    """
    Returns:
      results_df: table of Normal + Student-t VaR for 95% and 99% (1-day)
      fig: matplotlib Figure (hist + fitted pdfs + VaR cutoffs)
      selected_var: VaR($) for the passed confidence_level using Student-t (default choice)
    """

    # ---- parse portfolio table from Excel ----
    # Expect columns: ticker, weight (case-insensitive)
    cols = {c.lower(): c for c in portfolio_df.columns}
    if "ticker" not in cols or "weight" not in cols:
        raise ValueError("portfolio_df must have columns: ticker, weight")

    t_col = cols["ticker"]
    w_col = cols["weight"]

    tickers = portfolio_df[t_col].astype(str).str.strip().tolist()
    weights = portfolio_df[w_col].astype(float).tolist()
    weights_dict = {t: float(w) for t, w in zip(tickers, weights)}

    # ---- determine history length from years ----
    # download_daily_closes takes days_back; approximate with 365*(years)
    years = max(1, int(end_year) - int(start_year) + 1)
    days_back = 365 * years

    pr = get_portfolio_returns_from_daily_closes(tickers, weights_dict, days_back)

    r = pr.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise RuntimeError("Need at least ~30 daily returns.")

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))

    t_params = estimate_student_t_params_from_history(pr)
    df = float(t_params["df"])
    loc = float(t_params["loc"])
    scale = float(t_params["scale"])

    # 1-day VaR at 95 and 99
    alphas = (0.95, 0.99)
    normal_vars = [var_1day_parametric_normal(mu, sigma, a, portfolio_value) for a in alphas]
    t_vars = [var_1day_parametric_t(loc, scale, df, a, portfolio_value) for a in alphas]

    # results table for Excel
    rows = []
    for d in normal_vars:
        rows.append({"dist": "normal", "alpha": d["alpha"], "q_return": d["q_return"], "VaR_$": d["VaR_$"]})
    for d in t_vars:
        rows.append({"dist": "student_t", "alpha": d["alpha"], "q_return": d["q_return"], "VaR_$": d["VaR_$"]})
    results_df = pd.DataFrame(rows).sort_values(["dist", "alpha"]).reset_index(drop=True)

    # Select a single VaR to show as “headline” (Student-t at requested confidence_level)
    # Snap confidence to nearest of {0.95, 0.99} unless you extend logic
    target_alpha = 0.99 if float(confidence_level) >= 0.975 else 0.95
    selected_var = float(results_df[(results_df["dist"] == "student_t") & (results_df["alpha"] == target_alpha)]["VaR_$"].iloc[0])

    # ---- Plot: histogram + PDFs + VaR lines ----
    q_candidates = [d["q_return"] for d in normal_vars + t_vars]
    x_min = min(r.min(), min(q_candidates)) - 0.01
    x_max = max(r.max(), mu + 4 * sigma) + 0.01
    x = np.linspace(x_min, x_max, 2000)

    pdf_n = normal_pdf(x, mu, sigma)
    pdf_t = student_t_pdf(x, df=df, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r, bins=60, density=True, alpha=0.5, label="Historical daily returns")
    ax.plot(x, pdf_n, linewidth=2, label="Normal fit")
    ax.plot(x, pdf_t, linewidth=2, label=f"Student-t fit (df={df:.2f})")

    styles = {
        ("normal", 0.95): ("--", "Normal 95% VaR cutoff"),
        ("normal", 0.99): (":", "Normal 99% VaR cutoff"),
        ("t", 0.95): ("-.", "Student-t 95% VaR cutoff"),
        ("t", 0.99): ("-", "Student-t 99% VaR cutoff"),
    }

    for dct in normal_vars:
        ls, lab = styles[("normal", dct["alpha"])]
        ax.axvline(dct["q_return"], linestyle=ls, linewidth=2, label=lab)

    for dct in t_vars:
        ls, lab = styles[("t", dct["alpha"])]
        ax.axvline(dct["q_return"], linestyle=ls, linewidth=2, label=lab)

    ax.set_title("Daily Portfolio Returns: Historical vs Normal vs Student-t (with 1-day VaR cutoffs)")
    ax.set_xlabel("Daily portfolio return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    return results_df, fig, selected_var
