"""
Market Risk Engine - Flask Web Application
Wraps the Market_Risk_spglobal Python library and exposes a local web UI.
"""

import sys
import os
import io
import base64
import json
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ── Path setup ─────────────────────────────────────────────────────────────────
# Adjust this to point at your actual Market_Risk_spglobal folder
LIBRARY_ROOT = Path(r"C:\Users\franc\Downloads\VSC\Portfolio_Analytics_Engine\Market_Risk_spglobal")
sys.path.insert(0, str(LIBRARY_ROOT))
sys.path.insert(0, str(LIBRARY_ROOT / "Monte_Carlo VaR"))
sys.path.insert(0, str(LIBRARY_ROOT / "Stress_Testing"))

# Also change working directory so relative imports inside your scripts work
os.chdir(str(LIBRARY_ROOT))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def parse_portfolio(tickers_raw: str, weights_raw: str):
    """
    Parse comma-separated tickers and weights from form data.
    Returns (tickers list, weights dict, weights list).
    """
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    weights = [float(w.strip()) for w in weights_raw.split(",") if w.strip()]
    if len(tickers) != len(weights):
        raise ValueError(f"Tickers ({len(tickers)}) and weights ({len(weights)}) counts differ.")
    weights_dict = {t: w for t, w in zip(tickers, weights)}
    return tickers, weights_dict, weights


def _portfolio_df(tickers, weights_list):
    """Build the DataFrame that run_* functions expect."""
    return pd.DataFrame({"ticker": tickers, "weight": weights_list})


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run_calculation():
    data = request.get_json(force=True)

    tickers_raw  = data.get("tickers", "")
    weights_raw  = data.get("weights", "")
    sim_type     = data.get("simulation", "")
    hist_var_start = data.get("hist_var_start", "")
    hist_var_end   = data.get("hist_var_end", "")
    confidence   = float(data.get("confidence", 0.95))
    horizon      = int(data.get("horizon", 1))
    start_year   = int(data.get("start_year", 2018))
    end_year     = int(data.get("end_year", 2024))
    port_value   = float(data.get("portfolio_value", 1_000_000))
    days_back    = int(data.get("days_back", 365 * 5))
    n_sims       = int(data.get("n_sims", 10_000))

    try:
        tickers, weights_dict, weights_list = parse_portfolio(tickers_raw, weights_raw)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)})

    try:
        result = _dispatch(
            sim_type, tickers, weights_dict, weights_list,
            confidence, horizon, start_year, end_year,
            port_value, days_back, n_sims, data, 
            hist_var_start, hist_var_end  # ADD 'data' here
        )
        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"success": False, "error": str(e), "traceback": tb})


def _dispatch(sim_type, tickers, weights_dict, weights_list,
              confidence, horizon, start_year, end_year,
              port_value, days_back, n_sims, request_data,
              hist_var_start="", hist_var_end=""):
    os.chdir(str(LIBRARY_ROOT))

    if sim_type == "historical_var":
        from historical_var import (
            get_portfolio_returns_from_daily_closes,
            historical_var_1day,
        )
        import datetime as dt

        # Use date range from UI if provided, otherwise fall back to 3 years
        if hist_var_start and hist_var_end:
            try:
                start_dt = dt.datetime.strptime(hist_var_start, "%Y-%m-%d")
                end_dt   = dt.datetime.strptime(hist_var_end,   "%Y-%m-%d")
                years_back = (end_dt - start_dt).days / 365.0
            except ValueError:
                years_back = 3.0
        else:
            years_back = 3.0

        years_back = max(0.1, years_back)   # safety floor

        pr = get_portfolio_returns_from_daily_closes(tickers, weights_dict, years_back)
        r  = pr.to_numpy(dtype=float)
        r  = r[np.isfinite(r)]

        res   = historical_var_1day(pr, port_value)
        var95 = res["VaR_95_$"]
        var99 = res["VaR_99_$"]

        # Date range label for the chart title
        date_label = f"{hist_var_start} → {hist_var_end}" if hist_var_start and hist_var_end \
                    else f"{years_back:.1f} years"

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        ax.hist(r, bins=60, density=True, color="#1e88e5", alpha=0.55, label="Daily returns")
        ax.axvline(-var95 / port_value, color="#f59e0b", lw=2, ls="--",
                label=f"95% VaR ({-var95/port_value:.2%})")
        ax.axvline(-var99 / port_value, color="#ef4444", lw=2, ls=":",
                label=f"99% VaR ({-var99/port_value:.2%})")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.set_xlabel("Daily return", color="#8b949e")
        ax.set_ylabel("Density", color="#8b949e")
        ax.set_title(f"Historical VaR – {date_label}  ({len(r)} obs)",
                    color="#e6edf3", fontsize=13, pad=10)
        leg = ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        fig.tight_layout()

        table = [
            {"Metric": "95% Historical VaR (1-day)", "Value": f"${var95:,.0f}",
            "Return": f"{-var95/port_value:.4%}"},
            {"Metric": "99% Historical VaR (1-day)", "Value": f"${var99:,.0f}",
            "Return": f"{-var99/port_value:.4%}"},
            {"Metric": "Observations",               "Value": str(len(r)),      "Return": ""},
            {"Metric": "History Window",             "Value": date_label,       "Return": ""},
            {"Metric": "Portfolio Value",            "Value": f"${port_value:,.0f}", "Return": ""},
        ]

        return {
            "success": True,
            "title": "Historical VaR",
            "headline_var": f"${var95:,.0f}",
            "headline_label": "95% Historical VaR (1-day)",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 2. Parametric (Normal + Student-t) VaR ────────────────────────────────
    elif sim_type == "parametric_var":
        from bin_t_stu_var import run_bin_t_student_var
        pf_df = _portfolio_df(tickers, weights_list)

        results_df, fig, selected_var = run_bin_t_student_var(
            portfolio_df=pf_df,
            confidence_level=confidence,
            horizon_days=horizon,
            start_year=start_year,
            end_year=end_year,
            portfolio_value=port_value,
        )

        # Style the existing figure
        fig.patch.set_facecolor("#0d1117")
        for ax in fig.get_axes():
            ax.set_facecolor("#0d1117")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.tick_params(colors="#8b949e")
            ax.xaxis.label.set_color("#8b949e")
            ax.yaxis.label.set_color("#8b949e")
            ax.title.set_color("#e6edf3")
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor("#161b22")
                leg.get_frame().set_edgecolor("#30363d")
                for t in leg.get_texts():
                    t.set_color("#e6edf3")

        table = []
        for _, row in results_df.iterrows():
            table.append({
                "Distribution": row["dist"].title().replace("_", "-"),
                "Confidence":   f"{int(row['alpha']*100)}%",
                "Return Cutoff": f"{row['q_return']:.4%}",
                "VaR ($)":       f"${row['VaR_$']:,.0f}",
            })

        return {
            "success": True,
            "title": "Parametric VaR (Normal + Student-t)",
            "headline_var": f"${selected_var:,.0f}",
            "headline_label": f"Student-t VaR @ {int(confidence*100)}% confidence",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 3. Monte Carlo – Gaussian ──────────────────────────────────────────────
    elif sim_type == "mc_gaussian":
        from mc_var_gaussian import run_mc_var_gaussian
        pf_df = _portfolio_df(tickers, weights_list)
        results_df, fig, headline = run_mc_var_gaussian(
            portfolio_df=pf_df,
            confidence_level=confidence,
            horizon_days=horizon,
            start_year=start_year,
            end_year=end_year,
            portfolio_value=port_value,
            n_simulations=n_sims,
        )
        _style_fig(fig)
        table = results_df.to_dict(orient="records")
        return {
            "success": True,
            "title": "Monte Carlo VaR – Gaussian",
            "headline_var": f"${headline:,.0f}" if isinstance(headline, float) else str(headline),
            "headline_label": f"MC Gaussian VaR @ {int(confidence*100)}%",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 4. Monte Carlo – Student-t ─────────────────────────────────────────────
    elif sim_type == "mc_student":
        from mc_var_student import run_mc_var_student
        pf_df = _portfolio_df(tickers, weights_list)
        results_df, fig, headline = run_mc_var_student(
            portfolio_df=pf_df,
            confidence_level=confidence,
            horizon_days=horizon,
            start_year=start_year,
            end_year=end_year,
            portfolio_value=port_value,
            n_simulations=n_sims,
        )
        _style_fig(fig)
        table = results_df.to_dict(orient="records")
        return {
            "success": True,
            "title": "Monte Carlo VaR – Student-t",
            "headline_var": f"${headline:,.0f}" if isinstance(headline, float) else str(headline),
            "headline_label": f"MC Student-t VaR @ {int(confidence*100)}%",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 5. Monte Carlo – Copula ────────────────────────────────────────────────
    elif sim_type == "mc_copula":
        from mc_var_copula import run_mc_var_copula
        pf_df = _portfolio_df(tickers, weights_list)
        results_df, fig, headline = run_mc_var_copula(
            portfolio_df=pf_df,
            confidence_level=confidence,
            horizon_days=horizon,
            start_year=start_year,
            end_year=end_year,
            portfolio_value=port_value,
            n_simulations=n_sims,
        )
        _style_fig(fig)
        table = results_df.to_dict(orient="records")
        return {
            "success": True,
            "title": "Monte Carlo VaR – Copula",
            "headline_var": f"${headline:,.0f}" if isinstance(headline, float) else str(headline),
            "headline_label": f"MC Copula VaR @ {int(confidence*100)}%",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 6. Pure Stress Test ────────────────────────────────────────────────────
# ── 6. Pure Stress Test ────────────────────────────────────────
    elif sim_type == "stress_pure":
        from stress_test_pure import pure_hypothetical_stress_test
        
        # Get ticker shocks from request
        pure_shocks_json = request_data.get("pure_shocks", "{}")  # CHANGED from factor_shocks
        try:
            pure_shocks_dict = json.loads(pure_shocks_json)
            # Convert shock percentages to decimals
            ticker_shocks = {k: float(v) / 100.0 for k, v in pure_shocks_dict.items()}
        except:
            ticker_shocks = {}
        
        default_shock = float(request_data.get("default_shock", -15)) / 100.0
        
        # Fetch current prices for portfolio tickers
        from daily_closes import download_daily_closes
        try:
            closes = download_daily_closes(tickers, days_back=5)
            if closes.empty or 'Date' not in closes.columns:
                return {"success": False, "error": "Could not fetch current prices for portfolio tickers."}
            
            closes['Date'] = pd.to_datetime(closes['Date'], errors='coerce')
            closes = closes.dropna(subset=['Date']).sort_values('Date')
            latest = closes.iloc[-1]
            
            prices = {}
            for t in tickers:
                if t in closes.columns:
                    prices[t] = float(latest[t])
                else:
                    return {"success": False, "error": f"Could not fetch price for {t}"}
        except Exception as e:
            return {"success": False, "error": f"Error fetching prices: {str(e)}"}
        
        # Build shocks dict: use factor shocks if ticker is in factor_shocks, else use default
        shocks = {}
        for t in tickers:
            if t in ticker_shocks:
                shocks[t] = ticker_shocks[t]
            # else: leave empty to use default_shock
        
        # Convert portfolio weights to shares based on portfolio value
        holdings = {}
        for t, w in zip(tickers, weights_list):
            if t in prices:
                position_value = port_value * w
                holdings[t] = position_value / prices[t]
        
        # Run stress test
        detail, summary = pure_hypothetical_stress_test(prices, holdings, shocks, default_shock)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        detail_ok = detail[detail["Status"] == "OK"].copy()
        if detail_ok.empty:
            return {"success": False, "error": "No valid positions to stress test"}
        
        detail_sorted = detail_ok.sort_values("PnL")
        
        colors = ['#ef4444' if x < 0 else '#3fb950' for x in detail_sorted["PnL"]]
        ax.barh(detail_sorted["Ticker"], detail_sorted["PnL"], color=colors, alpha=0.7)
        ax.axvline(0, color='#8b949e', linestyle='--', linewidth=1)
        ax.set_xlabel("P&L ($)", color='#8b949e')
        ax.set_ylabel("Ticker", color='#8b949e')
        ax.set_title("Pure Stress Test - Position-Level P&L", color='#e6edf3', fontsize=13, pad=10)
        
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.grid(axis='x', alpha=0.2, color='#30363d')
        fig.tight_layout()
        
        # Build table
        table = []
        for _, row in detail_sorted.iterrows():
            table.append({
                "Ticker": row["Ticker"],
                "Shares": f"{row['Shares']:.2f}",
                "Price": f"${row['Price']:.2f}",
                "Shock": f"{row['ShockReturn']:.1%}",
                "Base Value": f"${row['BaseValue']:,.0f}",
                "Stressed Value": f"${row['StressedValue']:,.0f}",
                "P&L": f"${row['PnL']:,.0f}",
            })
        
        total_pnl = summary['PnL'].iloc[0] if not summary.empty else 0
        total_pnl_pct = summary['PnL_pct'].iloc[0] if not summary.empty else 0
        
        return {
            "success": True,
            "title": "Pure Stress Test",
            "headline_var": f"${total_pnl:,.0f}" + (f" ({total_pnl_pct:.1%})" if total_pnl_pct else ""),
            "headline_label": "Total Portfolio P&L",
            "table": table,
            "chart": fig_to_base64(fig),
        }

    # ── 7. Factor Stress Test ──────────────────────────────────────────────────
    elif sim_type == "stress_factor":
        # Parse factor configuration
        factor_config_json = request_data.get("factor_config", "{}")
        beta_start_year = int(request_data.get("beta_start_year", 2021))
        beta_end_year = int(request_data.get("beta_end_year", 2024))
        
        try:
            factor_config = json.loads(factor_config_json)
            factors_dict = factor_config.get("factors", {})  # e.g., {"MKT": "SPY", "SEMI": "SMH"}
            shocks_dict = factor_config.get("shocks", {})    # e.g., {"MKT": -20, "SEMI": -25}
            
            # Convert shock percentages to decimals
            factor_shocks = {k: float(v) / 100.0 for k, v in shocks_dict.items()}
        except Exception as e:
            return {"success": False, "error": f"Error parsing factor configuration: {str(e)}"}
        
        if not factors_dict or not factor_shocks:
            return {"success": False, "error": "Please specify at least one factor with its proxy and shock"}
        
        # Import and run factor stress test
        try:
            # We need to create a wrapper function that matches your factor_stress_test.py
            # Since your current code has hardcoded configs, we'll need to adapt it
            
            from pathlib import Path
            import sys
            
            # Temporary: write a dynamic config and run
            # Better approach: refactor factor_stress_test.py to accept parameters
            
            # For now, let's create a simplified version that calls the core logic
            from daily_closes import download_daily_closes
            import yfinance as yf
            
            # 1. Download asset prices for portfolio tickers
            lookback_days = (beta_end_year - beta_start_year + 1) * 365 + 60
            asset_closes = download_daily_closes(tickers, days_back=lookback_days)
            
            if asset_closes.empty or 'Date' not in asset_closes.columns:
                return {"success": False, "error": "Could not download asset price history"}
            
            asset_closes['Date'] = pd.to_datetime(asset_closes['Date'], errors='coerce')
            asset_closes = asset_closes.dropna(subset=['Date']).sort_values('Date').set_index('Date')
            
            # Filter to beta estimation window
            start_dt = pd.Timestamp(f"{beta_start_year}-01-01")
            end_dt = pd.Timestamp(f"{beta_end_year}-12-31")
            asset_closes = asset_closes.loc[start_dt:end_dt]
            
            # 2. Download factor prices
            factor_tickers = list(factors_dict.values())
            factor_data = yf.download(
                tickers=factor_tickers,
                start=(start_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                end=(end_dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            
            if factor_data.empty:
                return {"success": False, "error": "Could not download factor proxy data"}
            
            factor_closes = factor_data["Close"]
            if isinstance(factor_closes, pd.Series):
                factor_closes = factor_closes.to_frame()
            factor_closes.index = pd.to_datetime(factor_closes.index)
            factor_closes = factor_closes.sort_index()
            
            # Rename columns to factor names (MKT, SEMI, etc.)
            rename_map = {v: k for k, v in factors_dict.items()}
            factor_closes = factor_closes.rename(columns=rename_map)
            
            # 3. Compute returns
            asset_rets = asset_closes.pct_change().dropna(how='all')
            factor_rets = factor_closes.pct_change().dropna(how='all')
            
            # Align dates
            common_index = asset_rets.index.intersection(factor_rets.index)
            asset_rets = asset_rets.loc[common_index]
            factor_rets = factor_rets.loc[common_index]
            
            if len(common_index) < 60:
                return {"success": False, "error": f"Not enough overlapping data: {len(common_index)} days"}
            
            # 4. Estimate betas via OLS for each asset
            betas_data = []
            for asset in tickers:
                if asset not in asset_rets.columns:
                    continue
                    
                y = asset_rets[asset].dropna()
                X = factor_rets.loc[y.index, list(factors_dict.keys())].dropna(how='any')
                
                # Align again
                idx = y.index.intersection(X.index)
                y = y.loc[idx]
                X = X.loc[idx]
                
                if len(y) < 30:
                    continue
                
                # OLS regression
                yv = y.to_numpy(dtype=float)
                Xv = X.to_numpy(dtype=float)
                Xv = np.column_stack([np.ones(len(yv)), Xv])  # add intercept
                
                try:
                    coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
                    betas = pd.Series(coef[1:], index=list(factors_dict.keys()), dtype=float)
                    
                    # Calculate R²
                    yhat = Xv @ coef
                    ss_res = float(np.sum((yv - yhat) ** 2))
                    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    
                    betas_data.append({"Asset": asset, "R2": r2, **betas.to_dict()})
                except:
                    continue
            
            if not betas_data:
                return {"success": False, "error": "Could not estimate betas for any assets"}
            
            betas_df = pd.DataFrame(betas_data).set_index("Asset")
            
            # 5. Apply factor shocks to compute stressed returns
            shocks_vec = pd.Series(factor_shocks, dtype=float)
            stressed_returns = betas_df[list(factors_dict.keys())].mul(shocks_vec, axis=1).sum(axis=1)
            
            # 6. Get current prices and compute P&L
            current_prices_df = download_daily_closes(tickers, days_back=5)
            if current_prices_df.empty or 'Date' not in current_prices_df.columns:
                return {"success": False, "error": "Could not fetch current prices"}
            
            current_prices_df['Date'] = pd.to_datetime(current_prices_df['Date'], errors='coerce')
            current_prices_df = current_prices_df.dropna(subset=['Date']).sort_values('Date')
            latest_row = current_prices_df.iloc[-1]
            
            current_prices = {}
            for t in tickers:
                if t in current_prices_df.columns and t in stressed_returns.index:
                    current_prices[t] = float(latest_row[t])
            
            # Calculate position values and P&L
            results_data = []
            for t in tickers:
                if t not in current_prices or t not in stressed_returns.index:
                    continue
                
                weight = weights_dict[t]
                base_value = port_value * weight
                shares = base_value / current_prices[t]
                
                stressed_ret = stressed_returns[t]
                pnl = base_value * stressed_ret
                stressed_value = base_value + pnl
                
                # Get betas for display
                beta_vals = betas_df.loc[t, list(factors_dict.keys())].to_dict()
                
                results_data.append({
                    "Ticker": t,
                    "Price": current_prices[t],
                    "Shares": shares,
                    "BaseValue": base_value,
                    "StressedReturn": stressed_ret,
                    "PnL": pnl,
                    "StressedValue": stressed_value,
                    **{f"Beta_{k}": v for k, v in beta_vals.items()}
                })
            
            if not results_data:
                return {"success": False, "error": "No valid stress test results"}
            
            results_df = pd.DataFrame(results_data)
            
            # 7. Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Chart 1: P&L by position
            results_sorted = results_df.sort_values("PnL")
            colors = ['#ef4444' if x < 0 else '#3fb950' for x in results_sorted["PnL"]]
            ax1.barh(results_sorted["Ticker"], results_sorted["PnL"], color=colors, alpha=0.7)
            ax1.axvline(0, color='#8b949e', linestyle='--', linewidth=1)
            ax1.set_xlabel("P&L ($)", color='#8b949e')
            ax1.set_ylabel("Ticker", color='#8b949e')
            ax1.set_title("Factor Stress Test - Position P&L", color='#e6edf3', fontsize=12)
            ax1.grid(axis='x', alpha=0.2, color='#30363d')
            
            # Chart 2: Beta exposure heatmap (simplified as bars for first factor)
            first_factor = list(factors_dict.keys())[0]
            beta_col = f"Beta_{first_factor}"
            if beta_col in results_df.columns:
                beta_sorted = results_df.sort_values(beta_col)
                colors_beta = ['#1e88e5' if x > 0 else '#f59e0b' for x in beta_sorted[beta_col]]
                ax2.barh(beta_sorted["Ticker"], beta_sorted[beta_col], color=colors_beta, alpha=0.7)
                ax2.axvline(0, color='#8b949e', linestyle='--', linewidth=1)
                ax2.set_xlabel(f"Beta to {first_factor}", color='#8b949e')
                ax2.set_ylabel("Ticker", color='#8b949e')
                ax2.set_title(f"Factor Exposures ({first_factor})", color='#e6edf3', fontsize=12)
                ax2.grid(axis='x', alpha=0.2, color='#30363d')
            
            _style_fig(fig)
            fig.tight_layout()
            
            # 8. Build table for display
            table = []
            for _, row in results_sorted.iterrows():
                table_row = {
                    "Ticker": row["Ticker"],
                    "Base Value": f"${row['BaseValue']:,.0f}",
                    "Stressed Return": f"{row['StressedReturn']:.2%}",
                    "P&L": f"${row['PnL']:,.0f}",
                }
                # Add beta columns
                for factor_name in factors_dict.keys():
                    beta_col = f"Beta_{factor_name}"
                    if beta_col in row:
                        table_row[f"β_{factor_name}"] = f"{row[beta_col]:.3f}"
                table.append(table_row)
            
            total_pnl = results_df["PnL"].sum()
            total_base = results_df["BaseValue"].sum()
            total_ret = total_pnl / total_base if total_base > 0 else 0
            
            return {
                "success": True,
                "title": "Factor Stress Test",
                "headline_var": f"${total_pnl:,.0f} ({total_ret:.2%})",
                "headline_label": "Total Portfolio P&L",
                "table": table,
                "chart": fig_to_base64(fig),
            }
            
        except Exception as e:
            tb = traceback.format_exc()
            return {"success": False, "error": f"Factor stress test failed: {str(e)}", "traceback": tb}

    # ── 8. Factor Historical Stress Test ──────────────────────────────────────
    elif sim_type == "stress_factor_historical":
        # Parse historical stress configuration
        historical_factors_json = request_data.get("historical_factors", "{}")
        stress_start_date = request_data.get("stress_start_date", "2020-02-19")
        stress_end_date = request_data.get("stress_end_date", "2020-03-23")
        hist_beta_start_year = int(request_data.get("hist_beta_start_year", 2021))
        hist_beta_end_year = int(request_data.get("hist_beta_end_year", 2024))
        
        try:
            historical_factors = json.loads(historical_factors_json)
        except Exception as e:
            return {"success": False, "error": f"Error parsing historical factors: {str(e)}"}
        
        if not historical_factors:
            return {"success": False, "error": "Please specify at least one factor with its proxy"}
        
        try:
            from daily_closes import download_daily_closes
            import yfinance as yf
            
            # 1. Download asset prices for beta estimation period
            beta_lookback_days = (hist_beta_end_year - hist_beta_start_year + 1) * 365 + 60
            asset_closes_beta = download_daily_closes(tickers, days_back=beta_lookback_days)
            
            if asset_closes_beta.empty or 'Date' not in asset_closes_beta.columns:
                return {"success": False, "error": "Could not download asset price history for beta estimation"}
            
            asset_closes_beta['Date'] = pd.to_datetime(asset_closes_beta['Date'], errors='coerce')
            asset_closes_beta = asset_closes_beta.dropna(subset=['Date']).sort_values('Date').set_index('Date')
            
            # Filter to beta estimation window
            beta_start_dt = pd.Timestamp(f"{hist_beta_start_year}-01-01")
            beta_end_dt = pd.Timestamp(f"{hist_beta_end_year}-12-31")
            asset_closes_beta = asset_closes_beta.loc[beta_start_dt:beta_end_dt]
            
            # 2. Download factor prices for beta estimation
            factor_tickers = list(historical_factors.values())
            factor_data_beta = yf.download(
                tickers=factor_tickers,
                start=(beta_start_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                end=(beta_end_dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            
            if factor_data_beta.empty:
                return {"success": False, "error": "Could not download factor proxy data for beta estimation"}
            
            factor_closes_beta = factor_data_beta["Close"]
            if isinstance(factor_closes_beta, pd.Series):
                factor_closes_beta = factor_closes_beta.to_frame()
            factor_closes_beta.index = pd.to_datetime(factor_closes_beta.index)
            factor_closes_beta = factor_closes_beta.sort_index()
            
            # Rename columns to factor names
            rename_map = {v: k for k, v in historical_factors.items()}
            factor_closes_beta = factor_closes_beta.rename(columns=rename_map)
            
            # 3. Estimate betas
            asset_rets_beta = asset_closes_beta.pct_change().dropna(how='all')
            factor_rets_beta = factor_closes_beta.pct_change().dropna(how='all')
            
            common_beta = asset_rets_beta.index.intersection(factor_rets_beta.index)
            asset_rets_beta = asset_rets_beta.loc[common_beta]
            factor_rets_beta = factor_rets_beta.loc[common_beta]
            
            if len(common_beta) < 60:
                return {"success": False, "error": f"Not enough beta estimation data: {len(common_beta)} days"}
            
            betas_data = []
            for asset in tickers:
                if asset not in asset_rets_beta.columns:
                    continue
                    
                y = asset_rets_beta[asset].dropna()
                X = factor_rets_beta.loc[y.index, list(historical_factors.keys())].dropna(how='any')
                
                idx = y.index.intersection(X.index)
                y = y.loc[idx]
                X = X.loc[idx]
                
                if len(y) < 30:
                    continue
                
                yv = y.to_numpy(dtype=float)
                Xv = X.to_numpy(dtype=float)
                Xv = np.column_stack([np.ones(len(yv)), Xv])
                
                try:
                    coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
                    betas = pd.Series(coef[1:], index=list(historical_factors.keys()), dtype=float)
                    
                    yhat = Xv @ coef
                    ss_res = float(np.sum((yv - yhat) ** 2))
                    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    
                    betas_data.append({"Asset": asset, "R2": r2, **betas.to_dict()})
                except:
                    continue
            
            if not betas_data:
                return {"success": False, "error": "Could not estimate betas for any assets"}
            
            betas_df = pd.DataFrame(betas_data).set_index("Asset")
            
            # 4. Download factor prices for historical stress period
            stress_start_dt = pd.Timestamp(stress_start_date)
            stress_end_dt = pd.Timestamp(stress_end_date)
            
            factor_data_stress = yf.download(
                tickers=factor_tickers,
                start=(stress_start_dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                end=(stress_end_dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            
            if factor_data_stress.empty:
                return {"success": False, "error": "Could not download factor data for stress period"}
            
            factor_closes_stress = factor_data_stress["Close"]
            if isinstance(factor_closes_stress, pd.Series):
                factor_closes_stress = factor_closes_stress.to_frame()
            factor_closes_stress.index = pd.to_datetime(factor_closes_stress.index)
            factor_closes_stress = factor_closes_stress.sort_index()
            factor_closes_stress = factor_closes_stress.rename(columns=rename_map)
            
            # Filter to stress period
            factor_closes_stress = factor_closes_stress.loc[stress_start_dt:stress_end_dt]
            
            if len(factor_closes_stress) < 2:
                return {"success": False, "error": "Not enough factor data in stress period"}
            
            # 5. Compute daily factor returns during stress period
            factor_rets_stress = factor_closes_stress.pct_change().dropna(how='all')
            
            # 6. Compute implied asset returns: r_asset = sum(beta_factor * r_factor)
            betas_only = betas_df[list(historical_factors.keys())]
            implied_asset_rets = factor_rets_stress @ betas_only.T  # dates x assets
            
            # 7. Get current prices
            current_prices_df = download_daily_closes(tickers, days_back=5)
            if current_prices_df.empty or 'Date' not in current_prices_df.columns:
                return {"success": False, "error": "Could not fetch current prices"}
            
            current_prices_df['Date'] = pd.to_datetime(current_prices_df['Date'], errors='coerce')
            current_prices_df = current_prices_df.dropna(subset=['Date']).sort_values('Date')
            latest_row = current_prices_df.iloc[-1]
            
            current_prices = {}
            for t in tickers:
                if t in current_prices_df.columns and t in implied_asset_rets.columns:
                    current_prices[t] = float(latest_row[t])
            
            # 8. Compute base values and daily portfolio values
            base_values = {}
            for t in tickers:
                if t in current_prices:
                    weight = weights_dict[t]
                    base_values[t] = port_value * weight
            
            base_values_series = pd.Series(base_values)
            
            # Portfolio value path: start at port_value, apply daily returns
            portfolio_values = [port_value]
            asset_values = {t: [base_values[t]] for t in base_values.keys()}
            
            current_port_value = port_value
            current_asset_values = base_values.copy()
            
            for date, returns_row in implied_asset_rets.iterrows():
                # Apply returns to each asset
                for t in base_values.keys():
                    if t in returns_row.index and pd.notna(returns_row[t]):
                        current_asset_values[t] *= (1 + returns_row[t])
                        asset_values[t].append(current_asset_values[t])
                
                # Portfolio value is sum of all assets
                current_port_value = sum(current_asset_values.values())
                portfolio_values.append(current_port_value)
            
            # 9. Create visualization: Portfolio + Individual Asset Evolution
            dates_for_plot = [stress_start_dt] + list(implied_asset_rets.index)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Chart 1: Portfolio Value Evolution
            ax1.plot(dates_for_plot, portfolio_values, linewidth=2.5, color='#1e88e5', label='Total Portfolio')
            ax1.axhline(port_value, color='#8b949e', linestyle='--', linewidth=1, label='Initial Value')
            ax1.fill_between(dates_for_plot, portfolio_values, port_value, 
                            where=[v < port_value for v in portfolio_values],
                            color='#ef4444', alpha=0.2, label='Drawdown')
            ax1.set_ylabel("Portfolio Value ($)", color='#8b949e')
            ax1.set_title(f"Historical Stress Test: {stress_start_date} to {stress_end_date}", 
                        color='#e6edf3', fontsize=13, pad=10)
            ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
            ax1.grid(alpha=0.2, color='#30363d')
            
            # Format y-axis as currency
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Chart 2: Individual Asset Values Evolution
            colors_assets = ['#1e88e5', '#3fb950', '#f59e0b', '#ef4444', '#8b5cf6']
            for i, (ticker, values) in enumerate(asset_values.items()):
                color = colors_assets[i % len(colors_assets)]
                ax2.plot(dates_for_plot, values, linewidth=2, label=ticker, color=color)
            
            ax2.set_xlabel("Date", color='#8b949e')
            ax2.set_ylabel("Position Value ($)", color='#8b949e')
            ax2.set_title("Individual Position Values", color='#e6edf3', fontsize=12)
            ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
            ax2.grid(alpha=0.2, color='#30363d')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Format x-axis dates
            import matplotlib.dates as mdates
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            _style_fig(fig)
            fig.tight_layout()
            
            # 10. Build results table
            final_pnl = {}
            for t in base_values.keys():
                final_value = asset_values[t][-1]
                pnl = final_value - base_values[t]
                final_pnl[t] = pnl
            
            total_final_value = portfolio_values[-1]
            total_pnl = total_final_value - port_value
            total_return = total_pnl / port_value
            
            table = []
            for ticker in sorted(final_pnl.keys(), key=lambda t: final_pnl[t]):
                pnl = final_pnl[ticker]
                ret = pnl / base_values[ticker] if base_values[ticker] > 0 else 0
                
                table_row = {
                    "Ticker": ticker,
                    "Base Value": f"${base_values[ticker]:,.0f}",
                    "Final Value": f"${asset_values[ticker][-1]:,.0f}",
                    "P&L": f"${pnl:,.0f}",
                    "Return": f"{ret:.2%}",
                }
                
                # Add betas
                for factor_name in historical_factors.keys():
                    if ticker in betas_df.index:
                        beta_val = betas_df.loc[ticker, factor_name]
                        table_row[f"β_{factor_name}"] = f"{beta_val:.3f}"
                
                table.append(table_row)
            
            # Add summary row
            table.append({
                "Ticker": "TOTAL",
                "Base Value": f"${port_value:,.0f}",
                "Final Value": f"${total_final_value:,.0f}",
                "P&L": f"${total_pnl:,.0f}",
                "Return": f"{total_return:.2%}",
            })
            
            return {
                "success": True,
                "title": f"Historical Factor Stress Test ({stress_start_date} to {stress_end_date})",
                "headline_var": f"${total_pnl:,.0f} ({total_return:.2%})",
                "headline_label": "Total Portfolio P&L",
                "table": table,
                "chart": fig_to_base64(fig),
            }
            
        except Exception as e:
            tb = traceback.format_exc()
            return {"success": False, "error": f"Historical stress test failed: {str(e)}", "traceback": tb}


def _style_fig(fig):
    """Apply dark terminal theme to any matplotlib Figure in-place."""
    fig.patch.set_facecolor("#0d1117")
    for ax in fig.get_axes():
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")
        if ax.xaxis.label:
            ax.xaxis.label.set_color("#8b949e")
        if ax.yaxis.label:
            ax.yaxis.label.set_color("#8b949e")
        if ax.title:
            ax.title.set_color("#e6edf3")
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor("#161b22")
            leg.get_frame().set_edgecolor("#30363d")
            for t in leg.get_texts():
                t.set_color("#e6edf3")


if __name__ == "__main__":
    print("=" * 60)
    print("  Market Risk Engine  –  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)