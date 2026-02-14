# Portfolio Analytics Engine (Market Risk Engine)

A multi-asset market risk analytics prototype built during a Spring Week placement at **S&P Global Financial Risk Analytics**.  
It ingests a portfolio of instruments, pulls market data from **Yahoo Finance (free API)**, and runs a suite of VaR models and stress-testing workflows via a local Flask web app—each with accompanying visualisations and (for some) preset historical scenarios (e.g., COVID crash, 2008).

> Note: Built in ~2 days as a prototype for **single-portfolio** analysis (typically **< 20s** runtime). Not performance-optimised for large-scale production use.

---

## What it ingests (Yahoo Finance)

The engine can fetch pricing/market series for:
- **Equities & ETFs**
- **Market indices**
- **Commodities**
- **FX (forex) pairs**
- **Cryptocurrencies**
- **Futures tickers**
- **Major bond yields**

You provide a portfolio (tickers + weights), and the app fetches data and runs the selected analytics.

---

## Analytics included (8 modules)

1. **Historical VaR**
   - 95% and 99% confidence

2. **Parametric VaR**
   - Normal and Student-t fits  
   - 1-day VaR at 95% and 99%

3. **Monte Carlo VaR (Gaussian)**
   - Assumes normally distributed returns  
   - User-adjustable:
     - Lookback window for return distribution
     - Number of simulations
     - VaR horizon (days)
     - Confidence levels: **90, 95, 97.5, 99, 99.5, 99.9**

4. **Monte Carlo VaR (Student-t)**
   - Heavy-tailed returns
   - Same adjustable parameters as Gaussian MC

5. **Monte Carlo VaR (Copula)**
   - Dependence structure modeled via copula
   - Most computationally expensive (typically renders last)

6. **Pure Stress Test (Asset Shocks)**
   - Apply user-defined percentage shocks to selected assets
   - Supports shocking multiple assets simultaneously

7. **Factor Stress Test**
   - Infers factor exposures (betas) over a user-defined period
   - Applies shocks to factors and propagates impact to:
     - Individual components (detailed P&L profile)
     - Total portfolio P&L

8. **Historical Stress Test (Scenario Replay)**
   - Replays portfolio evolution over a user-selected historical window
   - Includes preset scenarios (e.g., COVID, 2008)

---

## Screenshots

### Parametric + Historical VaR
![Parametric + Historical VaR](images/Parametric%20%2B%20Historical%20VaR.png)

### Student-t + Copula Monte Carlo VaR (1Y Horizon)
![Student + Copula MC VaR 1y](images/Student%20%2B%20Copula%20MC%20VaR%201y.png)

### Historical + Factor Stress Test
![Historical + Factor Stress Test](images/Historical%20%2B%20Factor%20Stress%20Test.png)

---

## Local Web App (Flask)

A Flask-powered local website that wraps the `Market_Risk_spglobal` Python library and lets you run the analytics from a browser.

---

## Setup

### 1) Install dependencies

```bash
pip install flask yfinance pandas numpy matplotlib certifi mpmath

Or:

pip install -r market_risk_app/requirements.txt

Run the app
cd market_risk_app
python app.py


Open: http://127.0.0.1:5000

Using the interface

Portfolio: enter comma-separated tickers and matching weights (weights should sum to 1)

Parameters: choose confidence level, time horizon, history window, MC simulations

Simulations: toggle one or more modules to run

Run analysis: fetches live data and runs selected models

Outputs include plots (distributions/scenarios) and results tables, including component-level P&L breakdowns where applicable.

Developer notes (wrappers)

The web app expects callable run_* wrappers in the analytics scripts (e.g., run_bin_t_student_var()), returning results tables and matplotlib figures for embedding in the UI. Add wrappers to any scripts that do not yet expose a run_* entry point.

Troubleshooting
Problem	Fix
ModuleNotFoundError	Confirm LIBRARY_ROOT and folder layout.
SSL errors	Ensure cert handling in set_certifi.py / daily_closes.py.
ImportError: cannot import name run_*	Add a run_* wrapper to the referenced script.
Port 5000 already in use	Change the port in app.py (e.g., 5001).
Blank chart	Ensure the script returns a valid matplotlib Figure.
Disclaimer

Educational prototype built during a short placement project. Outputs are for demonstration/learning only and are not intended for production risk management or investment decisions.

---
