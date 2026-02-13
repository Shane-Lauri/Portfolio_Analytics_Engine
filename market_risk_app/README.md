Readme · MD
Copy

# Market Risk Engine — Local Web App

A Flask-powered local website that wraps your `Market_Risk_spglobal` Python library
and lets you run VaR calculations and stress tests from any browser.

---

## Folder Structure

Place `market_risk_app/` **next to** (i.e. as a sibling of) `Market_Risk_spglobal/`:

```
C:\Users\franc\Downloads\VSC\AdSynthAI\
│
├── Market_Risk_spglobal\           ← YOUR EXISTING LIBRARY (unchanged)
│   ├── bin_t_stu_var.py
│   ├── daily_closes.py
│   ├── historical_var.py
│   ├── set_certifi.py
│   ├── Monte_Carlo VaR\
│   │   ├── mc_var_copula.py
│   │   ├── mc_var_gaussian.py
│   │   └── mc_var_student.py
│   └── Stress_Testing\
│       ├── factor_historical_stress_test.py
│       ├── factor_stress_test.py
│       └── stress_test_pure.py
│
└── market_risk_app\                ← THIS FOLDER
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── templates\
        └── index.html
```

> **Important:** `app.py` auto-detects the library path as `../Market_Risk_spglobal`.
> If you put this folder somewhere else, edit the `LIBRARY_ROOT` variable near the
> top of `app.py`:
> ```python
> LIBRARY_ROOT = Path(r"C:\path\to\Market_Risk_spglobal")
> ```

---

## Setup

### 1 – Install dependencies

```bash
pip install flask yfinance pandas numpy matplotlib certifi mpmath
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2 – Expose `run_*` functions in your library scripts

The web app calls a `run_<name>()` function in each of your scripts.
For `bin_t_stu_var.py` you already have `run_bin_t_student_var()` at the bottom.

For the Monte Carlo and Stress Testing scripts, add a similar wrapper if it doesn't
exist yet. Each wrapper should accept at minimum:

```python
def run_(
    portfolio_df: pd.DataFrame,   # columns: ticker, weight
    confidence_level: float,
    horizon_days: int,
    start_year: int,
    end_year: int,
    portfolio_value: float = 1_000_000.0,
    n_simulations: int = 10_000,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure, float]:
    """
    Returns:
        results_df  – DataFrame summarising the results
        fig         – matplotlib Figure (will be styled and embedded)
        headline    – single headline number (e.g. VaR at requested confidence)
    """
```

For the Stress Testing scripts the signature is slightly different (no `n_simulations`,
but add `days_back`):

```python
def run_(
    portfolio_df: pd.DataFrame,
    portfolio_value: float = 1_000_000.0,
    days_back: int = 1825,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    ...
```

### 3 – Add `historical_var.py` to `Market_Risk_spglobal/`

If you don't have one yet, use the stub provided (`historical_var_stub.py`) as a
starting point — rename it and drop it into your library folder.

---

## Running the App

```bash
cd market_risk_app
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Using the Interface

| Section | What to do |
|---|---|
| **Portfolio** | Enter comma-separated tickers and matching weights (must sum to 1). |
| **Parameters** | Choose confidence level, time horizon, history window, MC simulations. |
| **Simulations** | Click to toggle which analyses to run (multiple supported). |
| **RUN ANALYSIS** | Fetches live data from Yahoo Finance and runs every selected model. |

Results display side-by-side: a rendered chart on the left, a results table on the right.
All charts are automatically styled to match the dark terminal theme.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` for a script | Check `LIBRARY_ROOT` in `app.py`; ensure scripts exist at those paths. |
| SSL errors downloading data | Your existing `set_certifi.py` / `daily_closes.py` already handles this. |
| `ImportError: cannot import name run_*` | Add the `run_*` wrapper function to the relevant script (see §2 above). |
| Port 5000 already in use | Change `port=5000` in `app.py` to e.g. `5001`. |
| Browser shows blank chart | The matplotlib figure returned by your script may be `None`; ensure `run_*` always returns a Figure. |