import datetime as dt
import yfinance as yf
import pandas as pd
import os
import certifi
from pathlib import Path

def download_daily_closes(tickers, days_back):
    # --- SSL FIX ---
    actual_cert_path = certifi.where()
    os.environ["SSL_CERT_FILE"] = actual_cert_path
    os.environ["REQUESTS_CA_BUNDLE"] = actual_cert_path
    if "CURL_CA_BUNDLE" in os.environ:
        del os.environ["CURL_CA_BUNDLE"]

    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days_back)

    print(f"Validating connection using: {actual_cert_path}")

    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        progress=True,
        auto_adjust=True,
        group_by="column"  # helps keep output consistent
    )

    if data.empty:
        return pd.DataFrame()

    # --- robustly extract adjusted close prices into columns per ticker ---
    if isinstance(data.columns, pd.MultiIndex):
        # columns look like: ('Close','AAPL'), ('Close','NVDA'), ...
        closes = data.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        # single ticker case
        closes = data[["Close"]].copy()
        closes.columns = [tickers[0]]

    # Ensure columns are exactly ticker strings (not tuples)
    closes.columns = [str(c) for c in closes.columns]

    # Date from index -> column
    closes = closes.reset_index().rename(columns={"index": "Date"})

    # Format Date nicely for CSV
    if pd.api.types.is_datetime64_any_dtype(closes["Date"]):
        closes["Date"] = closes["Date"].dt.strftime("%Y-%m-%d")

    return closes

if __name__ == "__main__":
    TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA"]  # add as many as you want
    DAYS_BACK = 365

    df_closes = download_daily_closes(TICKERS, DAYS_BACK)

    if not df_closes.empty:
        print("\n--- Success! Data Received ---")
        print(df_closes.tail())

        out_file = Path(r"C:\Users\franc\Downloads\VSC\AdSynthAI\Market_Risk_spglobal\daily_adjusted_closes.csv")

        # Avoid PermissionError if file is open (Excel lock): write a timestamped fallback
        try:
            df_closes.to_csv(out_file, index=False)
            print(f"\nFile saved: {out_file}")
        except PermissionError:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = out_file.with_name(f"{out_file.stem}_{ts}{out_file.suffix}")
            df_closes.to_csv(fallback, index=False)
            print(f"\nCould not overwrite (file likely open). Saved instead: {fallback}")
    else:
        print("\n--- Still failing ---")
        print("Run this in your terminal to refresh certificates:")
        print("python -m pip install --upgrade certifi")
